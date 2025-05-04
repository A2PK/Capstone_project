package repository

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgconn"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"

	"golang-microservices-boilerplate/pkg/core/repository"
	"golang-microservices-boilerplate/pkg/core/types"
	"golang-microservices-boilerplate/services/water-quality-service/internal/entity"
)

// DataPointRepository defines the interface for data point operations
type DataPointRepository interface {
	repository.BaseRepository[entity.DataPoint]
	// Overridden Create to handle duplicates specifically for DataPoints, updating the input pointer.
	Create(ctx context.Context, dataPoint *entity.DataPoint) error // Return type changed back to error
	// Overridden CreateMany to handle duplicates specifically for DataPoints
	CreateMany(ctx context.Context, dataPoints []*entity.DataPoint) ([]*entity.DataPoint, error)
	// FindByStationID retrieves data points for a specific station with pagination, preloading indicators
	FindByStationID(ctx context.Context, stationID uuid.UUID, opts types.FilterOptions) (*types.PaginationResult[entity.DataPoint], error)
}

// gormDataPointRepository implements the DataPointRepository interface using GORM
type gormDataPointRepository struct {
	*repository.GormBaseRepository[entity.DataPoint]
}

// NewGormDataPointRepository creates a new GORM data point repository instance
func NewGormDataPointRepository(db *gorm.DB) DataPointRepository {
	baseRepo := repository.NewGormBaseRepository[entity.DataPoint](db)
	return &gormDataPointRepository{baseRepo}
}

// Create overrides the base repository's Create method to handle unique constraint violations gracefully.
// If a data point with the same MonitoringTime, StationID, ObservationType, and WQI already exists,
// this method updates the existing record with the new data.
// NOTE: This error checking is specific to PostgreSQL (pgconn.PgError, code 23505).
func (r *gormDataPointRepository) Create(ctx context.Context, dataPoint *entity.DataPoint) error {
	// Begin a transaction for the operation
	tx := r.DB.WithContext(ctx).Begin()
	if tx.Error != nil {
		return fmt.Errorf("failed to begin transaction for create: %w", tx.Error)
	}

	// Setup deferred rollback that will be ignored if we commit successfully
	defer func() {
		if r := recover(); r != nil {
			tx.Rollback()
			panic(r) // re-throw panic after rollback
		}
	}()

	// Try to create the data point
	err := tx.Create(dataPoint).Error
	if err != nil {
		var pgErr *pgconn.PgError
		if errors.As(err, &pgErr) && pgErr.Code == "23505" { // 23505 is unique_violation for PostgreSQL
			// Duplicate found based on idx_datapoint_time_station_observation_wqi
			// Get the existing record ID
			var existingDataPoint entity.DataPoint
			query := tx.
				Select("id").
				Where("monitoring_time = ? AND station_id = ? AND observation_type = ?",
					dataPoint.MonitoringTime, dataPoint.StationID, dataPoint.ObservationType)

			// Add WQI condition handling NULL values correctly
			if dataPoint.WQI == nil {
				query = query.Where("wqi IS NULL")
			} else {
				query = query.Where("wqi = ?", *dataPoint.WQI)
			}

			findErr := query.First(&existingDataPoint).Error

			if findErr != nil {
				tx.Rollback()
				return fmt.Errorf("failed to retrieve existing data point after duplicate error: %w", findErr)
			}

			// Update the existing record with new data, preserving the ID
			dataPoint.ID = existingDataPoint.ID
			updateErr := tx.Model(&entity.DataPoint{}).
				Where("id = ?", existingDataPoint.ID).
				Updates(map[string]interface{}{
					"source":                dataPoint.Source,
					"data_source_schema_id": dataPoint.DataSourceSchemaID,
					"features":              dataPoint.Features,
					"updated_at":            time.Now(),
				}).Error

			if updateErr != nil {
				tx.Rollback()
				return fmt.Errorf("failed to update existing data point: %w", updateErr)
			}

			// Since we updated an existing record, commit and exit early
			if err := tx.Commit().Error; err != nil {
				return fmt.Errorf("failed to commit transaction after updating duplicate: %w", err)
			}
			return nil
		}
		// It's some other error
		tx.Rollback()
		return err
	}

	// Check for threshold violations if this is an actual observation
	if dataPoint.ObservationType == entity.Actual {
		// 1. Collect names of numeric features
		featureNamesToCheck := make(map[string]struct{})
		numericFeatures := make(map[string]float64)

		for _, feature := range dataPoint.Features {
			if feature.Value != nil {
				lowerName := strings.ToLower(feature.Name)
				featureNamesToCheck[lowerName] = struct{}{}
				numericFeatures[lowerName] = *feature.Value
			}
		}

		if len(featureNamesToCheck) > 0 {
			namesList := make([]string, 0, len(featureNamesToCheck))
			for name := range featureNamesToCheck {
				namesList = append(namesList, name)
			}

			// 2. Fetch Thresholds directly using the transaction DB handle
			var thresholdConfigs []*entity.ThresholdConfig
			if err := tx.Where("LOWER(element_name) IN (?)", namesList).Find(&thresholdConfigs).Error; err != nil {
				fmt.Printf("ERROR: Failed to fetch threshold configs for data point: %v\n", err)
			} else {
				thresholdMap := make(map[string]*entity.ThresholdConfig)
				for _, cfg := range thresholdConfigs {
					thresholdMap[strings.ToLower(cfg.ElementName)] = cfg
				}

				// 3. Perform Checks and Collect Violations
				var violations []string
				violatedFeatures := make(map[string]struct{})

				for lowerName, value := range numericFeatures {
					if config, ok := thresholdMap[lowerName]; ok {
						if value < config.MinValue || value > config.MaxValue {
							// Get station name using JOIN
							var stationName string
							err := tx.Table("stations").
								Select("name").
								Where("id = ?", dataPoint.StationID).
								Scan(&stationName).Error
							if err != nil {
								fmt.Printf("ERROR: Failed to fetch station name: %v\n", err)
								stationName = fmt.Sprintf("Trạm %s", dataPoint.StationID)
							}

							violationMsg := fmt.Sprintf("Trạm %s (%s): Giá trị '%s' là %.2f vượt ngưỡng cho phép (%.2f - %.2f) vào lúc %s\n",
								stationName, dataPoint.StationID, config.ElementName, value, config.MinValue, config.MaxValue,
								dataPoint.MonitoringTime.Format(time.RFC3339))
							violations = append(violations, violationMsg)
							violatedFeatures[config.ElementName] = struct{}{}
						}
					}
				}

				// 4. If violations found, fetch ALL user IDs and create notifications
				if len(violations) > 0 {
					var userIDs []uuid.UUID
					if err := tx.Table("users").Select("id").Find(&userIDs).Error; err != nil {
						fmt.Printf("ERROR: Failed to fetch user IDs for notification: %v\n", err)
					} else if len(userIDs) > 0 {

						title := fmt.Sprintf("Cảnh báo: %d trạm vượt ngưỡng", len(violatedFeatures))
						description := fmt.Sprintf("Phát hiện %d giá trị vượt ngưỡng tại các trạm:\n%s",
							len(violations), strings.Join(violations, "\n"))

						// Create notification structs for each user
						notificationsToInsert := make([]*entity.NotificationForHook, 0, len(userIDs))
						for _, userID := range userIDs {
							notificationsToInsert = append(notificationsToInsert, &entity.NotificationForHook{
								ID:          uuid.New(),
								UserID:      userID,
								Title:       title,
								Description: description,
								Read:        false,
							})
						}

						// Bulk insert all notifications in a single operation
						if len(notificationsToInsert) > 0 {
							fmt.Printf("Batch inserting %d notifications for threshold violations\n", len(notificationsToInsert))
							if err := tx.Create(&notificationsToInsert).Error; err != nil {
								fmt.Printf("ERROR: Failed to insert batch notifications: %v\n", err)
								// Don't fail the data point creation
							}
						}
					}
				}
			}
		}
	}

	// Commit the transaction
	if err := tx.Commit().Error; err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	// Success - the dataPoint now has its ID populated
	return nil
}

// CreateMany overrides the base repository's CreateMany method to handle batch insertions
// with the new unique constraint on MonitoringTime, StationID, ObservationType, and WQI.
// When conflicts occur, it updates the existing records with the new data, properly handling JSONB fields.
func (r *gormDataPointRepository) CreateMany(ctx context.Context, dataPoints []*entity.DataPoint) ([]*entity.DataPoint, error) {
	if len(dataPoints) == 0 {
		return []*entity.DataPoint{}, nil
	}

	// Begin a transaction for the batch operation
	tx := r.DB.WithContext(ctx).Begin()
	if tx.Error != nil {
		return nil, fmt.Errorf("failed to begin transaction for batch create: %w", tx.Error)
	}

	// Setup deferred rollback that will be ignored if we commit successfully
	defer func() {
		if r := recover(); r != nil {
			tx.Rollback()
			panic(r) // re-throw panic after rollback
		}
	}()

	// Deduplicate data points based on the unique key to avoid PostgreSQL errors
	// when trying to update the same row multiple times
	type uniqueKey struct {
		MonitoringTime  time.Time
		StationID       uuid.UUID
		ObservationType entity.ObservationType
		WQI             *float64
	}

	// Use a map to handle duplicates - last one wins
	uniqueDataPoints := make(map[uniqueKey]*entity.DataPoint)
	for _, dp := range dataPoints {
		key := uniqueKey{
			MonitoringTime:  dp.MonitoringTime,
			StationID:       dp.StationID,
			ObservationType: dp.ObservationType,
			WQI:             dp.WQI,
		}
		// This automatically handles duplicates by overwriting
		uniqueDataPoints[key] = dp
	}

	// Convert map back to slice for processing
	dedupedDataPoints := make([]*entity.DataPoint, 0, len(uniqueDataPoints))
	for _, dp := range uniqueDataPoints {
		dedupedDataPoints = append(dedupedDataPoints, dp)
	}

	// Collect all the data points that need threshold checking (actual observations only)
	var actualDataPoints []*entity.DataPoint
	for _, dp := range dedupedDataPoints {
		if dp.ObservationType == entity.Actual {
			actualDataPoints = append(actualDataPoints, dp)
		}
	}

	// Insert data points with ON CONFLICT UPDATE
	// This handles the JSONB serialization properly and is much faster than individual updates
	err := tx.Clauses(clause.OnConflict{
		Columns: []clause.Column{
			{Name: "monitoring_time"},
			{Name: "station_id"},
			{Name: "observation_type"},
			{Name: "wqi"},
		},
		DoUpdates: clause.AssignmentColumns([]string{
			"source",
			"data_source_schema_id",
			"features",
			"updated_at",
		}),
	}).Create(&dedupedDataPoints).Error

	if err != nil {
		tx.Rollback()
		return nil, fmt.Errorf("error during batch create/update data points: %w", err)
	}

	// Query for all inserted/updated data points to get their IDs
	var resultDataPoints []*entity.DataPoint

	// Build query conditions for finding all processed data points
	var conditions []string
	var args []interface{}
	for _, dp := range dedupedDataPoints {
		cond := "(monitoring_time = ? AND station_id = ? AND observation_type = ? AND "
		args = append(args, dp.MonitoringTime, dp.StationID, dp.ObservationType)

		if dp.WQI == nil {
			cond += "wqi IS NULL)"
		} else {
			cond += "wqi = ?)"
			args = append(args, *dp.WQI)
		}

		conditions = append(conditions, cond)
	}

	if len(conditions) > 0 {
		query := tx.Where(strings.Join(conditions, " OR "), args...)
		findErr := query.Find(&resultDataPoints).Error
		if findErr != nil {
			tx.Rollback()
			return nil, fmt.Errorf("failed to retrieve data points after batch operation: %w", findErr)
		}
	}

	// If we have actual data points, perform batch threshold checks
	if len(actualDataPoints) > 0 {
		// 1. Collect unique feature names across all data points
		featureNamesToCheck := make(map[string]struct{})

		for _, dp := range actualDataPoints {
			for _, feature := range dp.Features {
				if feature.Value != nil {
					lowerName := strings.ToLower(feature.Name)
					featureNamesToCheck[lowerName] = struct{}{}
				}
			}
		}

		// 2. Fetch all relevant threshold configs in a single query
		var thresholdConfigs []*entity.ThresholdConfig
		namesList := make([]string, 0, len(featureNamesToCheck))
		for name := range featureNamesToCheck {
			namesList = append(namesList, name)
		}

		if len(namesList) > 0 {
			if err := tx.Where("LOWER(element_name) IN (?)", namesList).Find(&thresholdConfigs).Error; err != nil {
				// Log error but continue - don't fail the insert operation
				fmt.Printf("ERROR: Failed to fetch threshold configs for batch: %v\n", err)
			} else {
				// Map for quick lookup of threshold configs by name
				thresholdMap := make(map[string]*entity.ThresholdConfig)
				for _, cfg := range thresholdConfigs {
					thresholdMap[strings.ToLower(cfg.ElementName)] = cfg
				}

				// 3. Check all data points against thresholds and collect violations by user
				violationsByUser := make(map[uuid.UUID][]string)
				violatedStations := make(map[uuid.UUID]map[string]struct{}) // Map station IDs to feature names

				for _, dp := range actualDataPoints {
					dpViolations := []string{}

					for _, feature := range dp.Features {
						if feature.Value != nil {
							lowerName := strings.ToLower(feature.Name)
							if config, ok := thresholdMap[lowerName]; ok {
								value := *feature.Value
								if value < config.MinValue || value > config.MaxValue {
									// Get station name using JOIN
									var stationName string
									err := tx.Table("stations").
										Select("name").
										Where("id = ?", dp.StationID).
										Scan(&stationName).Error
									if err != nil {
										fmt.Printf("ERROR: Failed to fetch station name: %v\n", err)
										stationName = fmt.Sprintf("Trạm %s", dp.StationID)
									}

									violationMsg := fmt.Sprintf("Trạm %s (%s): Giá trị '%s' là %.2f vượt ngưỡng cho phép (%.2f - %.2f) vào lúc %s\n",
										stationName, dp.StationID, config.ElementName, value, config.MinValue, config.MaxValue,
										dp.MonitoringTime.Format(time.RFC3339))
									dpViolations = append(dpViolations, violationMsg)

									// Record which station had violations
									if _, ok := violatedStations[dp.StationID]; !ok {
										violatedStations[dp.StationID] = make(map[string]struct{})
									}
									violatedStations[dp.StationID][config.ElementName] = struct{}{}
								}
							}
						}
					}

					// If this data point has violations, add them to the by-user map
					if len(dpViolations) > 0 {
						// Collect all violations first, then fetch users once at the end
						for _, violationMsg := range dpViolations {
							if _, exists := violationsByUser[uuid.Nil]; !exists {
								violationsByUser[uuid.Nil] = []string{}
							}
							violationsByUser[uuid.Nil] = append(violationsByUser[uuid.Nil], violationMsg)
						}
					}
				}

				// 4. If we have any violations, fetch users and create notifications
				if _, hasViolations := violationsByUser[uuid.Nil]; hasViolations {
					violations := violationsByUser[uuid.Nil]
					var userIDs []uuid.UUID

					if err := tx.Table("users").Select("id").Find(&userIDs).Error; err != nil {
						fmt.Printf("ERROR: Failed to fetch user IDs for batch notification: %v\n", err)
					} else if len(userIDs) > 0 {
						title := fmt.Sprintf("Cảnh báo: %d trạm vượt ngưỡng", len(violatedStations))
						description := fmt.Sprintf("Phát hiện %d giá trị vượt ngưỡng tại các trạm:\n%s",
							len(violations), strings.Join(violations, "\n"))

						// Create notification structs for each user
						notificationsToInsert := make([]*entity.NotificationForHook, 0, len(userIDs))
						for _, userID := range userIDs {
							notificationsToInsert = append(notificationsToInsert, &entity.NotificationForHook{
								ID:          uuid.New(),
								UserID:      userID,
								Title:       title,
								Description: description,
								Read:        false,
							})
						}

						// Bulk insert all notifications in a single operation
						if len(notificationsToInsert) > 0 {
							fmt.Printf("Batch inserting %d notifications for threshold violations\n", len(notificationsToInsert))
							if err := tx.Create(&notificationsToInsert).Error; err != nil {
								fmt.Printf("ERROR: Failed to insert batch notifications: %v\n", err)
								// Don't fail the data point creation
							}
						}
					}
				}
			}
		}
	}

	// Commit the transaction
	if err := tx.Commit().Error; err != nil {
		return nil, fmt.Errorf("failed to commit transaction for batch create: %w", err)
	}

	return resultDataPoints, nil
}

// FindByStationID retrieves data points for a specific station with pagination, preloading associated indicators.
func (r *gormDataPointRepository) FindByStationID(ctx context.Context, stationID uuid.UUID, opts types.FilterOptions) (*types.PaginationResult[entity.DataPoint], error) {
	var dataPoints []*entity.DataPoint
	var total int64

	// Base query with context and model
	query := r.DB.WithContext(ctx).Model(&entity.DataPoint{})

	// Apply filter for station ID
	query = query.Where("station_id = ?", stationID)

	if len(opts.Filters) > 0 {
		query = query.Where(opts.Filters)
	}

	// Count total matching records before applying pagination
	if err := query.Count(&total).Error; err != nil {
		// Handle count error, maybe log it
		return nil, err
	}

	// Apply pagination (Offset, Limit)
	// Ensure opts.Limit > 0 to avoid issues
	if opts.Limit > 0 {
		query = query.Limit(opts.Limit).Offset(opts.Offset)
	} else {
		// Default limit if not provided or invalid?
		// Or return error?
		// For now, let GORM handle Offset(0) Limit(0)
		query = query.Offset(opts.Offset)
	}

	// Apply sorting if specified in opts
	if opts.SortBy != "" {
		order := opts.SortBy
		if opts.SortDesc {
			order += " DESC"
		}
		query = query.Order(order)
	} else {
		// Default sort order if needed, e.g., by timestamp
		query = query.Order("created_at ASC")
	}

	// Add Preload for Features before finding the results
	// GORM handles finding into a slice of pointers
	result := query.Find(&dataPoints)

	if result.Error != nil {
		return nil, result.Error // Handle potential errors, e.g., logging
	}

	// Construct the PaginationResult using the correct field names from types.common.go
	return &types.PaginationResult[entity.DataPoint]{
		Items:      dataPoints,
		TotalItems: total,
		Limit:      opts.Limit,
		Offset:     opts.Offset,
	}, nil
}
