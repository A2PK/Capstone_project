package repository

import (
	"context"
	"errors"
	"fmt"

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
// If a data point with the same MonitoringTime, StationID, and Source already exists,
// this method retrieves the existing data point's ID and timestamps, updates the input pointer,
// and returns nil error.
// NOTE: This error checking is specific to PostgreSQL (pgconn.PgError, code 23505).
func (r *gormDataPointRepository) Create(ctx context.Context, dataPoint *entity.DataPoint) error { // Return type changed back to error
	err := r.DB.WithContext(ctx).Create(dataPoint).Error
	if err != nil {
		var pgErr *pgconn.PgError
		if errors.As(err, &pgErr) && pgErr.Code == "23505" { // 23505 is unique_violation for PostgreSQL
			// Duplicate found. Query for the existing record to get its ID and timestamps.
			var existingDataPoint entity.DataPoint
			findErr := r.DB.WithContext(ctx).
				Select("id", "created_at", "updated_at"). // Select only necessary fields
				Where("monitoring_time = ? AND station_id = ? AND source = ?",
					dataPoint.MonitoringTime, dataPoint.StationID, dataPoint.Source).
				First(&existingDataPoint).Error

			if findErr != nil {
				// Handle error during the find operation
				return fmt.Errorf("failed to retrieve existing data point after duplicate error: %w", findErr)
			}
			// Update the original pointer with the ID and timestamps from the existing record
			dataPoint.ID = existingDataPoint.ID
			dataPoint.CreatedAt = existingDataPoint.CreatedAt
			dataPoint.UpdatedAt = existingDataPoint.UpdatedAt
			return nil // Indicate success (idempotent operation)
		}
		// It's some other error
		return err
	}
	// Success - the dataPoint passed in now has its ID populated by GORM
	return nil
}

// CreateMany overrides the base repository's CreateMany method to ensure idempotency
// using ON CONFLICT DO NOTHING. It attempts to insert all data points. If a data point
// already exists (based on the unique constraint idx_datapoint_time_station_source),
// the insertion for that specific data point is skipped. Afterwards, it queries and
// returns all data points matching the unique keys of the input slice.
// NOTE: This approach uses one batch insert (with conflict handling) and one select query.
func (r *gormDataPointRepository) CreateMany(ctx context.Context, dataPoints []*entity.DataPoint) ([]*entity.DataPoint, error) {
	if len(dataPoints) == 0 {
		return []*entity.DataPoint{}, nil
	}

	// Attempt to insert all data points, ignoring conflicts on the unique index
	err := r.DB.WithContext(ctx).
		Clauses(clause.OnConflict{
			Columns:   []clause.Column{{Name: "monitoring_time"}, {Name: "station_id"}, {Name: "source"}},
			DoNothing: true,
		}).
		Create(&dataPoints).Error

	// Check for errors other than constraint violations (which are handled by DoNothing)
	if err != nil {
		return nil, fmt.Errorf("error during batch create data points: %w", err)
	}

	// Now, query for all data points matching the unique keys of the input slice
	// to get both the newly inserted and pre-existing ones with their IDs.
	query := r.DB.WithContext(ctx)
	if len(dataPoints) > 0 {
		// Start with the first condition using Where
		firstDp := dataPoints[0]
		query = query.Where("monitoring_time = ? AND station_id = ? AND source = ?", firstDp.MonitoringTime, firstDp.StationID, firstDp.Source)

		// Add subsequent conditions using Or
		for i := 1; i < len(dataPoints); i++ {
			dp := dataPoints[i]
			query = query.Or("monitoring_time = ? AND station_id = ? AND source = ?", dp.MonitoringTime, dp.StationID, dp.Source)
		}
	} else {
		// If there were no input data points, return an empty slice immediately
		return []*entity.DataPoint{}, nil
	}

	var resultDataPoints []*entity.DataPoint
	// Find the matching data points
	findErr := query.Find(&resultDataPoints).Error // Removed .Preload("Features") for debugging, add back if needed
	if findErr != nil {
		return nil, fmt.Errorf("failed to retrieve data points after batch create: %w", findErr)
	}

	// All data points processed successfully (inserted or ignored), and then retrieved.
	// Note: Features are NOT preloaded in this version. If needed, they must be loaded separately.
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
	result := query.Preload("Features").Find(&dataPoints)

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
