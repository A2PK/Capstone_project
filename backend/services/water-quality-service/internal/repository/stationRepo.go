package repository

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5/pgconn" // Import for PostgreSQL error checking
	"gorm.io/gorm"
	"gorm.io/gorm/clause" // Import for OnConflict

	"golang-microservices-boilerplate/pkg/core/repository"
	"golang-microservices-boilerplate/services/water-quality-service/internal/entity"
)

// StationRepository defines the interface for station operations
type StationRepository interface {
	repository.BaseRepository[entity.Station]
	// FindByName finds a station by its name
	FindByName(ctx context.Context, name string) (*entity.Station, error)
	// Overridden Create to handle duplicates specifically for Stations, updating the input pointer.
	Create(ctx context.Context, station *entity.Station) error // Return type changed back to error
	// Overridden CreateMany to handle duplicates specifically for Stations
	CreateMany(ctx context.Context, stations []*entity.Station) ([]*entity.Station, error)
}

// gormStationRepository implements the StationRepository interface using GORM
type gormStationRepository struct {
	*repository.GormBaseRepository[entity.Station]
}

// NewGormStationRepository creates a new GORM station repository instance
func NewGormStationRepository(db *gorm.DB) StationRepository {
	baseRepo := repository.NewGormBaseRepository[entity.Station](db)
	return &gormStationRepository{baseRepo}
}

// Create overrides the base repository's Create method to handle unique constraint violations gracefully.
// If a station with the same Name, Latitude, and Longitude already exists (violating idx_name_lat_lon),
// this method retrieves the existing station's ID and timestamps, updates the input pointer,
// and returns nil error.
// NOTE: This error checking is specific to PostgreSQL (pgconn.PgError, code 23505).
func (r *gormStationRepository) Create(ctx context.Context, station *entity.Station) error { // Return type changed back to error
	err := r.DB.WithContext(ctx).Create(station).Error
	if err != nil {
		var pgErr *pgconn.PgError
		if errors.As(err, &pgErr) && pgErr.Code == "23505" { // 23505 is unique_violation for PostgreSQL
			// Duplicate found. Query for the existing record to get its ID and timestamps.
			var existingStation entity.Station
			findErr := r.DB.WithContext(ctx).
				Select("id", "created_at", "updated_at"). // Select only necessary fields
				Where("name = ? AND latitude = ? AND longitude = ?",
					station.Name, station.Latitude, station.Longitude).
				First(&existingStation).Error

			if findErr != nil {
				// Handle error during the find operation
				return fmt.Errorf("failed to retrieve existing station after duplicate error: %w", findErr)
			}
			// Update the original pointer with the ID and timestamps from the existing record
			station.ID = existingStation.ID
			station.CreatedAt = existingStation.CreatedAt
			station.UpdatedAt = existingStation.UpdatedAt
			return nil // Indicate success (idempotent operation)
		}
		// It's some other error
		return err
	}
	// Success - the station passed in now has its ID populated by GORM
	return nil
}

// CreateMany overrides the base repository's CreateMany method to ensure idempotency
// using ON CONFLICT DO NOTHING. It attempts to insert all stations. If a station
// already exists (based on the unique constraint idx_name_lat_lon), the insertion
// for that specific station is skipped. Afterwards, it queries and returns all
// stations matching the unique keys of the input slice.
// NOTE: This approach uses one batch insert (with conflict handling) and one select query.
func (r *gormStationRepository) CreateMany(ctx context.Context, stations []*entity.Station) ([]*entity.Station, error) {
	if len(stations) == 0 {
		return []*entity.Station{}, nil
	}

	// Attempt to insert all stations, ignoring conflicts on the unique index
	err := r.DB.WithContext(ctx).
		Clauses(clause.OnConflict{
			Columns:   []clause.Column{{Name: "name"}, {Name: "latitude"}, {Name: "longitude"}},
			DoNothing: true,
		}).
		Create(&stations).Error

	// Check for errors other than constraint violations (which are handled by DoNothing)
	if err != nil {
		// Note: Depending on the GORM version and database driver, certain errors
		// during batch operations might still be reported. Handle non-conflict errors.
		return nil, fmt.Errorf("error during batch create stations: %w", err)
	}

	// Now, query for all stations matching the unique keys of the input slice
	// to get both the newly inserted and pre-existing ones with their IDs.
	query := r.DB.WithContext(ctx)
	if len(stations) > 0 {
		// Start with the first condition using Where
		first := stations[0]
		query = query.Where("name = ? AND latitude = ? AND longitude = ?", first.Name, first.Latitude, first.Longitude)

		// Add subsequent conditions using Or
		for i := 1; i < len(stations); i++ {
			s := stations[i]
			query = query.Or("name = ? AND latitude = ? AND longitude = ?", s.Name, s.Latitude, s.Longitude)
		}
	} else {
		// Should not happen based on the initial check, but good practice.
		return []*entity.Station{}, nil
	}

	var resultStations []*entity.Station
	findErr := query.Find(&resultStations).Error
	if findErr != nil {
		return nil, fmt.Errorf("failed to retrieve stations after batch create: %w", findErr)
	}

	// All stations processed successfully (inserted or ignored), and then retrieved.
	return resultStations, nil
}

// FindByName finds a station by its name
func (r *gormStationRepository) FindByName(ctx context.Context, name string) (*entity.Station, error) {
	var station entity.Station
	result := r.DB.WithContext(ctx).Where("name = ?", name).First(&station)
	if result.Error != nil {
		if errors.Is(result.Error, gorm.ErrRecordNotFound) {
			// Consider returning a specific error type from your domain/app layer instead of a plain string
			// e.g., return nil, apperrors.ErrStationNotFound
			return nil, errors.New("station not found")
		}
		return nil, result.Error
	}
	return &station, nil
}

// Note: If you use CreateMany or BatchInsert for Stations, you would need to
// override that method here as well with similar duplicate-checking logic.
// Handling batch errors can be more complex depending on GORM/driver behavior.
