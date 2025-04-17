package repository

import (
	"context"
	"errors"
	"fmt"

	coreRepo "golang-microservices-boilerplate/pkg/core/repository"
	"golang-microservices-boilerplate/pkg/core/types"
	"golang-microservices-boilerplate/services/model-service/internal/entity"

	// Keep uuid import
	"github.com/google/uuid"
	"gorm.io/gorm"
)

// ModelRepository defines the interface for AIModel operations.
// It embeds the BaseRepository for common CRUD operations and adds model-specific methods.
type ModelRepository interface {
	// Embed base CRUD methods for *entity.AIModel
	coreRepo.BaseRepository[entity.AIModel]

	// --- Custom Model Repository Methods --- //

	// FindByName retrieves an AIModel by its unique name.
	// If includeDeleted is true, it includes soft-deleted records.
	// Should return a specific error (e.g., types.ErrNotFound) if not found.
	FindByName(ctx context.Context, name string, includeDeleted bool) (*entity.AIModel, error)

	// ListByStationID retrieves models for a specific station.
	ListByStationID(ctx context.Context, stationID uuid.UUID, opts types.FilterOptions) ([]*entity.AIModel, int64, error)
}

// --- GORM Implementation --- //

// Ensure gormModelRepository implements ModelRepository
var _ ModelRepository = (*gormModelRepository)(nil)

// gormModelRepository implements the ModelRepository interface using GORM.
type gormModelRepository struct {
	// Embed the base GORM repository implementation
	*coreRepo.GormBaseRepository[entity.AIModel]
}

// NewGormModelRepository creates a new GORM model repository instance.
// It expects the raw *gorm.DB connection.
func NewGormModelRepository(db *gorm.DB) ModelRepository {
	// Create the base repository implementation using the provided db connection
	baseRepo := coreRepo.NewGormBaseRepository[entity.AIModel](db)
	return &gormModelRepository{
		GormBaseRepository: baseRepo,
	}
}

// FindByName implements the custom FindByName method.
func (r *gormModelRepository) FindByName(ctx context.Context, name string, includeDeleted bool) (*entity.AIModel, error) {
	var model entity.AIModel
	db := r.DB.WithContext(ctx) // Use the embedded DB connection from GormBaseRepository
	if includeDeleted {
		db = db.Unscoped()
	}
	result := db.Where("name = ?", name).First(&model)
	if result.Error != nil {
		if errors.Is(result.Error, gorm.ErrRecordNotFound) {
			// Use standard error type
			return nil, fmt.Errorf("%w: model with name '%s'", types.ErrNotFound, name)
		}
		// Use standard error type
		return nil, fmt.Errorf("%w: %v", types.ErrDatabase, result.Error)
	}
	return &model, nil
}

// ListByStationID implementation
func (r *gormModelRepository) ListByStationID(ctx context.Context, stationID uuid.UUID, opts types.FilterOptions) ([]*entity.AIModel, int64, error) {
	var models []*entity.AIModel
	var totalCount int64
	db := r.DB.WithContext(ctx).Model(&entity.AIModel{}).Where("station_id = ?", &stationID) // Filter by station_id

	// Count logic (adapt from coreRepo.FindAll or similar)
	countDb := db
	if !opts.IncludeDeleted {
		countDb = countDb.Where("deleted_at IS NULL")
	}
	// TODO: Apply other filters from opts.Filters if necessary for count
	err := countDb.Count(&totalCount).Error
	if err != nil {
		return nil, 0, fmt.Errorf("%w: counting models by station: %v", types.ErrDatabase, err)
	}
	if totalCount == 0 {
		return models, 0, nil // Return empty slice and 0 count
	}

	// Query logic (adapt from coreRepo.FindAll or similar)
	// Assuming coreRepo has or we add a helper ApplyFilterOptions
	// queryDb := coreRepo.ApplyFilterOptions(db, opts) // Need ApplyFilterOptions helper
	queryDb := r.ApplyFilterOptions(db, opts) // Assuming ApplyFilterOptions is available via embedded repo or defined locally
	err = queryDb.Find(&models).Error
	if err != nil {
		return nil, 0, fmt.Errorf("%w: finding models by station: %v", types.ErrDatabase, err)
	}

	return models, totalCount, nil
}

// ApplyFilterOptions is needed by ListByStationID (might exist in embedded repo or needs adding here)
// Placeholder implementation - assumes it exists in coreRepo.GormBaseRepository or needs to be added/adapted
func (r *gormModelRepository) ApplyFilterOptions(tx *gorm.DB, opts types.FilterOptions) *gorm.DB {
	// This should ideally reuse the logic from coreRepo.GormBaseRepository
	// If GormBaseRepository doesn't expose it, duplicate or refactor it here/core.
	if !opts.IncludeDeleted {
		tx = tx.Where("deleted_at IS NULL")
	}
	if opts.SortBy != "" {
		direction := "ASC"
		if opts.SortDesc {
			direction = "DESC"
		}
		tx = tx.Order(fmt.Sprintf("%s %s", opts.SortBy, direction))
	} else {
		tx = tx.Order("created_at DESC") // Default sort
	}
	if opts.Limit > 0 {
		tx = tx.Limit(opts.Limit)
	}
	if opts.Offset >= 0 {
		tx = tx.Offset(opts.Offset)
	}
	// TODO: Apply opts.Filters map
	return tx
}

// Note: Methods like Create, Update, Delete, FindByID, FindAll, Count, CreateMany, UpdateMany, DeleteMany
// are inherited from the embedded *coreRepo.GormBaseRepository[entity.AIModel].
