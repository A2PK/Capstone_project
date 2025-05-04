package repository

import (
	// Add if specific methods need it, BaseRepository methods have it
	"context" // Add if specific methods need it, BaseRepository methods have it
	"errors"  // Import errors package
	core_repo "golang-microservices-boilerplate/pkg/core/repository"
	"golang-microservices-boilerplate/services/user-service/internal/entity"

	"github.com/google/uuid"
	"gorm.io/gorm"
	// "github.com/google/uuid" // Add if needed for specific finders
	// core_types "golang-microservices-boilerplate/pkg/core/types" // Add if needed
)

// FileRepository defines specific persistence operations for File entities.
type FileRepository interface {
	core_repo.BaseRepository[entity.File] // Embed base CRUD
	// FindManyByIDs retrieves multiple File entities based on a list of IDs.
	// Returns a slice of found files and an error.
	FindManyByIDs(ctx context.Context, ids []uuid.UUID) ([]*entity.File, error)
	// Add other specific finders if needed, e.g.:
	// FindByServiceInternalID(ctx context.Context, serviceID string) (*entity.File, error)
}

// gormFileRepository implements FileRepository using GORM.
type gormFileRepository struct {
	*core_repo.GormBaseRepository[entity.File]
}

// NewFileRepository creates a new FileRepository.
func NewFileRepository(db *gorm.DB) FileRepository {
	return &gormFileRepository{
		GormBaseRepository: core_repo.NewGormBaseRepository[entity.File](db),
	}
}

// --- Implement FileRepository Specific Methods ---

// FindManyByIDs retrieves multiple File entities based on a list of IDs.
func (r *gormFileRepository) FindManyByIDs(ctx context.Context, ids []uuid.UUID) ([]*entity.File, error) {
	var files []*entity.File
	if len(ids) == 0 {
		return files, nil // Return empty slice if no IDs provided
	}

	// Use Where "id IN (?)" to find matching records
	// Note: GORM handles potential DB limits on the number of parameters in IN clause
	// depending on the underlying database driver.
	if err := r.DB.WithContext(ctx).Where("id IN (?)", ids).Find(&files).Error; err != nil {
		// Don't return gorm.ErrRecordNotFound as an error in this case,
		// just return the (potentially empty) slice found.
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return files, nil
		}
		return nil, err // Return other database errors
	}

	return files, nil
}

/*
// Example specific finder:
func (r *gormFileRepository) FindByServiceInternalID(ctx context.Context, serviceID string) (*entity.File, error) {
	// ... implementation ...
}
*/
