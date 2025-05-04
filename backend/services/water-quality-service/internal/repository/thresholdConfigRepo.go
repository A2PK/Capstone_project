package repository

import (
	"context"
	"strings"

	core_repo "golang-microservices-boilerplate/pkg/core/repository"
	"golang-microservices-boilerplate/services/water-quality-service/internal/entity"

	"gorm.io/gorm"
)

// ThresholdConfigRepository defines specific persistence operations for ThresholdConfig entities.
type ThresholdConfigRepository interface {
	core_repo.BaseRepository[entity.ThresholdConfig] // Embed base CRUD

	// FindByElementNames retrieves multiple ThresholdConfig entities based on a list of element names (case-insensitive).
	// Returns a map where the key is the lowercase element name and the value is the config entity.
	FindByElementNames(ctx context.Context, names []string) (map[string]*entity.ThresholdConfig, error)
}

// gormThresholdConfigRepository implements ThresholdConfigRepository using GORM.
type gormThresholdConfigRepository struct {
	*core_repo.GormBaseRepository[entity.ThresholdConfig]
}

// NewThresholdConfigRepository creates a new ThresholdConfigRepository.
func NewThresholdConfigRepository(db *gorm.DB) ThresholdConfigRepository {
	return &gormThresholdConfigRepository{
		GormBaseRepository: core_repo.NewGormBaseRepository[entity.ThresholdConfig](db),
	}
}

// --- Implement ThresholdConfigRepository Specific Methods ---

// FindByElementNames retrieves multiple ThresholdConfig entities based on a list of element names (case-insensitive).
func (r *gormThresholdConfigRepository) FindByElementNames(ctx context.Context, names []string) (map[string]*entity.ThresholdConfig, error) {
	results := make(map[string]*entity.ThresholdConfig)
	if len(names) == 0 {
		return results, nil // Return empty map if no names provided
	}

	// Normalize names to lowercase for query and map keys
	lowerNames := make([]string, len(names))
	for i, name := range names {
		lowerNames[i] = strings.ToLower(name)
	}

	var configs []*entity.ThresholdConfig
	// Use WHERE LOWER(element_name) IN (?) for case-insensitive matching
	// Note: Ensure the 'element_name' column exists and potentially has an index on its lowercase form for performance.
	if err := r.DB.WithContext(ctx).Where("LOWER(element_name) IN (?)", lowerNames).Find(&configs).Error; err != nil {
		// Don't return gorm.ErrRecordNotFound as an error, just return the empty map.
		if err == gorm.ErrRecordNotFound {
			return results, nil
		}
		return nil, err // Return other database errors
	}

	// Populate the map with lowercase names as keys
	for _, config := range configs {
		results[strings.ToLower(config.ElementName)] = config
	}

	return results, nil
}
