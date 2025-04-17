package repository

import (
	"context"
	"errors"

	"gorm.io/gorm"

	"golang-microservices-boilerplate/pkg/core/repository"
	"golang-microservices-boilerplate/services/water-quality-service/internal/entity"
)

// DataSourceSchemaRepository defines the interface for data source schema operations
type DataSourceSchemaRepository interface {
	repository.BaseRepository[entity.DataSourceSchema]
	// FindByNameAndSource finds a schema by its name and source identifier
	FindByNameAndSource(ctx context.Context, name, sourceIdentifier string, sourceType entity.SourceType) (*entity.DataSourceSchema, error)
}

// gormDataSourceSchemaRepository implements the DataSourceSchemaRepository interface using GORM
type gormDataSourceSchemaRepository struct {
	*repository.GormBaseRepository[entity.DataSourceSchema]
}

// NewGormDataSourceSchemaRepository creates a new GORM data source schema repository instance
func NewGormDataSourceSchemaRepository(db *gorm.DB) DataSourceSchemaRepository {
	baseRepo := repository.NewGormBaseRepository[entity.DataSourceSchema](db)
	return &gormDataSourceSchemaRepository{baseRepo}
}

// FindByNameAndSource finds a schema by its name, source identifier and source type
func (r *gormDataSourceSchemaRepository) FindByNameAndSource(ctx context.Context, name, sourceIdentifier string, sourceType entity.SourceType) (*entity.DataSourceSchema, error) {
	var schema entity.DataSourceSchema
	filter := map[string]interface{}{
		"name":              name,
		"source_identifier": sourceIdentifier,
		"source_type":       sourceType,
	}
	result := r.DB.WithContext(ctx).Where(filter).First(&schema)
	if result.Error != nil {
		if errors.Is(result.Error, gorm.ErrRecordNotFound) {
			// Return nil, nil to indicate not found without an error, allowing creation
			return nil, nil
		}
		return nil, result.Error // Other database error
	}
	return &schema, nil
}
