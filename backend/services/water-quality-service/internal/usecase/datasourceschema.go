package usecase

import (
	"context"

	coreLogger "golang-microservices-boilerplate/pkg/core/logger"
	coreUsecase "golang-microservices-boilerplate/pkg/core/usecase"
	"golang-microservices-boilerplate/services/water-quality-service/internal/entity"
	"golang-microservices-boilerplate/services/water-quality-service/internal/repository"
)

// DataSourceSchemaUsecase defines the interface for data source schema related business logic
type DataSourceSchemaUsecase interface {
	// Embed the core use case
	coreUsecase.BaseUseCase[entity.DataSourceSchema]
	// FindByNameAndSource finds a schema by its name and source
	FindByNameAndSource(ctx context.Context, name, sourceIdentifier string, sourceType entity.SourceType) (*entity.DataSourceSchema, error)
}

// dataSourceSchemaUseCaseImpl implements the DataSourceSchemaUsecase interface
type dataSourceSchemaUseCaseImpl struct {
	// Embed the core use case implementation
	*coreUsecase.BaseUseCaseImpl[entity.DataSourceSchema]
	repo   repository.DataSourceSchemaRepository
	logger coreLogger.Logger
}

// NewDataSourceSchemaUsecase creates a new instance of DataSourceSchemaUsecase
func NewDataSourceSchemaUsecase(
	repo repository.DataSourceSchemaRepository,
	logger coreLogger.Logger,
) DataSourceSchemaUsecase {
	baseUseCase := coreUsecase.NewBaseUseCase(repo, logger)
	return &dataSourceSchemaUseCaseImpl{
		BaseUseCaseImpl: baseUseCase,
		repo:            repo,
		logger:          logger,
	}
}

// FindByNameAndSource finds a schema by its name and source
func (uc *dataSourceSchemaUseCaseImpl) FindByNameAndSource(ctx context.Context, name, sourceIdentifier string, sourceType entity.SourceType) (*entity.DataSourceSchema, error) {
	uc.logger.Info("Finding schema by name and source", "name", name, "sourceIdentifier", sourceIdentifier, "sourceType", sourceType)
	schema, err := uc.repo.FindByNameAndSource(ctx, name, sourceIdentifier, sourceType)
	if err != nil {
		uc.logger.Error("Failed to find schema by name and source", "name", name, "sourceIdentifier", sourceIdentifier, "sourceType", sourceType, "error", err)
		return nil, coreUsecase.NewUseCaseError(coreUsecase.ErrInternal, "failed to find schema")
	}
	if schema == nil {
		// No error from repository but nil schema means not found
		return nil, coreUsecase.NewUseCaseError(coreUsecase.ErrNotFound, "schema not found")
	}
	return schema, nil
}
