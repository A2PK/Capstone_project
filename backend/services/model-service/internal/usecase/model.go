package usecase

import (
	"context"
	"errors"
	"fmt"

	// "io" // No longer needed for HandleUpload

	coreLogger "golang-microservices-boilerplate/pkg/core/logger"
	"golang-microservices-boilerplate/pkg/core/types"
	coreUsecase "golang-microservices-boilerplate/pkg/core/usecase"
	"golang-microservices-boilerplate/services/model-service/internal/entity"
	"golang-microservices-boilerplate/services/model-service/internal/repository"

	"github.com/google/uuid"
)

// ModelUseCase defines specific business logic operations for models,
// embedding common CRUD operations from BaseUseCase.
type ModelUseCase interface {
	coreUsecase.BaseUseCase[entity.AIModel] // Embed standard CRUD & Bulk Ops

	// --- Custom Model Operations --- //

	FindModelByName(ctx context.Context, name string) (*entity.AIModel, error)

	// ListModelsByStation retrieves models filtered by a specific station ID.
	ListModelsByStation(ctx context.Context, stationID uuid.UUID, opts types.FilterOptions) (*types.PaginationResult[entity.AIModel], error)
}

// modelUseCase implements the ModelUseCase interface.
type modelUseCase struct {
	*coreUsecase.BaseUseCaseImpl[entity.AIModel]
	modelRepo repository.ModelRepository
	logger    coreLogger.Logger
}

// NewModelUseCase creates a new ModelUseCase implementation.
func NewModelUseCase(modelRepo repository.ModelRepository, logger coreLogger.Logger) ModelUseCase {
	baseUseCase := coreUsecase.NewBaseUseCase[entity.AIModel](modelRepo, logger)
	return &modelUseCase{
		BaseUseCaseImpl: baseUseCase,
		modelRepo:       modelRepo,
		logger:          logger,
	}
}

// --- Custom Method Implementations --- //

// FindModelByName finds a single model by its name.
func (uc *modelUseCase) FindModelByName(ctx context.Context, name string) (*entity.AIModel, error) {
	uc.logger.Info("Finding model by name", "name", name)
	if name == "" {
		return nil, coreUsecase.NewUseCaseError(coreUsecase.ErrInvalidInput, "model name cannot be empty")
	}
	includeDeleted := false // Default: Don't include deleted
	// Use the specific FindByName method from the modelRepo
	model, err := uc.modelRepo.FindByName(ctx, name, includeDeleted)
	if err != nil {
		if errors.Is(err, types.ErrNotFound) {
			uc.logger.Warn("Model not found by name", "name", name)
			return nil, coreUsecase.NewUseCaseError(coreUsecase.ErrNotFound, fmt.Sprintf("model with name '%s' not found", name))
		}
		uc.logger.Error("Failed to find model by name in repository", "name", name, "error", err)
		// Consider wrapping the error
		return nil, fmt.Errorf("repository error finding model by name: %w", err)
	}
	uc.logger.Info("Model found by name", "name", name, "id", model.ID)
	return model, nil
}

// ListModelsByStation retrieves models for a specific station.
func (uc *modelUseCase) ListModelsByStation(ctx context.Context, stationID uuid.UUID, opts types.FilterOptions) (*types.PaginationResult[entity.AIModel], error) {
	uc.logger.Info("Listing models by station", "station_id", stationID, "options", opts)
	// Use the specific modelRepo for this custom query
	models, totalCount, err := uc.modelRepo.ListByStationID(ctx, stationID, opts)
	if err != nil {
		uc.logger.Error("Failed to list models by station from repository", "station_id", stationID, "error", err)
		return nil, fmt.Errorf("repository error listing models for station %s: %w", stationID, err)
	}

	result := &types.PaginationResult[entity.AIModel]{
		Items:      models,
		TotalItems: totalCount,
		Limit:      opts.Limit,
		Offset:     opts.Offset,
	}
	uc.logger.Info("Models listed successfully by station", "station_id", stationID, "count", len(models), "total", totalCount)
	return result, nil
}

// --- Removed Methods --- //
// HandleUpload was removed.
// ListModelsByStation was removed.
