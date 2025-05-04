package usecase

import (
	"context"

	coreLogger "golang-microservices-boilerplate/pkg/core/logger"
	coreUsecase "golang-microservices-boilerplate/pkg/core/usecase"
	"golang-microservices-boilerplate/services/water-quality-service/internal/entity"
	"golang-microservices-boilerplate/services/water-quality-service/internal/repository"
)

// StationUsecase defines the interface for station-related business logic
type StationUsecase interface {
	// Embed the core use case
	coreUsecase.BaseUseCase[entity.Station]
	// Custom method to find by name
	FindByName(ctx context.Context, name string) (*entity.Station, error)
}

// stationUseCaseImpl implements the StationUsecase interface
type stationUseCaseImpl struct {
	// Embed the core use case implementation
	*coreUsecase.BaseUseCaseImpl[entity.Station]
	repo   repository.StationRepository
	logger coreLogger.Logger
}

// NewStationUsecase creates a new instance of StationUsecase
func NewStationUsecase(
	repo repository.StationRepository,
	logger coreLogger.Logger,
) StationUsecase {
	baseUseCase := coreUsecase.NewBaseUseCase(repo, logger)
	return &stationUseCaseImpl{
		BaseUseCaseImpl: baseUseCase,
		repo:            repo,
		logger:          logger,
	}
}

// FindByName finds a station by its name
func (uc *stationUseCaseImpl) FindByName(ctx context.Context, name string) (*entity.Station, error) {
	uc.logger.Info("Finding station by name", "name", name)
	station, err := uc.repo.FindByName(ctx, name)
	if err != nil {
		uc.logger.Error("Failed to find station by name", "name", name, "error", err)
		// Convert repository error to usecase error
		if err.Error() == "station not found" {
			return nil, coreUsecase.NewUseCaseError(coreUsecase.ErrNotFound, "station not found")
		}
		return nil, coreUsecase.NewUseCaseError(coreUsecase.ErrInternal, "failed to find station")
	}
	return station, nil
}
