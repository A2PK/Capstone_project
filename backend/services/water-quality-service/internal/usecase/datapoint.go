package usecase

import (
	"context"

	"github.com/google/uuid"

	coreLogger "golang-microservices-boilerplate/pkg/core/logger"
	coreTypes "golang-microservices-boilerplate/pkg/core/types"
	coreUsecase "golang-microservices-boilerplate/pkg/core/usecase"
	"golang-microservices-boilerplate/services/water-quality-service/internal/entity"
	"golang-microservices-boilerplate/services/water-quality-service/internal/repository"
)

// DataPointUsecase defines the interface for data point-related business logic
type DataPointUsecase interface {
	// Embed the core use case
	coreUsecase.BaseUseCase[entity.DataPoint]
	// ListByStation retrieves data points for a specific station with pagination
	ListByStation(ctx context.Context, stationID uuid.UUID, opts coreTypes.FilterOptions) (*coreTypes.PaginationResult[entity.DataPoint], error)
	// CreateWithIndicators method removed as Indicators are now Features within DataPoint
}

// dataPointUseCaseImpl implements the DataPointUsecase interface
type dataPointUseCaseImpl struct {
	// Embed the core use case implementation
	*coreUsecase.BaseUseCaseImpl[entity.DataPoint]
	repo repository.DataPointRepository
	// indicatorRepo repository.IndicatorRepository // Removed
	logger coreLogger.Logger
}

// NewDataPointUsecase creates a new instance of DataPointUsecase
func NewDataPointUsecase(
	repo repository.DataPointRepository,
	// indicatorRepo repository.IndicatorRepository, // Removed
	logger coreLogger.Logger,
) DataPointUsecase {
	baseUseCase := coreUsecase.NewBaseUseCase(repo, logger)
	return &dataPointUseCaseImpl{
		BaseUseCaseImpl: baseUseCase,
		repo:            repo,
		// indicatorRepo:   indicatorRepo, // Removed
		logger: logger,
	}
}

// ListByStation retrieves data points for a specific station with pagination
func (uc *dataPointUseCaseImpl) ListByStation(ctx context.Context, stationID uuid.UUID, opts coreTypes.FilterOptions) (*coreTypes.PaginationResult[entity.DataPoint], error) {
	uc.logger.Info("Listing data points for station", "station_id", stationID, "page", opts.Offset, "limit", opts.Limit)
	// Call the repository method that handles pagination and potentially preloading/filtering
	// Note: FindByStationID might need adjustments if filtering by features is required
	result, err := uc.repo.FindByStationID(ctx, stationID, opts)
	if err != nil {
		uc.logger.Error("Failed to list data points by station", "station_id", stationID, "error", err)
		return nil, err
	}
	uc.logger.Info("Successfully listed data points for station", "station_id", stationID, "count", len(result.Items))
	return result, nil
}

// CreateWithIndicators method removed
// Creating a DataPoint now automatically includes its Features, handled by the standard Create method
// func (uc *dataPointUseCaseImpl) CreateWithIndicators(...) { ... }
