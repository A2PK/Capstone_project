package usecase

import (
	"context"
	"strings"

	coreLogger "golang-microservices-boilerplate/pkg/core/logger"
	coreUsecase "golang-microservices-boilerplate/pkg/core/usecase"
	"golang-microservices-boilerplate/services/water-quality-service/internal/entity"
	"golang-microservices-boilerplate/services/water-quality-service/internal/repository"
)

// ThresholdConfigUsecase defines the interface for threshold configuration business logic
type ThresholdConfigUsecase interface {
	// Embed the core use case
	coreUsecase.BaseUseCase[entity.ThresholdConfig]

	// FindByElementNames retrieves multiple ThresholdConfig entities based on a list of element names.
	// Returns a map where the key is the lowercase element name and the value is the config entity.
	FindByElementNames(ctx context.Context, names []string) (map[string]*entity.ThresholdConfig, error)
}

// thresholdConfigUseCaseImpl implements the ThresholdConfigUsecase interface
type thresholdConfigUseCaseImpl struct {
	// Embed the core use case implementation
	*coreUsecase.BaseUseCaseImpl[entity.ThresholdConfig]
	repo   repository.ThresholdConfigRepository
	logger coreLogger.Logger
}

// NewThresholdConfigUsecase creates a new instance of ThresholdConfigUsecase
func NewThresholdConfigUsecase(
	repo repository.ThresholdConfigRepository,
	logger coreLogger.Logger,
) ThresholdConfigUsecase {
	baseUseCase := coreUsecase.NewBaseUseCase(repo, logger)
	return &thresholdConfigUseCaseImpl{
		BaseUseCaseImpl: baseUseCase,
		repo:            repo,
		logger:          logger,
	}
}

// FindByElementNames retrieves threshold configurations by element names.
func (uc *thresholdConfigUseCaseImpl) FindByElementNames(ctx context.Context, names []string) (map[string]*entity.ThresholdConfig, error) {
	uc.logger.Info("Finding threshold configs by element names", "count", len(names))
	if len(names) == 0 {
		return make(map[string]*entity.ThresholdConfig), nil
	}

	// Prepare unique lowercase names for query
	uniqueLowerNamesMap := make(map[string]struct{})
	for _, name := range names {
		uniqueLowerNamesMap[strings.ToLower(name)] = struct{}{}
	}
	queryNames := make([]string, 0, len(uniqueLowerNamesMap))
	for name := range uniqueLowerNamesMap {
		queryNames = append(queryNames, name)
	}

	results, err := uc.repo.FindByElementNames(ctx, queryNames)
	if err != nil {
		uc.logger.Error("Failed to find threshold configs by element names", "error", err)
		// Consider returning a domain-specific error here instead of the raw repo error
		return nil, coreUsecase.NewUseCaseError(coreUsecase.ErrInternal, "failed to retrieve threshold configurations")
	}

	uc.logger.Info("Successfully found threshold configs", "found_count", len(results))
	return results, nil
}
