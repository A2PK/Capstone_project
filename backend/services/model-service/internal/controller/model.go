package controller

import (
	"context"
	"errors"

	// "io" // No longer needed

	modelpb "golang-microservices-boilerplate/proto/model-service"

	"github.com/google/uuid"

	"golang-microservices-boilerplate/pkg/core/logger"
	"golang-microservices-boilerplate/pkg/core/types"
	coreUsecase "golang-microservices-boilerplate/pkg/core/usecase"
	"golang-microservices-boilerplate/services/model-service/internal/entity"
	"golang-microservices-boilerplate/services/model-service/internal/usecase"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// ModelServiceServer implements the gRPC server interface for the ModelService.
type ModelServiceServer struct {
	modelpb.UnimplementedModelServiceServer // Embed for forward compatibility
	logger                                  logger.Logger
	mapper                                  Mapper
	modelUseCase                            usecase.ModelUseCase
}

// NewModelServiceServer creates a new ModelServiceServer instance.
func NewModelServiceServer(
	logger logger.Logger,
	mapper Mapper,
	modelUseCase usecase.ModelUseCase,
) *ModelServiceServer {
	return &ModelServiceServer{
		logger:       logger,
		mapper:       mapper,
		modelUseCase: modelUseCase,
	}
}

// --- gRPC Method Implementations ---

// CreateModels handles creating new model metadata records.
func (s *ModelServiceServer) CreateModels(ctx context.Context, req *modelpb.CreateModelsRequest) (*modelpb.CreateModelsResponse, error) {
	s.logger.Info("gRPC CreateModels called")
	if len(req.Models) == 0 {
		return nil, status.Error(codes.InvalidArgument, "no models provided for creation")
	}

	entitiesToCreate := make([]*entity.AIModel, 0, len(req.Models))
	for i, modelInput := range req.Models {
		ent, err := s.mapper.FromProtoModelInput(modelInput)
		if err != nil {
			s.logger.Error("Failed to map model input from proto", "index", i, "error", err)
			return nil, status.Errorf(codes.InvalidArgument, "invalid model input data at index %d: %v", i, err)
		}
		// FilePath will be empty here, system needs to update it later.
		entitiesToCreate = append(entitiesToCreate, ent)
	}

	// Use the CreateMany method from the embedded BaseUseCase
	createdEntities, err := s.modelUseCase.CreateMany(ctx, entitiesToCreate)
	if err != nil {
		s.logger.Error("Failed to create models via use case", "error", err)
		// TODO: Map specific use case/repo errors (e.g., conflict)
		return nil, status.Errorf(codes.Internal, "failed to create models: %v", err)
	}

	resp := &modelpb.CreateModelsResponse{
		Models: s.mapper.ToProtoModelList(createdEntities), // Map the results back
	}
	return resp, nil
}

// UpdateModels handles updating model metadata.
func (s *ModelServiceServer) UpdateModels(ctx context.Context, req *modelpb.UpdateModelsRequest) (*modelpb.UpdateModelsResponse, error) {
	s.logger.Info("gRPC UpdateModels called")
	if len(req.Models) == 0 {
		return nil, status.Error(codes.InvalidArgument, "no models provided for update")
	}

	entitiesToUpdate, err := s.mapper.FromProtoModelList(req.Models)
	if err != nil {
		s.logger.Error("Failed to map models from proto for update", "error", err)
		return nil, status.Errorf(codes.InvalidArgument, "invalid model data: %v", err)
	}

	// Use the UpdateMany method from the embedded BaseUseCase
	updatedEntities, err := s.modelUseCase.UpdateMany(ctx, entitiesToUpdate)
	if err != nil {
		s.logger.Error("Failed to update models via use case", "error", err)
		// Map specific use case errors to gRPC status codes
		// Check for use case specific errors first
		var ucErr *coreUsecase.UseCaseError
		if errors.As(err, &ucErr) {
			if ucErr.Type == coreUsecase.ErrNotFound {
				return nil, status.Errorf(codes.NotFound, "one or more models not found for update: %v", err)
			}
			// Add mappings for other use case error types if needed
		}
		// Check for standard types.ErrNotFound from repo layer if not wrapped by use case
		if errors.Is(err, types.ErrNotFound) {
			return nil, status.Errorf(codes.NotFound, "one or more models not found for update: %v", err)
		}
		return nil, status.Errorf(codes.Internal, "failed to update models: %v", err)
	}

	resp := &modelpb.UpdateModelsResponse{
		Models: s.mapper.ToProtoModelList(updatedEntities), // Map results back
	}
	return resp, nil
}

// DeleteModels handles deleting model metadata.
func (s *ModelServiceServer) DeleteModels(ctx context.Context, req *modelpb.DeleteModelsRequest) (*modelpb.DeleteModelsResponse, error) {
	s.logger.Info("gRPC DeleteModels called", "ids", req.Ids, "hard_delete", req.HardDelete)
	if len(req.Ids) == 0 {
		return nil, status.Error(codes.InvalidArgument, "no model IDs provided for deletion")
	}

	idsToDelete := make([]uuid.UUID, 0, len(req.Ids))
	var invalidIDs []string
	for _, idStr := range req.Ids {
		id, err := uuid.Parse(idStr)
		if err != nil {
			s.logger.Warn("Invalid UUID format in delete request", "id", idStr, "error", err)
			invalidIDs = append(invalidIDs, idStr)
			continue // Skip invalid IDs for now, but report later
		}
		idsToDelete = append(idsToDelete, id)
	}

	if len(invalidIDs) > 0 {
		return nil, status.Errorf(codes.InvalidArgument, "invalid UUID format for IDs: %v", invalidIDs)
	}
	if len(idsToDelete) == 0 {
		return nil, status.Error(codes.InvalidArgument, "no valid model IDs provided for deletion")
	}

	// Use the DeleteMany method from the embedded BaseUseCase
	err := s.modelUseCase.DeleteMany(ctx, idsToDelete, req.HardDelete)
	if err != nil {
		s.logger.Error("Failed to delete models via use case", "error", err)
		// BaseUseCase DeleteMany might not return specific errors like NotFound
		// It might just delete those that exist. If specific error handling is needed,
		// the use case might need to override DeleteMany.
		return nil, status.Errorf(codes.Internal, "failed to delete models: %v", err)
	}

	// BaseUseCase.DeleteMany doesn't return affected count directly.
	// We return the count of IDs *requested* for deletion if successful.
	// A more accurate count would require changes to BaseUseCase/Repo.
	resp := &modelpb.DeleteModelsResponse{
		AffectedCount: int64(len(idsToDelete)),
	}
	return resp, nil
}

// ListModels handles listing model metadata.
func (s *ModelServiceServer) ListModels(ctx context.Context, req *modelpb.ListModelsRequest) (*modelpb.ListModelsResponse, error) {
	s.logger.Info("gRPC ListModels called")

	filterOpts := s.mapper.FromProtoFilterOptions(req.Options)

	// Use the FindAll method from the embedded BaseUseCase
	// Note: BaseRepository uses FindAll, BaseUseCase uses List which calls FindAll.
	paginatedResult, err := s.modelUseCase.List(ctx, filterOpts)
	if err != nil {
		s.logger.Error("Failed to list models via use case", "error", err)
		return nil, status.Errorf(codes.Internal, "failed to list models: %v", err)
	}

	resp := &modelpb.ListModelsResponse{
		Models:     s.mapper.ToProtoModelList(paginatedResult.Items),
		Pagination: s.mapper.ToProtoPaginationInfo(paginatedResult),
	}
	return resp, nil
}

// ListModelsByStation handles listing models by station (Custom use case method).
func (s *ModelServiceServer) ListModelsByStation(ctx context.Context, req *modelpb.ListModelsByStationRequest) (*modelpb.ListModelsByStationResponse, error) {
	s.logger.Info("gRPC ListModelsByStation called", "station_id", req.StationId)

	stationID, err := uuid.Parse(req.StationId)
	if err != nil {
		s.logger.Warn("Invalid station UUID format", "station_id", req.StationId, "error", err)
		return nil, status.Errorf(codes.InvalidArgument, "invalid station ID format: %s", req.StationId)
	}

	filterOpts := s.mapper.FromProtoFilterOptions(req.Options)

	// Call the specific method defined in the ModelUseCase interface/implementation
	paginatedResult, err := s.modelUseCase.ListModelsByStation(ctx, stationID, filterOpts)
	if err != nil {
		s.logger.Error("Failed to list models by station via use case", "station_id", req.StationId, "error", err)
		// Map specific errors if ListModelsByStation returns them
		return nil, status.Errorf(codes.Internal, "failed to list models by station: %v", err)
	}

	resp := &modelpb.ListModelsByStationResponse{
		Models:     s.mapper.ToProtoModelList(paginatedResult.Items),
		Pagination: s.mapper.ToProtoPaginationInfo(paginatedResult),
	}
	return resp, nil
}

// FindModelByName handles finding a model by name (Custom use case method).
func (s *ModelServiceServer) FindModelByName(ctx context.Context, req *modelpb.FindModelByNameRequest) (*modelpb.FindModelByNameResponse, error) {
	s.logger.Info("gRPC FindModelByName called", "name", req.Name)
	if req.Name == "" {
		return nil, status.Error(codes.InvalidArgument, "model name cannot be empty")
	}

	modelEntity, err := s.modelUseCase.FindModelByName(ctx, req.Name)
	if err != nil {
		s.logger.Error("Failed to find model by name via use case", "name", req.Name, "error", err)
		// Check for specific use case errors
		var ucErr *coreUsecase.UseCaseError
		if errors.As(err, &ucErr) {
			if ucErr.Type == coreUsecase.ErrNotFound {
				return nil, status.Errorf(codes.NotFound, "model not found with name: %s", req.Name)
			}
			// Add mappings for other use case error types
		}
		// Fallback check for generic not found from repo if not wrapped
		if errors.Is(err, types.ErrNotFound) {
			return nil, status.Errorf(codes.NotFound, "model not found with name: %s", req.Name)
		}
		return nil, status.Errorf(codes.Internal, "failed to find model by name: %v", err)
	}

	if modelEntity == nil {
		s.logger.Error("FindModelByName use case returned nil entity without error", "name", req.Name)
		return nil, status.Error(codes.Internal, "internal error retrieving model")
	}

	resp := &modelpb.FindModelByNameResponse{
		Model: s.mapper.ToProtoModel(modelEntity),
	}
	return resp, nil
}

// --- Removed UploadModel gRPC handler and related code --- //
