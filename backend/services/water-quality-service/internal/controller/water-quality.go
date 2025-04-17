package controller

import (
	"context"
	"io"
	"net/http"

	"github.com/google/uuid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/status"

	coreController "golang-microservices-boilerplate/pkg/core/controller"
	pb "golang-microservices-boilerplate/proto/water-quality-service"
	"golang-microservices-boilerplate/services/water-quality-service/internal/entity"
	"golang-microservices-boilerplate/services/water-quality-service/internal/usecase"
)

// WaterQualityServer defines the interface for the gRPC service handler
type WaterQualityServer interface {
	pb.WaterQualityServiceServer // Embed the generated interface
}

// waterQualityServer implements the WaterQualityServer interface
type waterQualityServer struct {
	pb.UnimplementedWaterQualityServiceServer
	stationUC          usecase.StationUsecase
	dataPointUC        usecase.DataPointUsecase
	importUC           usecase.ImportUseCase
	dataSourceSchemaUC usecase.DataSourceSchemaUsecase
	mapper             Mapper
}

// Ensure waterQualityServer implements WaterQualityServer interface
var _ WaterQualityServer = (*waterQualityServer)(nil)

// NewWaterQualityServer creates a new gRPC server instance
func NewWaterQualityServer(
	stationUC usecase.StationUsecase,
	dataPointUC usecase.DataPointUsecase,
	importUC usecase.ImportUseCase,
	dataSourceSchemaUC usecase.DataSourceSchemaUsecase,
	mapper Mapper,
) WaterQualityServer {
	return &waterQualityServer{
		stationUC:          stationUC,
		dataPointUC:        dataPointUC,
		importUC:           importUC,
		dataSourceSchemaUC: dataSourceSchemaUC,
		mapper:             mapper,
	}
}

// RegisterWaterQualityServiceServer registers the water quality service with the gRPC server
func RegisterWaterQualityServiceServer(
	s *grpc.Server,
	stationUC usecase.StationUsecase,
	dataPointUC usecase.DataPointUsecase,
	importUC usecase.ImportUseCase,
	dataSourceSchemaUC usecase.DataSourceSchemaUsecase,
	mapper Mapper,
) {
	server := NewWaterQualityServer(stationUC, dataPointUC, importUC, dataSourceSchemaUC, mapper)
	pb.RegisterWaterQualityServiceServer(s, server)
}

// --- Station Methods ---

// CreateStations implements the pb.WaterQualityServiceServer.CreateStations method
func (s *waterQualityServer) CreateStations(ctx context.Context, req *pb.CreateStationsRequest) (*pb.CreateStationsResponse, error) {
	if req == nil || len(req.Stations) == 0 {
		return &pb.CreateStationsResponse{Stations: []*pb.Station{}}, nil
	}

	// Map proto requests to entities
	stationEntities := make([]*entity.Station, 0, len(req.Stations))
	for i, stationInput := range req.Stations {
		stationEntity, err := s.mapper.ProtoStationInputToEntity(stationInput)
		if err != nil {
			return nil, status.Errorf(http.StatusBadRequest, "failed to map station %d in request: %v", i, err)
		}
		stationEntities = append(stationEntities, stationEntity)
	}

	// Create the stations - capture returned slice and error
	createdStations, err := s.stationUC.CreateMany(ctx, stationEntities)
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	// Map created entities (use the returned slice) back to proto
	stationProtos := make([]*pb.Station, 0, len(createdStations))
	for _, stationEntity := range createdStations { // Iterate over the returned slice
		stationProto, err := s.mapper.StationEntityToProto(stationEntity)
		if err != nil {
			return nil, status.Errorf(http.StatusInternalServerError, "failed to map station entity: %v", err)
		}
		stationProtos = append(stationProtos, stationProto)
	}

	return &pb.CreateStationsResponse{Stations: stationProtos}, nil
}

// UpdateStations implements the pb.WaterQualityServiceServer.UpdateStations method
func (s *waterQualityServer) UpdateStations(ctx context.Context, req *pb.UpdateStationsRequest) (*pb.UpdateStationsResponse, error) {
	if req == nil || len(req.Stations) == 0 {
		return &pb.UpdateStationsResponse{Stations: []*pb.Station{}}, nil
	}

	// Process each station update
	entitiesToUpdate := make([]*entity.Station, 0, len(req.Stations))
	for i, stationProto := range req.Stations {
		// Parse ID
		id, err := uuid.Parse(stationProto.Id)
		if err != nil {
			return nil, status.Errorf(http.StatusBadRequest, "invalid station ID for item %d: %v", i, err)
		}

		// Get existing station
		existingStation, err := s.stationUC.GetByID(ctx, id)
		if err != nil {
			return nil, coreController.MapErrorToHttpStatus(err)
		}

		// Apply updates from proto to entity
		if err := s.mapper.ApplyProtoStationUpdateToEntity(stationProto, existingStation); err != nil {
			return nil, status.Errorf(http.StatusBadRequest, "failed to map station %d update: %v", i, err)
		}

		entitiesToUpdate = append(entitiesToUpdate, existingStation)
	}

	// Update all stations - capture returned slice and error
	updatedStations, err := s.stationUC.UpdateMany(ctx, entitiesToUpdate)
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	// Map updated entities (use the returned slice) back to proto
	stationProtos := make([]*pb.Station, 0, len(updatedStations))
	for _, stationEntity := range updatedStations { // Iterate over the returned slice
		stationProto, err := s.mapper.StationEntityToProto(stationEntity)
		if err != nil {
			return nil, status.Errorf(http.StatusInternalServerError, "failed to map updated station entity: %v", err)
		}
		stationProtos = append(stationProtos, stationProto)
	}

	return &pb.UpdateStationsResponse{Stations: stationProtos}, nil
}

// DeleteStations implements the pb.WaterQualityServiceServer.DeleteStations method
func (s *waterQualityServer) DeleteStations(ctx context.Context, req *pb.DeleteRequest) (*pb.DeleteResponse, error) {
	if req == nil || len(req.Ids) == 0 {
		return &pb.DeleteResponse{AffectedCount: 0}, nil
	}

	// Convert string IDs to UUIDs
	uuids := make([]uuid.UUID, 0, len(req.Ids))
	for i, idStr := range req.Ids {
		id, err := uuid.Parse(idStr)
		if err != nil {
			return nil, status.Errorf(http.StatusBadRequest, "invalid station ID at index %d: %v", i, err)
		}
		uuids = append(uuids, id)
	}

	// Delete stations
	if err := s.stationUC.DeleteMany(ctx, uuids, req.HardDelete); err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	return &pb.DeleteResponse{AffectedCount: int64(len(uuids))}, nil
}

// ListStations implements the pb.WaterQualityServiceServer.ListStations method
func (s *waterQualityServer) ListStations(ctx context.Context, req *pb.ListStationsRequest) (*pb.ListStationsResponse, error) {
	// Convert proto filter options to core filter options
	opts := s.mapper.ProtoListStationsRequestToFilterOptions(req)

	// Get stations with pagination
	result, err := s.stationUC.List(ctx, opts)
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	// Map pagination result to proto response
	response, err := s.mapper.StationPaginationResultToProtoList(result)
	if err != nil {
		return nil, status.Errorf(http.StatusInternalServerError, "failed to map station pagination result: %v", err)
	}

	return response, nil
}

// --- DataPoint Methods ---

// CreateDataPoints implements the pb.WaterQualityServiceServer.CreateDataPoints method
func (s *waterQualityServer) CreateDataPoints(ctx context.Context, req *pb.CreateDataPointsRequest) (*pb.CreateDataPointsResponse, error) {
	if req == nil || len(req.DataPoints) == 0 {
		return &pb.CreateDataPointsResponse{DataPoints: []*pb.DataPoint{}}, nil
	}

	// 1. Map all inputs to entities first
	dpEntities := make([]*entity.DataPoint, 0, len(req.DataPoints))
	for i, dpInput := range req.DataPoints {
		// Map data point input (including its features)
		dpEntity, err := s.mapper.ProtoDataPointInputToEntity(dpInput)
		if err != nil {
			return nil, status.Errorf(http.StatusBadRequest, "failed to map data point input %d: %v", i, err)
		}

		// Parse and set StationID and DataSourceSchemaID (UUIDs from strings)
		stationID, err := uuid.Parse(dpInput.StationId)
		if err != nil {
			return nil, status.Errorf(http.StatusBadRequest, "invalid station ID for data point %d: %v", i, err)
		}
		dpEntity.StationID = stationID

		schemaID, err := uuid.Parse(dpInput.DataSourceSchemaId)
		if err != nil {
			return nil, status.Errorf(http.StatusBadRequest, "invalid data source schema ID for data point %d: %v", i, err)
		}
		dpEntity.DataSourceSchemaID = schemaID

		dpEntities = append(dpEntities, dpEntity)
	}

	// 2. Create data points using the use case (CreateMany handles features implicitly)
	createdDPs, err := s.dataPointUC.CreateMany(ctx, dpEntities)
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	// 3. Map created entities back to proto
	allDataPoints := make([]*pb.DataPoint, 0, len(createdDPs))
	for _, dpEntity := range createdDPs {
		dpProto, err := s.mapper.DataPointEntityToProto(dpEntity)
		if err != nil {
			// Log error, but maybe don't fail the whole request?
			// Depending on requirements, partial success might be okay.
			return nil, status.Errorf(http.StatusInternalServerError, "failed to map created data point entity %s: %v", dpEntity.ID.String(), err)
		}
		allDataPoints = append(allDataPoints, dpProto)
	}

	return &pb.CreateDataPointsResponse{DataPoints: allDataPoints}, nil
}

// UpdateDataPoints implements the pb.WaterQualityServiceServer.UpdateDataPoints method
func (s *waterQualityServer) UpdateDataPoints(ctx context.Context, req *pb.UpdateDataPointsRequest) (*pb.UpdateDataPointsResponse, error) {
	if req == nil || len(req.DataPoints) == 0 {
		return &pb.UpdateDataPointsResponse{DataPoints: []*pb.DataPoint{}}, nil
	}

	entitiesToUpdate := make([]*entity.DataPoint, 0, len(req.DataPoints))
	for i, dpProto := range req.DataPoints {
		id, err := uuid.Parse(dpProto.Id)
		if err != nil {
			return nil, status.Errorf(http.StatusBadRequest, "invalid data point ID for item %d: %v", i, err)
		}

		existingDP, err := s.dataPointUC.GetByID(ctx, id)
		if err != nil {
			return nil, coreController.MapErrorToHttpStatus(err)
		}

		if err := s.mapper.ApplyProtoDataPointUpdateToEntity(dpProto, existingDP); err != nil {
			return nil, status.Errorf(http.StatusBadRequest, "failed to map data point %d update: %v", i, err)
		}
		entitiesToUpdate = append(entitiesToUpdate, existingDP)
	}

	updatedDPs, err := s.dataPointUC.UpdateMany(ctx, entitiesToUpdate)
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	dpProtos := make([]*pb.DataPoint, 0, len(updatedDPs))
	for _, dpEntity := range updatedDPs {
		dpProto, err := s.mapper.DataPointEntityToProto(dpEntity)
		if err != nil {
			return nil, status.Errorf(http.StatusInternalServerError, "failed to map updated data point entity: %v", err)
		}
		dpProtos = append(dpProtos, dpProto)
	}

	return &pb.UpdateDataPointsResponse{DataPoints: dpProtos}, nil
}

// DeleteDataPoints implements the pb.WaterQualityServiceServer.DeleteDataPoints method
func (s *waterQualityServer) DeleteDataPoints(ctx context.Context, req *pb.DeleteRequest) (*pb.DeleteResponse, error) {
	if req == nil || len(req.Ids) == 0 {
		return &pb.DeleteResponse{AffectedCount: 0}, nil
	}

	uuids := make([]uuid.UUID, 0, len(req.Ids))
	for i, idStr := range req.Ids {
		id, err := uuid.Parse(idStr)
		if err != nil {
			return nil, status.Errorf(http.StatusBadRequest, "invalid data point ID at index %d: %v", i, err)
		}
		uuids = append(uuids, id)
	}

	if err := s.dataPointUC.DeleteMany(ctx, uuids, req.HardDelete); err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	return &pb.DeleteResponse{AffectedCount: int64(len(uuids))}, nil
}

// ListDataPointsByStation implements the pb.WaterQualityServiceServer.ListDataPointsByStation method
func (s *waterQualityServer) ListDataPointsByStation(ctx context.Context, req *pb.ListDataPointsByStationRequest) (*pb.ListDataPointsByStationResponse, error) {
	if req == nil || req.StationId == "" {
		return nil, status.Errorf(http.StatusBadRequest, "station ID is required")
	}

	stationID, err := uuid.Parse(req.StationId)
	if err != nil {
		return nil, status.Errorf(http.StatusBadRequest, "invalid station ID format: %v", err)
	}

	opts := s.mapper.ProtoListDataPointsRequestToFilterOptions(req)

	result, err := s.dataPointUC.ListByStation(ctx, stationID, opts)
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	response, err := s.mapper.DataPointPaginationResultToProtoList(result)
	if err != nil {
		return nil, status.Errorf(http.StatusInternalServerError, "failed to map data point pagination result: %v", err)
	}

	return response, nil
}

// --- File Upload Method ---

// UploadData implements the streaming pb.WaterQualityServiceServer.UploadData method
func (s *waterQualityServer) UploadData(stream pb.WaterQualityService_UploadDataServer) error {
	var filename string
	var filetype string
	var metadataReceived bool
	var pr *io.PipeReader
	var pw *io.PipeWriter
	var importErr error
	importDone := make(chan struct{}) // Channel to signal import completion
	ctx := stream.Context()

	for {
		req, err := stream.Recv()
		if err == io.EOF {
			if !metadataReceived {
				return status.Error(http.StatusBadRequest, "EOF received before metadata (filename, filetype)")
			}
			if pw != nil {
				pw.Close() // Signal end of data to the reader
			}
			break // Exit the receive loop
		}
		if err != nil {
			return status.Errorf(http.StatusInternalServerError, "error receiving stream message: %v", err)
		}

		switch payload := req.GetPayload().(type) {
		case *pb.UploadRequest_Filename:
			if filename != "" {
				return status.Error(http.StatusBadRequest, "filename sent more than once")
			}
			filename = payload.Filename
			if filetype != "" && !metadataReceived {
				metadataReceived = true
				pr, pw = io.Pipe()
				go s.runImportAndHandleResponse(ctx, pr, filename, filetype, importDone, &importErr, stream)
			}
		case *pb.UploadRequest_FileType:
			if filetype != "" {
				return status.Error(http.StatusBadRequest, "file_type sent more than once")
			}
			filetype = payload.FileType
			if filename != "" && !metadataReceived {
				metadataReceived = true
				pr, pw = io.Pipe()
				go s.runImportAndHandleResponse(ctx, pr, filename, filetype, importDone, &importErr, stream)
			}
		case *pb.UploadRequest_DataChunk:
			if !metadataReceived {
				return status.Error(http.StatusBadRequest, "data_chunk received before complete metadata (filename, filetype)")
			}
			if _, err := pw.Write(payload.DataChunk); err != nil {
				<-importDone
				return status.Errorf(http.StatusInternalServerError, "error writing chunk to processing pipe: %v", err)
			}
		default:
			return status.Error(http.StatusBadRequest, "unknown payload type in UploadRequest")
		}
	}

	// Wait for the import process to finish if it started
	if metadataReceived {
		<-importDone
	}

	// Return the final error captured from the goroutine (nil if successful)
	return importErr
}

// runImportAndHandleResponse runs import logic and sends the response back via the stream.
func (s *waterQualityServer) runImportAndHandleResponse(
	ctx context.Context,
	pr *io.PipeReader,
	filename, filetype string,
	done chan<- struct{},
	errResult *error,
	stream pb.WaterQualityService_UploadDataServer,
) {
	defer close(done) // Ensure done is closed on exit

	response, err := s.importUC.ImportData(ctx, pr, filename, filetype)
	if err != nil {
		pr.CloseWithError(err) // Close the reader part of the pipe on error
		*errResult = status.Errorf(http.StatusInternalServerError, "error processing data: %v", err)
		// We cannot send the response here as SendAndClose is called by the main handler upon return.
		// The error is captured in errResult.
		return
	}

	// If successful, send the response and close the stream.
	if err := stream.SendAndClose(response); err != nil {
		*errResult = status.Errorf(http.StatusInternalServerError, "failed to send upload response: %v", err)
		return
	}

	*errResult = nil // Signal success
}

// --- DataSourceSchema Methods ---

// CreateDataSourceSchema implements the pb.WaterQualityServiceServer.CreateDataSourceSchema method
func (s *waterQualityServer) CreateDataSourceSchema(ctx context.Context, req *pb.CreateDataSourceSchemaRequest) (*pb.CreateDataSourceSchemaResponse, error) {
	if req == nil || req.Schema == nil {
		return nil, status.Errorf(http.StatusBadRequest, "schema is required")
	}

	// Map proto to entity
	schemaEntity, err := s.mapper.ProtoDataSourceSchemaInputToEntity(req.Schema)
	if err != nil {
		return nil, status.Errorf(http.StatusBadRequest, "failed to map schema: %v", err)
	}

	// Create the schema
	if err := s.dataSourceSchemaUC.Create(ctx, schemaEntity); err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	// Map entity back to proto
	schemaProto, err := s.mapper.DataSourceSchemaEntityToProto(schemaEntity)
	if err != nil {
		return nil, status.Errorf(http.StatusInternalServerError, "failed to map schema entity: %v", err)
	}

	return &pb.CreateDataSourceSchemaResponse{Schema: schemaProto}, nil
}

// UpdateDataSourceSchema implements the pb.WaterQualityServiceServer.UpdateDataSourceSchema method
func (s *waterQualityServer) UpdateDataSourceSchema(ctx context.Context, req *pb.UpdateDataSourceSchemaRequest) (*pb.UpdateDataSourceSchemaResponse, error) {
	if req == nil || req.Schema == nil {
		return nil, status.Errorf(http.StatusBadRequest, "schema is required")
	}

	// Parse ID
	id, err := uuid.Parse(req.Schema.Id)
	if err != nil {
		return nil, status.Errorf(http.StatusBadRequest, "invalid schema ID: %v", err)
	}

	// Get existing schema
	existingSchema, err := s.dataSourceSchemaUC.GetByID(ctx, id)
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	// Apply updates from proto to entity
	if err := s.mapper.ApplyProtoDataSourceSchemaUpdateToEntity(req.Schema, existingSchema); err != nil {
		return nil, status.Errorf(http.StatusBadRequest, "failed to map schema update: %v", err)
	}

	// Update the schema
	if err := s.dataSourceSchemaUC.Update(ctx, existingSchema); err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	// Map updated entity back to proto
	schemaProto, err := s.mapper.DataSourceSchemaEntityToProto(existingSchema)
	if err != nil {
		return nil, status.Errorf(http.StatusInternalServerError, "failed to map updated schema entity: %v", err)
	}

	return &pb.UpdateDataSourceSchemaResponse{Schema: schemaProto}, nil
}

// GetDataSourceSchema implements the pb.WaterQualityServiceServer.GetDataSourceSchema method
func (s *waterQualityServer) GetDataSourceSchema(ctx context.Context, req *pb.GetDataSourceSchemaRequest) (*pb.GetDataSourceSchemaResponse, error) {
	if req == nil || req.Id == "" {
		return nil, status.Errorf(http.StatusBadRequest, "schema ID is required")
	}

	// Parse ID
	id, err := uuid.Parse(req.Id)
	if err != nil {
		return nil, status.Errorf(http.StatusBadRequest, "invalid schema ID: %v", err)
	}

	// Get schema
	schema, err := s.dataSourceSchemaUC.GetByID(ctx, id)
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	// Map entity to proto
	schemaProto, err := s.mapper.DataSourceSchemaEntityToProto(schema)
	if err != nil {
		return nil, status.Errorf(http.StatusInternalServerError, "failed to map schema entity: %v", err)
	}

	return &pb.GetDataSourceSchemaResponse{Schema: schemaProto}, nil
}

// ListDataSourceSchemas implements the pb.WaterQualityServiceServer.ListDataSourceSchemas method
func (s *waterQualityServer) ListDataSourceSchemas(ctx context.Context, req *pb.ListDataSourceSchemasRequest) (*pb.ListDataSourceSchemasResponse, error) {
	// Convert proto filter options to core filter options
	opts := s.mapper.ProtoListDataSourceSchemasRequestToFilterOptions(req)

	// Get schemas with pagination
	result, err := s.dataSourceSchemaUC.List(ctx, opts)
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	// Map pagination result to proto response
	response, err := s.mapper.DataSourceSchemaPaginationResultToProtoList(result)
	if err != nil {
		return nil, status.Errorf(http.StatusInternalServerError, "failed to map schema pagination result: %v", err)
	}

	return response, nil
}
