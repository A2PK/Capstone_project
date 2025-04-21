package controller

import (
	"encoding/json"
	"errors"
	"fmt"

	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"

	coreTypes "golang-microservices-boilerplate/pkg/core/types"
	corePb "golang-microservices-boilerplate/proto/core"
	pb "golang-microservices-boilerplate/proto/water-quality-service"
	"golang-microservices-boilerplate/services/water-quality-service/internal/entity"
	// Add schema import if needed later for complex operations like login in user-service
	// wqSchema "golang-microservices-boilerplate/services/water-quality-service/internal/schema"
)

// Mapper defines the interface for mapping between gRPC proto messages and internal types
// for the water quality service.
type Mapper interface {
	// Station Mappings
	StationEntityToProto(station *entity.Station) (*pb.Station, error)
	ProtoStationInputToEntity(req *pb.StationInput) (*entity.Station, error)
	ApplyProtoStationUpdateToEntity(req *pb.Station, existingStation *entity.Station) error
	ProtoListStationsRequestToFilterOptions(req *pb.ListStationsRequest) coreTypes.FilterOptions
	StationPaginationResultToProtoList(result *coreTypes.PaginationResult[entity.Station]) (*pb.ListStationsResponse, error)

	// DataPoint & Feature Mappings
	DataPointEntityToProto(dp *entity.DataPoint) (*pb.DataPoint, error)
	ProtoDataPointInputToEntity(req *pb.DataPointInput) (*entity.DataPoint, error)
	ApplyProtoDataPointUpdateToEntity(req *pb.DataPoint, existingDP *entity.DataPoint) error
	ProtoListDataPointsRequestToFilterOptions(req *pb.ListDataPointsByStationRequest) coreTypes.FilterOptions
	DataPointPaginationResultToProtoList(result *coreTypes.PaginationResult[entity.DataPoint]) (*pb.ListDataPointsByStationResponse, error)

	// DataSourceSchema Mappings
	DataSourceSchemaEntityToProto(schema *entity.DataSourceSchema) (*pb.DataSourceSchema, error)
	ProtoDataSourceSchemaInputToEntity(req *pb.DataSourceSchemaInput) (*entity.DataSourceSchema, error)
	ApplyProtoDataSourceSchemaUpdateToEntity(req *pb.DataSourceSchema, existingSchema *entity.DataSourceSchema) error
	ProtoListDataSourceSchemasRequestToFilterOptions(req *pb.ListDataSourceSchemasRequest) coreTypes.FilterOptions
	DataSourceSchemaPaginationResultToProtoList(result *coreTypes.PaginationResult[entity.DataSourceSchema]) (*pb.ListDataSourceSchemasResponse, error)

	// Add mappings for ListAllDataPoints
	ProtoListAllDataPointsRequestToFilterOptions(req *pb.ListAllDataPointsRequest) coreTypes.FilterOptions
	DataPointPaginationResultToProtoListAll(result *coreTypes.PaginationResult[entity.DataPoint]) (*pb.ListAllDataPointsResponse, error)
}

// Ensure WaterQualityMapper implements Mapper interface.
var _ Mapper = (*WaterQualityMapper)(nil)

// WaterQualityMapper handles mapping between gRPC proto messages and internal types.
type WaterQualityMapper struct{}

// NewWaterQualityMapper creates a new instance of WaterQualityMapper.
func NewWaterQualityMapper() *WaterQualityMapper {
	return &WaterQualityMapper{}
}

// Helper function to safely dereference string pointers (if needed, though protos use optional)
// func derefString(s *string) string {
// 	if s == nil {
// 		return ""
// 	}
// 	return *s
// }

// Helper function to convert proto Value to Go interface{}
func mapProtoValueToGo(v *structpb.Value) interface{} {
	if v == nil {
		return nil
	}
	switch v.Kind.(type) {
	case *structpb.Value_NullValue:
		return nil
	case *structpb.Value_NumberValue:
		return v.GetNumberValue()
	case *structpb.Value_StringValue:
		return v.GetStringValue()
	case *structpb.Value_BoolValue:
		return v.GetBoolValue()
	case *structpb.Value_StructValue:
		m := make(map[string]interface{})
		for k, val := range v.GetStructValue().Fields {
			m[k] = mapProtoValueToGo(val)
		}
		return m
	case *structpb.Value_ListValue:
		s := make([]interface{}, 0, len(v.GetListValue().Values))
		for _, val := range v.GetListValue().Values {
			s = append(s, mapProtoValueToGo(val))
		}
		return s
	default:
		return nil
	}
}

// --- Station Mappers ---

func (m *WaterQualityMapper) StationEntityToProto(station *entity.Station) (*pb.Station, error) {
	if station == nil {
		return nil, errors.New("cannot map nil station entity to proto")
	}
	var deletedAt *timestamppb.Timestamp
	if station.DeletedAt != nil {
		deletedAt = timestamppb.New(*station.DeletedAt)
	}

	return &pb.Station{
		Id:        station.ID.String(),
		CreatedAt: timestamppb.New(station.CreatedAt),
		UpdatedAt: timestamppb.New(station.UpdatedAt),
		DeletedAt: deletedAt,
		Name:      station.Name,
		Latitude:  station.Latitude,
		Longitude: station.Longitude,
		Country:   station.Country,
		Location:  station.Location,
	}, nil
}

func (m *WaterQualityMapper) ProtoStationInputToEntity(req *pb.StationInput) (*entity.Station, error) {
	if req == nil {
		return nil, errors.New("cannot map nil station input to entity")
	}
	if req.Name == "" || req.Country == "" {
		return nil, errors.New("station name and country are required")
	}

	// Assuming entity.Station mirrors proto.Station structure + base fields
	// ID, CreatedAt, UpdatedAt, DeletedAt are handled by the ORM/database
	return &entity.Station{
		Name:      req.Name,
		Latitude:  req.Latitude,
		Longitude: req.Longitude,
		Country:   req.Country,
		Location:  req.Location,
	}, nil
}

func (m *WaterQualityMapper) ApplyProtoStationUpdateToEntity(req *pb.Station, existingStation *entity.Station) error {
	if req == nil || existingStation == nil {
		return errors.New("request and existing station entity must not be nil")
	}
	// ID check might be redundant if lookup is done before calling this
	if req.Id != existingStation.ID.String() {
		return fmt.Errorf("mismatched station ID for update: req=%s, existing=%s", req.Id, existingStation.ID.String())
	}

	// Apply fields selectively based on what might be updatable
	// Usually, base fields like ID, CreatedAt, DeletedAt are not updated via this route.
	existingStation.Name = req.Name           // Assume Name is always provided in update payload if intended
	existingStation.Latitude = req.Latitude   // Assume Lat is always provided
	existingStation.Longitude = req.Longitude // Assume Lon is always provided
	existingStation.Country = req.Country     // Assume Country is always provided
	existingStation.Location = req.Location   // Assume Location is always provided
	// UpdatedAt is handled by the ORM/database hook

	return nil // Return nil on success
}

func (m *WaterQualityMapper) ProtoListStationsRequestToFilterOptions(req *pb.ListStationsRequest) coreTypes.FilterOptions {
	opts := coreTypes.DefaultFilterOptions()
	if req == nil || req.Options == nil {
		return opts
	}

	if req.Options.Limit != nil {
		opts.Limit = int(*req.Options.Limit)
	}
	if req.Options.Offset != nil {
		opts.Offset = int(*req.Options.Offset)
	}
	if req.Options.SortBy != nil {
		opts.SortBy = *req.Options.SortBy
	}
	if req.Options.SortDesc != nil {
		opts.SortDesc = *req.Options.SortDesc
	}
	if req.Options.IncludeDeleted != nil {
		opts.IncludeDeleted = *req.Options.IncludeDeleted
	}

	if len(req.Options.Filters) > 0 {
		opts.Filters = make(map[string]interface{}, len(req.Options.Filters))
		for k, v := range req.Options.Filters {
			opts.Filters[k] = mapProtoValueToGo(v)
		}
	}
	return opts
}

func (m *WaterQualityMapper) StationPaginationResultToProtoList(result *coreTypes.PaginationResult[entity.Station]) (*pb.ListStationsResponse, error) {
	if result == nil {
		// Return empty list instead of error? Consistent with user-service
		return &pb.ListStationsResponse{
			Stations: []*pb.Station{},
			Pagination: &corePb.PaginationInfo{
				TotalItems: 0,
				Limit:      0,
				Offset:     0,
			},
		}, nil
	}

	protoStations := make([]*pb.Station, 0, len(result.Items))
	for _, station := range result.Items {
		protoStation, err := m.StationEntityToProto(station) // Remove & as Items is []*entity.Station
		if err != nil {
			// Log the error? Skip the item? Return partial list with error?
			// Let's return the error for now, interrupting the mapping.
			return nil, fmt.Errorf("failed to map station entity %s to proto: %w", station.ID.String(), err)
		}
		protoStations = append(protoStations, protoStation)
	}

	return &pb.ListStationsResponse{
		Stations: protoStations,
		Pagination: &corePb.PaginationInfo{
			TotalItems: result.TotalItems,
			Limit:      int32(result.Limit),
			Offset:     int32(result.Offset),
		},
	}, nil
}

// --- DataPoint Feature Mappers (New) ---

func (m *WaterQualityMapper) DataPointFeatureEntityToProto(feature *entity.DataPointFeature) (*pb.DataPointFeature, error) {
	if feature == nil {
		return nil, errors.New("cannot map nil data point feature entity to proto")
	}

	// Map entity enums to proto enums
	var purpose pb.IndicatorPurpose
	switch feature.Purpose {
	case entity.PurposePrediction:
		purpose = pb.IndicatorPurpose_INDICATOR_PURPOSE_PREDICTION
	case entity.PurposeDisplay:
		purpose = pb.IndicatorPurpose_INDICATOR_PURPOSE_DISPLAY
	case entity.PurposeAnalysis:
		purpose = pb.IndicatorPurpose_INDICATOR_PURPOSE_ANALYSIS
	default:
		purpose = pb.IndicatorPurpose_INDICATOR_PURPOSE_UNSPECIFIED
	}

	return &pb.DataPointFeature{
		Name:         feature.Name,
		Value:        feature.Value,        // *float64 maps directly to optional double
		TextualValue: feature.TextualValue, // *string maps directly to optional string
		Purpose:      purpose,
		Source:       feature.Source,
	}, nil
}

func (m *WaterQualityMapper) ProtoDataPointFeatureInputToEntity(req *pb.DataPointFeatureInput) (*entity.DataPointFeature, error) {
	if req == nil {
		return nil, errors.New("cannot map nil data point feature input to entity")
	}
	if req.Name == "" {
		return nil, errors.New("feature name is required")
	}

	// Map proto enums to entity enums
	var purpose entity.IndicatorPurpose
	switch req.Purpose {
	case pb.IndicatorPurpose_INDICATOR_PURPOSE_PREDICTION:
		purpose = entity.PurposePrediction
	case pb.IndicatorPurpose_INDICATOR_PURPOSE_DISPLAY:
		purpose = entity.PurposeDisplay
	case pb.IndicatorPurpose_INDICATOR_PURPOSE_ANALYSIS:
		purpose = entity.PurposeAnalysis
	default:
		purpose = entity.PurposeAnalysis // Default
	}

	return &entity.DataPointFeature{
		Name:         req.Name,
		Value:        req.Value,        // optional double maps directly to *float64
		TextualValue: req.TextualValue, // optional string maps directly to *string
		Purpose:      purpose,
		Source:       req.Source,
	}, nil
}

// Added: Mapper from Proto DataPointFeature (non-input) to Entity DataPointFeature
func (m *WaterQualityMapper) ProtoDataPointFeatureToEntity(req *pb.DataPointFeature) (*entity.DataPointFeature, error) {
	if req == nil {
		return nil, errors.New("cannot map nil data point feature proto to entity")
	}
	if req.Name == "" {
		return nil, errors.New("feature name is required")
	}

	// Map proto enums to entity enums
	var purpose entity.IndicatorPurpose
	switch req.Purpose {
	case pb.IndicatorPurpose_INDICATOR_PURPOSE_PREDICTION:
		purpose = entity.PurposePrediction
	case pb.IndicatorPurpose_INDICATOR_PURPOSE_DISPLAY:
		purpose = entity.PurposeDisplay
	case pb.IndicatorPurpose_INDICATOR_PURPOSE_ANALYSIS:
		purpose = entity.PurposeAnalysis
	default:
		purpose = entity.PurposeAnalysis // Default
	}

	return &entity.DataPointFeature{
		Name:         req.Name,
		Value:        req.Value,
		TextualValue: req.TextualValue,
		Purpose:      purpose,
		Source:       req.Source,
	}, nil
}

// --- DataPoint Mappers (Updated) ---

func (m *WaterQualityMapper) DataPointEntityToProto(dp *entity.DataPoint) (*pb.DataPoint, error) {
	if dp == nil {
		return nil, errors.New("cannot map nil data point entity to proto")
	}
	var deletedAt *timestamppb.Timestamp
	if dp.DeletedAt != nil {
		deletedAt = timestamppb.New(*dp.DeletedAt)
	}

	// Map observation type enum
	var obsType pb.ObservationType
	switch dp.ObservationType {
	case entity.Actual:
		obsType = pb.ObservationType_OBSERVATION_TYPE_ACTUAL
	case entity.Interpolation:
		obsType = pb.ObservationType_OBSERVATION_TYPE_INTERPOLATION
	case entity.Predicted:
		obsType = pb.ObservationType_OBSERVATION_TYPE_PREDICTED
	case entity.RealtimeMonitoring:
		obsType = pb.ObservationType_OBSERVATION_TYPE_REALTIME_MONITORING
	default:
		obsType = pb.ObservationType_OBSERVATION_TYPE_UNSPECIFIED
	}

	// Map associated features
	protoFeatures := make([]*pb.DataPointFeature, 0, len(dp.Features))
	for i := range dp.Features { // Iterate by index to pass pointer to element
		protoFeature, err := m.DataPointFeatureEntityToProto(&dp.Features[i])
		if err != nil {
			return nil, fmt.Errorf("failed to map feature %s for data point %s: %w", dp.Features[i].Name, dp.ID.String(), err)
		}
		protoFeatures = append(protoFeatures, protoFeature)
	}

	return &pb.DataPoint{
		Id:                 dp.ID.String(),
		CreatedAt:          timestamppb.New(dp.CreatedAt),
		UpdatedAt:          timestamppb.New(dp.UpdatedAt),
		DeletedAt:          deletedAt,
		MonitoringTime:     timestamppb.New(dp.MonitoringTime),
		Wqi:                dp.WQI, // *float64 maps directly to optional double
		StationId:          dp.StationID.String(),
		Source:             dp.Source,
		ObservationType:    obsType,
		DataSourceSchemaId: dp.DataSourceSchemaID.String(),
		Features:           protoFeatures,
	}, nil
}

func (m *WaterQualityMapper) ProtoDataPointInputToEntity(req *pb.DataPointInput) (*entity.DataPoint, error) {
	if req == nil {
		return nil, errors.New("cannot map nil data point input to entity")
	}
	if !req.MonitoringTime.IsValid() || req.StationId == "" || req.Source == "" || req.DataSourceSchemaId == "" {
		return nil, errors.New("monitoring time, station ID, source, and data source schema ID are required for data point input")
	}

	// Map observation type enum
	var obsType entity.ObservationType
	switch req.ObservationType {
	case pb.ObservationType_OBSERVATION_TYPE_ACTUAL:
		obsType = entity.Actual
	case pb.ObservationType_OBSERVATION_TYPE_INTERPOLATION:
		obsType = entity.Interpolation
	case pb.ObservationType_OBSERVATION_TYPE_PREDICTED:
		obsType = entity.Predicted
	case pb.ObservationType_OBSERVATION_TYPE_REALTIME_MONITORING:
		obsType = entity.RealtimeMonitoring
	default:
		// Default to Actual or return error?
		obsType = entity.Actual
	}

	// StationID and DataSourceSchemaID need conversion from string to UUID
	// Perform this in the service/usecase layer before calling create/update
	// stationID, err := uuid.Parse(req.StationId)
	// if err != nil { ... }
	// schemaID, err := uuid.Parse(req.DataSourceSchemaId)
	// if err != nil { ... }

	dpEntity := &entity.DataPoint{
		MonitoringTime: req.MonitoringTime.AsTime(),
		WQI:            req.Wqi, // optional double maps directly to *float64
		// StationID:      stationID, // Set by caller after parsing
		Source:          req.Source,
		ObservationType: obsType,
		// DataSourceSchemaID: schemaID, // Set by caller after parsing
	}

	// Map features
	if len(req.Features) > 0 {
		dpEntity.Features = make([]entity.DataPointFeature, 0, len(req.Features))
		for _, featureInput := range req.Features {
			featureEntity, err := m.ProtoDataPointFeatureInputToEntity(featureInput)
			if err != nil {
				return nil, fmt.Errorf("failed to map feature input '%s': %w", featureInput.Name, err)
			}
			dpEntity.Features = append(dpEntity.Features, *featureEntity)
		}
	}

	// Return only the DataPoint entity
	return dpEntity, nil
}

func (m *WaterQualityMapper) ApplyProtoDataPointUpdateToEntity(req *pb.DataPoint, existingDP *entity.DataPoint) error {
	if req == nil || existingDP == nil {
		return errors.New("request and existing data point entity must not be nil")
	}
	if req.Id != existingDP.ID.String() {
		return fmt.Errorf("mismatched data point ID for update: req=%s, existing=%s", req.Id, existingDP.ID.String())
	}

	if req.MonitoringTime != nil && req.MonitoringTime.IsValid() {
		existingDP.MonitoringTime = req.MonitoringTime.AsTime()
	}

	// Map observation type enum if changed
	if req.ObservationType != pb.ObservationType_OBSERVATION_TYPE_UNSPECIFIED {
		var obsType entity.ObservationType
		switch req.ObservationType {
		case pb.ObservationType_OBSERVATION_TYPE_ACTUAL:
			obsType = entity.Actual
		case pb.ObservationType_OBSERVATION_TYPE_INTERPOLATION:
			obsType = entity.Interpolation
		case pb.ObservationType_OBSERVATION_TYPE_PREDICTED:
			obsType = entity.Predicted
		case pb.ObservationType_OBSERVATION_TYPE_REALTIME_MONITORING:
			obsType = entity.RealtimeMonitoring
		default:
			obsType = entity.Actual // Or existingDP.ObservationType?
		}
		if obsType != existingDP.ObservationType {
			existingDP.ObservationType = obsType
		}
	}

	// StationID update might be restricted, handle in service layer if needed
	// DataSourceSchemaID likely should not be updated
	if req.Source != "" { // Allow updating source?
		existingDP.Source = req.Source
	}

	// Update WQI - handles nil case from proto optional
	existingDP.WQI = req.Wqi

	// --- Feature Update Logic (Replace) ---
	if req.Features != nil { // Check if Features field is present in the request
		newFeatures := make([]entity.DataPointFeature, 0, len(req.Features))
		for _, featureProto := range req.Features {
			// Use the correct mapping function: ProtoDataPointFeatureToEntity
			featureEntity, err := m.ProtoDataPointFeatureToEntity(featureProto)
			if err != nil {
				return fmt.Errorf("failed to map incoming feature '%s' for update: %w", featureProto.Name, err)
			}
			newFeatures = append(newFeatures, *featureEntity)
		}
		existingDP.Features = newFeatures
	} else {
		// Leave existing features untouched if req.Features is nil
	}

	return nil
}

func (m *WaterQualityMapper) ProtoListDataPointsRequestToFilterOptions(req *pb.ListDataPointsByStationRequest) coreTypes.FilterOptions {
	opts := coreTypes.DefaultFilterOptions()
	if req == nil {
		return opts
	}

	// Add station_id filter (assuming StationId is non-empty based on proto)
	opts.Filters = map[string]interface{}{"station_id": req.StationId}

	if req.Options != nil {
		if req.Options.Limit != nil {
			opts.Limit = int(*req.Options.Limit)
		}
		if req.Options.Offset != nil {
			opts.Offset = int(*req.Options.Offset)
		}
		if req.Options.SortBy != nil {
			opts.SortBy = *req.Options.SortBy
		}
		if req.Options.SortDesc != nil {
			opts.SortDesc = *req.Options.SortDesc
		}
		if req.Options.IncludeDeleted != nil {
			opts.IncludeDeleted = *req.Options.IncludeDeleted
		}

		if len(req.Options.Filters) > 0 {
			for k, v := range req.Options.Filters {
				if k != "station_id" { // Avoid overwriting the mandatory filter
					opts.Filters[k] = mapProtoValueToGo(v)
				}
			}
		}
	}
	return opts
}

func (m *WaterQualityMapper) DataPointPaginationResultToProtoList(result *coreTypes.PaginationResult[entity.DataPoint]) (*pb.ListDataPointsByStationResponse, error) {
	if result == nil {
		return &pb.ListDataPointsByStationResponse{
			DataPoints: []*pb.DataPoint{},
			Pagination: &corePb.PaginationInfo{TotalItems: 0, Limit: 0, Offset: 0},
		}, nil
	}

	protoDataPoints := make([]*pb.DataPoint, 0, len(result.Items))
	for _, dp := range result.Items {
		protoDP, err := m.DataPointEntityToProto(dp)
		if err != nil {
			return nil, fmt.Errorf("failed to map data point entity %s to proto: %w", dp.ID.String(), err)
		}
		protoDataPoints = append(protoDataPoints, protoDP)
	}

	return &pb.ListDataPointsByStationResponse{
		DataPoints: protoDataPoints,
		Pagination: &corePb.PaginationInfo{
			TotalItems: result.TotalItems,
			Limit:      int32(result.Limit),
			Offset:     int32(result.Offset),
		},
	}, nil
}

// ProtoListAllDataPointsRequestToFilterOptions maps ListAllDataPointsRequest to core FilterOptions.
func (m *WaterQualityMapper) ProtoListAllDataPointsRequestToFilterOptions(req *pb.ListAllDataPointsRequest) coreTypes.FilterOptions {
	opts := coreTypes.DefaultFilterOptions()
	if req == nil || req.Options == nil {
		return opts
	}

	// No station_id filter applied here
	if req.Options.Limit != nil {
		opts.Limit = int(*req.Options.Limit)
	}
	if req.Options.Offset != nil {
		opts.Offset = int(*req.Options.Offset)
	}
	if req.Options.SortBy != nil {
		opts.SortBy = *req.Options.SortBy
	}
	if req.Options.SortDesc != nil {
		opts.SortDesc = *req.Options.SortDesc
	}
	if req.Options.IncludeDeleted != nil {
		opts.IncludeDeleted = *req.Options.IncludeDeleted
	}

	if len(req.Options.Filters) > 0 {
		opts.Filters = make(map[string]interface{}, len(req.Options.Filters))
		for k, v := range req.Options.Filters {
			opts.Filters[k] = mapProtoValueToGo(v)
		}
	}
	return opts
}

// DataPointPaginationResultToProtoListAll maps a PaginationResult to ListAllDataPointsResponse.
func (m *WaterQualityMapper) DataPointPaginationResultToProtoListAll(result *coreTypes.PaginationResult[entity.DataPoint]) (*pb.ListAllDataPointsResponse, error) {
	if result == nil {
		return &pb.ListAllDataPointsResponse{
			DataPoints: []*pb.DataPoint{},
			Pagination: &corePb.PaginationInfo{TotalItems: 0, Limit: 0, Offset: 0},
		}, nil
	}

	protoDataPoints := make([]*pb.DataPoint, 0, len(result.Items))
	for _, dp := range result.Items {
		protoDP, err := m.DataPointEntityToProto(dp)
		if err != nil {
			return nil, fmt.Errorf("failed to map data point entity %s to proto: %w", dp.ID.String(), err)
		}
		protoDataPoints = append(protoDataPoints, protoDP)
	}

	return &pb.ListAllDataPointsResponse{
		DataPoints: protoDataPoints,
		Pagination: &corePb.PaginationInfo{
			TotalItems: result.TotalItems,
			Limit:      int32(result.Limit),
			Offset:     int32(result.Offset),
		},
	}, nil
}

// --- DataSourceSchema Mappers (Updated) ---

func (m *WaterQualityMapper) DataSourceSchemaEntityToProto(schema *entity.DataSourceSchema) (*pb.DataSourceSchema, error) {
	if schema == nil {
		return nil, errors.New("cannot map nil schema entity to proto")
	}
	var deletedAt *timestamppb.Timestamp
	if schema.DeletedAt != nil {
		deletedAt = timestamppb.New(*schema.DeletedAt)
	}

	// Convert []entity.FieldDefinition to proto struct (structpb.Struct)
	// We need to marshal the Go slice into JSON bytes first.
	var schemaDefPb *structpb.Struct
	if len(schema.SchemaDefinition) > 0 {
		// Wrap the slice in a map to conform to Struct's expected object structure
		schemaMap := map[string]interface{}{"definitions": schema.SchemaDefinition}
		var err error
		schemaDefPb, err = structpb.NewStruct(schemaMap)
		if err != nil {
			// Fallback: Try marshalling the raw slice directly if wrapping fails (less ideal for Struct)
			// This might indicate an issue with how NewStruct handles slices of structs.
			// Let's try converting the slice to []interface{} first.
			defsInterface := make([]interface{}, len(schema.SchemaDefinition))
			for i, v := range schema.SchemaDefinition {
				// Convert each FieldDefinition to map[string]interface{}
				var itemMap map[string]interface{}
				itemBytes, _ := json.Marshal(v)
				_ = json.Unmarshal(itemBytes, &itemMap)
				defsInterface[i] = itemMap
			}
			schemaMapRetry := map[string]interface{}{"definitions": defsInterface}
			schemaDefPb, err = structpb.NewStruct(schemaMapRetry)
			if err != nil {
				return nil, fmt.Errorf("failed to convert schema definition slice to proto struct: %w", err)
			}
		}
	}

	return &pb.DataSourceSchema{
		Id:               schema.ID.String(),
		CreatedAt:        timestamppb.New(schema.CreatedAt),
		UpdatedAt:        timestamppb.New(schema.UpdatedAt),
		DeletedAt:        deletedAt,
		Name:             schema.Name,
		SourceIdentifier: schema.SourceIdentifier,
		SourceType:       string(schema.SourceType),
		Description:      schema.Description,
		SchemaDefinition: schemaDefPb,
	}, nil
}

func (m *WaterQualityMapper) ProtoDataSourceSchemaInputToEntity(req *pb.DataSourceSchemaInput) (*entity.DataSourceSchema, error) {
	if req == nil {
		return nil, errors.New("cannot map nil schema input to entity")
	}
	if req.Name == "" || req.SourceType == "" {
		return nil, errors.New("schema name and source type are required")
	}

	// Convert proto struct back to []entity.FieldDefinition
	var schemaDefEntity []entity.FieldDefinition
	if req.SchemaDefinition != nil {
		schemaMap := req.SchemaDefinition.AsMap()
		// Extract the list from the wrapped structure if we wrapped it during EntityToProto
		if defsInterface, ok := schemaMap["definitions"]; ok {
			// Marshal the extracted list/map back to JSON bytes
			jsonBytes, err := json.Marshal(defsInterface)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal schema definition from proto struct field: %w", err)
			}
			// Unmarshal the JSON bytes into the target Go slice type
			err = json.Unmarshal(jsonBytes, &schemaDefEntity)
			if err != nil {
				return nil, fmt.Errorf("failed to unmarshal schema definition into []FieldDefinition: %w", err)
			}
		} else {
			// If "definitions" key isn't found, maybe it wasn't wrapped?
			// Try marshalling the whole map and unmarshalling (less likely to work for a slice)
			jsonBytes, err := json.Marshal(schemaMap)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal whole schema definition struct: %w", err)
			}
			err = json.Unmarshal(jsonBytes, &schemaDefEntity)
			if err != nil {
				// Log warning or handle error - schema struct didn't match expected format
				// For now, let schemaDefEntity remain nil/empty
			}
		}
	}

	return &entity.DataSourceSchema{
		Name:             req.Name,
		SourceIdentifier: req.SourceIdentifier,
		SourceType:       entity.SourceType(req.SourceType),
		Description:      req.Description,
		SchemaDefinition: schemaDefEntity,
	}, nil
}

func (m *WaterQualityMapper) ApplyProtoDataSourceSchemaUpdateToEntity(req *pb.DataSourceSchema, existingSchema *entity.DataSourceSchema) error {
	if req == nil || existingSchema == nil {
		return errors.New("request and existing schema entity must not be nil")
	}
	if req.Id != existingSchema.ID.String() {
		return fmt.Errorf("mismatched schema ID for update: req=%s, existing=%s", req.Id, existingSchema.ID.String())
	}

	// Convert proto struct back to []entity.FieldDefinition
	var schemaDefEntity []entity.FieldDefinition
	if req.SchemaDefinition != nil {
		schemaMap := req.SchemaDefinition.AsMap()
		if defsInterface, ok := schemaMap["definitions"]; ok {
			jsonBytes, err := json.Marshal(defsInterface)
			if err != nil {
				return fmt.Errorf("failed to marshal schema definition from proto struct field for update: %w", err)
			}
			err = json.Unmarshal(jsonBytes, &schemaDefEntity)
			if err != nil {
				return fmt.Errorf("failed to unmarshal schema definition into []FieldDefinition for update: %w", err)
			}
		} else {
			// Handle case where struct might not be wrapped (optional)
		}
	}

	existingSchema.Name = req.Name
	existingSchema.SourceIdentifier = req.SourceIdentifier
	existingSchema.SourceType = entity.SourceType(req.SourceType)
	existingSchema.Description = req.Description
	existingSchema.SchemaDefinition = schemaDefEntity // Assign the mapped slice

	return nil
}

func (m *WaterQualityMapper) ProtoListDataSourceSchemasRequestToFilterOptions(req *pb.ListDataSourceSchemasRequest) coreTypes.FilterOptions {
	opts := coreTypes.DefaultFilterOptions()
	if req == nil || req.Options == nil {
		return opts
	}

	if req.Options.Limit != nil {
		opts.Limit = int(*req.Options.Limit)
	}
	if req.Options.Offset != nil {
		opts.Offset = int(*req.Options.Offset)
	}
	if req.Options.SortBy != nil {
		opts.SortBy = *req.Options.SortBy
	}
	if req.Options.SortDesc != nil {
		opts.SortDesc = *req.Options.SortDesc
	}
	if req.Options.IncludeDeleted != nil {
		opts.IncludeDeleted = *req.Options.IncludeDeleted
	}

	if len(req.Options.Filters) > 0 {
		opts.Filters = make(map[string]interface{}, len(req.Options.Filters))
		for k, v := range req.Options.Filters {
			opts.Filters[k] = mapProtoValueToGo(v)
		}
	}
	return opts
}

func (m *WaterQualityMapper) DataSourceSchemaPaginationResultToProtoList(result *coreTypes.PaginationResult[entity.DataSourceSchema]) (*pb.ListDataSourceSchemasResponse, error) {
	if result == nil {
		return &pb.ListDataSourceSchemasResponse{
			Schemas:    []*pb.DataSourceSchema{},
			Pagination: &corePb.PaginationInfo{TotalItems: 0, Limit: 0, Offset: 0},
		}, nil
	}

	protoSchemas := make([]*pb.DataSourceSchema, 0, len(result.Items))
	for _, schema := range result.Items {
		protoSchema, err := m.DataSourceSchemaEntityToProto(schema)
		if err != nil {
			return nil, fmt.Errorf("failed to map schema entity %s to proto: %w", schema.ID.String(), err)
		}
		protoSchemas = append(protoSchemas, protoSchema)
	}

	return &pb.ListDataSourceSchemasResponse{
		Schemas: protoSchemas,
		Pagination: &corePb.PaginationInfo{
			TotalItems: result.TotalItems,
			Limit:      int32(result.Limit),
			Offset:     int32(result.Offset),
		},
	}, nil
}
