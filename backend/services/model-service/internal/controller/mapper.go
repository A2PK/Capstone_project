package controller

import (
	"fmt"
	"time"

	commonpb "golang-microservices-boilerplate/proto/core"
	modelpb "golang-microservices-boilerplate/proto/model-service"
	"golang-microservices-boilerplate/services/model-service/internal/entity"

	"github.com/google/uuid"

	core_entity "golang-microservices-boilerplate/pkg/core/entity"
	"golang-microservices-boilerplate/pkg/core/types"

	"google.golang.org/protobuf/types/known/timestamppb"
)

// Add imports for entity types and protobuf types when defined
// entity "golang-microservices-boilerplate/services/model-service/internal/entity"

// Mapper defines the interface for mapping operations.
type Mapper interface {
	ToProtoModel(mod *entity.AIModel) *modelpb.Model
	FromProtoModel(pbMod *modelpb.Model) (*entity.AIModel, error)
	ToProtoModelList(models []*entity.AIModel) []*modelpb.Model
	FromProtoModelList(pbModels []*modelpb.Model) ([]*entity.AIModel, error)
	FromProtoModelInput(pbInput *modelpb.ModelInput) (*entity.AIModel, error)
	FromProtoFilterOptions(pbOpts *commonpb.FilterOptions) types.FilterOptions
	ToProtoPaginationInfo(pagination *types.PaginationResult[entity.AIModel]) *commonpb.PaginationInfo
}

// ModelMapper implements the Mapper interface.
type ModelMapper struct {
}

// NewModelMapper creates a new ModelMapper instance.
func NewModelMapper() Mapper {
	return &ModelMapper{}
}

// ToProtoModel converts an internal AIModel entity to its protobuf representation.
func (m *ModelMapper) ToProtoModel(mod *entity.AIModel) *modelpb.Model {
	if mod == nil {
		return nil
	}
	var deletedAt *timestamppb.Timestamp
	if mod.BaseEntity.DeletedAt != nil {
		deletedAt = timestamppb.New(*mod.BaseEntity.DeletedAt)
	}
	// StationID is non-nullable uuid.UUID in entity, always convert
	stationIDStr := mod.StationID.String()

	return &modelpb.Model{
		Id:           mod.ID.String(),
		CreatedAt:    timestamppb.New(mod.CreatedAt),
		UpdatedAt:    timestamppb.New(mod.UpdatedAt),
		DeletedAt:    deletedAt,
		Name:         mod.Name,
		FilePath:     mod.FilePath,
		Version:      mod.Version,
		Description:  mod.Description,
		TrainedAt:    timestamppb.New(mod.TrainedAt),
		StationId:    stationIDStr,
		Availability: mod.Availability,
	}
}

// FromProtoModel converts a protobuf Model message to its internal entity representation.
// Note: This usually maps input data, so BaseEntity fields (ID, CreatedAt, UpdatedAt, DeletedAt) might be ignored or handled specifically by the use case.
func (m *ModelMapper) FromProtoModel(pbMod *modelpb.Model) (*entity.AIModel, error) {
	if pbMod == nil {
		return nil, nil // Or return an error if nil is invalid
	}

	var id uuid.UUID
	var err error
	if pbMod.Id != "" {
		id, err = uuid.Parse(pbMod.Id)
		if err != nil {
			return nil, fmt.Errorf("invalid UUID format for model ID '%s': %w", pbMod.Id, err)
		}
	}

	// StationID is required in entity, parse or error out
	var stationID uuid.UUID
	if pbMod.StationId != "" {
		stationID, err = uuid.Parse(pbMod.StationId)
		if err != nil {
			return nil, fmt.Errorf("invalid UUID format for station ID '%s': %w", pbMod.StationId, err)
		}
	} else {
		// Handle missing required StationID from proto
		return nil, fmt.Errorf("station ID is required in Model proto")
	}

	var trainedAt time.Time
	if pbMod.TrainedAt != nil {
		if err := pbMod.TrainedAt.CheckValid(); err == nil {
			trainedAt = pbMod.TrainedAt.AsTime()
		} else {
			return nil, fmt.Errorf("invalid TrainedAt timestamp: %w", err)
		}
	} else {
		// Handle case where TrainedAt is mandatory but missing in proto
		// Based on proto, TrainedAt seems required in the Model message.
		return nil, fmt.Errorf("TrainedAt timestamp is required in Model proto")
	}

	// Note: BaseEntity fields are set here, but might be overwritten by GORM/DB
	return &entity.AIModel{
		BaseEntity: core_entity.BaseEntity{
			ID:        id,          // May be zero UUID if pbMod.Id was empty
			CreatedAt: time.Time{}, // Set by DB
			UpdatedAt: time.Time{}, // Set by DB
		},
		Name:         pbMod.Name,
		FilePath:     pbMod.FilePath,
		Version:      pbMod.Version,
		Description:  pbMod.Description,
		TrainedAt:    trainedAt,
		StationID:    stationID,
		Availability: pbMod.Availability,
	}, nil
}

// ToProtoModelList converts a slice of internal AIModel entities to a slice of protobuf Models.
func (m *ModelMapper) ToProtoModelList(models []*entity.AIModel) []*modelpb.Model {
	protoModels := make([]*modelpb.Model, 0, len(models))
	for _, mod := range models {
		if mod != nil {
			protoModels = append(protoModels, m.ToProtoModel(mod))
		}
	}
	return protoModels
}

// FromProtoModelList converts a slice of protobuf Models to a slice of internal entities.
func (m *ModelMapper) FromProtoModelList(pbModels []*modelpb.Model) ([]*entity.AIModel, error) {
	entities := make([]*entity.AIModel, 0, len(pbModels))
	for _, pbMod := range pbModels {
		ent, err := m.FromProtoModel(pbMod)
		if err != nil {
			// Decide on error handling: return immediately or collect errors?
			return nil, fmt.Errorf("error mapping model %s: %w", pbMod.GetId(), err)
		}
		if ent != nil { // Handle nil input proto message case
			entities = append(entities, ent)
		}
	}
	return entities, nil
}

// FromProtoModelInput converts a protobuf ModelInput message to a partial entity.AIModel.
// Used for creating new model metadata records.
// Ignores fields set by the system (ID, FilePath, CreatedAt, UpdatedAt, DeletedAt).
func (m *ModelMapper) FromProtoModelInput(pbInput *modelpb.ModelInput) (*entity.AIModel, error) {
	if pbInput == nil {
		return nil, fmt.Errorf("model input cannot be nil")
	}

	// StationID is required in entity, parse or error out
	var stationID uuid.UUID
	var err error
	if pbInput.StationId != "" {
		stationID, err = uuid.Parse(pbInput.StationId)
		if err != nil {
			return nil, fmt.Errorf("invalid UUID format for station ID '%s': %w", pbInput.StationId, err)
		}
	} else {
		// Handle missing required StationID from input
		return nil, fmt.Errorf("station ID is required in model input")
	}

	var trainedAt time.Time
	if pbInput.TrainedAt != nil {
		if err := pbInput.TrainedAt.CheckValid(); err == nil {
			trainedAt = pbInput.TrainedAt.AsTime()
		} else {
			return nil, fmt.Errorf("invalid TrainedAt timestamp in input: %w", err)
		}
	} else {
		// Check if TrainedAt is mandatory based on requirements
		return nil, fmt.Errorf("TrainedAt timestamp is required in model input")
	}

	// FilePath is NOT set from input, it will be set by the system later.
	return &entity.AIModel{
		// BaseEntity fields are not set from input
		Name:         pbInput.Name,
		Version:      pbInput.Version,
		Description:  pbInput.Description,
		TrainedAt:    trainedAt,
		StationID:    stationID,
		Availability: pbInput.Availability,
		// FilePath: "", // Explicitly empty or omitted
	}, nil
}

// FromProtoFilterOptions converts protobuf FilterOptions to internal FilterOptions type.
func (m *ModelMapper) FromProtoFilterOptions(pbOpts *commonpb.FilterOptions) types.FilterOptions {
	if pbOpts == nil {
		return types.DefaultFilterOptions()
	}

	opts := types.DefaultFilterOptions() // Start with defaults

	if pbOpts.Limit != nil {
		opts.Limit = int(*pbOpts.Limit)
	}
	if pbOpts.Offset != nil {
		opts.Offset = int(*pbOpts.Offset)
	}
	if pbOpts.SortBy != nil {
		opts.SortBy = *pbOpts.SortBy
	}
	if pbOpts.SortDesc != nil {
		opts.SortDesc = *pbOpts.SortDesc
	}
	if pbOpts.IncludeDeleted != nil {
		opts.IncludeDeleted = *pbOpts.IncludeDeleted
	}

	// TODO: Map pbOpts.Filters if it exists and is used
	opts.Filters = nil

	return opts
}

// ToProtoPaginationInfo converts internal PaginationResult to protobuf PaginationInfo.
// Note the generic type needs to be AIModel here.
func (m *ModelMapper) ToProtoPaginationInfo(pagination *types.PaginationResult[entity.AIModel]) *commonpb.PaginationInfo {
	if pagination == nil {
		return nil
	}
	return &commonpb.PaginationInfo{
		TotalItems: int64(pagination.TotalItems),
		Limit:      int32(pagination.Limit),
		Offset:     int32(pagination.Offset),
	}
}

// --- Removed Placeholder Mappings ---
