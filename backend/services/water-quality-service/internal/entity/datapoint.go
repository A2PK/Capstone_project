package entity

import (
	"time"

	core_entity "golang-microservices-boilerplate/pkg/core/entity" // Import the base entity

	"github.com/google/uuid"
)

// --- Enums moved from indicator.go and renamed --- //

// ObservationType defines the possible observation types for the data point.
// Renamed from IndicatorObservationType
type ObservationType string

const (
	Actual             ObservationType = "actual"
	Interpolation      ObservationType = "interpolation"
	Predicted          ObservationType = "predicted"
	RealtimeMonitoring ObservationType = "realtime-monitoring"
)

// IndicatorPurpose defines how an indicator feature is intended to be used.
// Moved from indicator.go
type IndicatorPurpose string

const (
	PurposePrediction IndicatorPurpose = "prediction"
	PurposeDisplay    IndicatorPurpose = "display"
	PurposeAnalysis   IndicatorPurpose = "analysis"
)

// DataPointFeature represents a single indicator/measurement stored within a DataPoint's JSON field.
// This struct is NOT a GORM entity itself.
type DataPointFeature struct {
	Name         string           `json:"name"`                    // Indicator name (e.g., pH, DO)
	Value        *float64         `json:"value,omitempty"`         // Numerical value (pointer for nullability)
	TextualValue *string          `json:"textual_value,omitempty"` // Textual value (pointer for nullability)
	Purpose      IndicatorPurpose `json:"purpose,omitempty"`       // How this feature is used
	Source       string           `json:"source,omitempty"`        // Source of this specific feature measurement
}

// --- DataPoint Entity --- //

// DataPoint represents a single data collection event at a station, including its features.
type DataPoint struct {
	core_entity.BaseEntity
	MonitoringTime     time.Time          `json:"monitoring_time,omitempty" gorm:"not null;index;uniqueIndex:idx_datapoint_time_station_source"`
	WQI                *float64           `json:"wqi,omitempty" gorm:"type:decimal(10,4)"` // Water Quality Index (use pointer for nullability)
	StationID          uuid.UUID          `json:"station_id,omitempty" gorm:"type:uuid;not null;index;uniqueIndex:idx_datapoint_time_station_source"`
	Source             string             `json:"source,omitempty" gorm:"type:varchar(255);index;uniqueIndex:idx_datapoint_time_station_source"`
	ObservationType    ObservationType    `json:"observation_type,omitempty" gorm:"type:varchar(50);not null;index;check:chk_observation_type,observation_type IN ('actual', 'interpolation', 'predicted', 'realtime-monitoring')"` // Renamed type, kept json tag
	DataSourceSchemaID uuid.UUID          `json:"data_source_schema_id,omitempty" gorm:"type:uuid;not null;index"`                                                                                                                  // Added foreign key with json tag
	Features           []DataPointFeature `json:"features,omitempty" gorm:"type:jsonb;not null;serializer:json"`                                                                                                                    // Changed type from datatypes.JSON, kept json tag
}
