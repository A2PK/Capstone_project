package entity

import (
	core_entity "golang-microservices-boilerplate/pkg/core/entity"
)

// SourceType defines the type of the data source.
type SourceType string

const (
	SourceTypeCSV    SourceType = "csv"
	SourceTypeJSON   SourceType = "json"
	SourceTypeExcel  SourceType = "excel" // Added Excel
	SourceTypeAPI    SourceType = "api"
	SourceTypeManual SourceType = "manual"
)

// --- Schema Definition Structures ---

// FieldDataType represents the expected data type of a field in the source.
type FieldDataType string

const (
	DataTypeString     FieldDataType = "string"
	DataTypeFloat      FieldDataType = "float"
	DataTypeInteger    FieldDataType = "integer"
	DataTypeBoolean    FieldDataType = "boolean"    // For fields like "Âm tính"/"Dương tính"
	DataTypeDate       FieldDataType = "date"       // Can add format info if needed
	DataTypeDateTime   FieldDataType = "datetime"   // Can add format info if needed
	DataTypeCoordinate FieldDataType = "coordinate" // Special type for "lat\nlon" format
	DataTypeText       FieldDataType = "text"       // For potentially long text
	DataTypeUnknown    FieldDataType = "unknown"
)

// FieldTargetEntity indicates which domain entity a source field maps to.
type FieldTargetEntity string

const (
	TargetEntityStation   FieldTargetEntity = "Station"
	TargetEntityDataPoint FieldTargetEntity = "DataPoint"
	TargetEntityIndicator FieldTargetEntity = "Indicator"
	TargetEntityIgnore    FieldTargetEntity = "Ignore" // If a column should be ignored
)

// FieldDefinition describes a single field/column within a data source schema.
// A slice of these []FieldDefinition will be stored as JSON in DataSourceSchema.SchemaDefinition.
type FieldDefinition struct {
	SourceName   string            `json:"source_name"`             // Exact name from header (e.g., "Điểm Quan Trắc")
	DataType     FieldDataType     `json:"data_type"`               // Expected data type (e.g., "float", "date")
	Unit         string            `json:"unit,omitempty"`          // Unit from the source (e.g., "mg/l")
	TargetEntity FieldTargetEntity `json:"target_entity"`           // Which entity it maps to ("Station", "Indicator", etc.)
	TargetField  string            `json:"target_field,omitempty"`  // Specific field if mapping to Station/DataPoint (e.g., "Name", "MonitoringTime", "WQI")
	Purpose      IndicatorPurpose  `json:"purpose,omitempty"`       // If TargetEntity is "Indicator", specify Purpose (Analysis/Display)
	Description  string            `json:"description,omitempty"`   // Optional description
	IsRequired   bool              `json:"is_required,omitempty"`   // Is the field required in the source?
	ExampleValue string            `json:"example_value,omitempty"` // An example value from the source data
	// Add other properties as needed: format string for dates, validation rules, etc.
}

// --- DataSourceSchema Entity ---

// DataSourceSchema stores the structure/schema definition of ingested data.
type DataSourceSchema struct {
	core_entity.BaseEntity
	Name             string            `json:"name,omitempty" gorm:"type:varchar(255);not null;uniqueIndex:idx_schema_name_source"`
	SourceIdentifier string            `json:"source_identifier,omitempty" gorm:"type:varchar(512);uniqueIndex:idx_schema_name_source"`
	SourceType       SourceType        `json:"source_type,omitempty" gorm:"type:varchar(50);not null;uniqueIndex:idx_schema_name_source"`
	Description      string            `json:"description,omitempty" gorm:"type:text"`
	SchemaDefinition []FieldDefinition `json:"schema_definition,omitempty" gorm:"type:jsonb;not null;serializer:json"`
}
