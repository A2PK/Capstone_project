package entity

import (
	coreEntity "golang-microservices-boilerplate/pkg/core/entity" // Import the base entity
)

// ThresholdConfig represents water quality parameter threshold configuration.
type ThresholdConfig struct {
	coreEntity.BaseEntity
	ElementName string  `json:"element_name,omitempty" gorm:"type:varchar(100);not null;uniqueIndex:idx_element_name_lower,expression:lower(element_name)"`
	MinValue    float64 `json:"min_value,omitempty" gorm:"type:decimal(12,6);not null"`
	MaxValue    float64 `json:"max_value,omitempty" gorm:"type:decimal(12,6);not null"`
}
