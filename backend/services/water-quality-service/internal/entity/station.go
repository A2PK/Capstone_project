package entity

import (
	core_entity "golang-microservices-boilerplate/pkg/core/entity" // Import the base entity
)

// Station represents a monitoring station.
type Station struct {
	core_entity.BaseEntity
	Name      string  `json:"name,omitempty" gorm:"type:varchar(255);not null;uniqueIndex:idx_name_lat_lon"`
	Latitude  float64 `json:"latitude,omitempty" gorm:"type:decimal(10,8);not null;uniqueIndex:idx_name_lat_lon"`
	Longitude float64 `json:"longitude,omitempty" gorm:"type:decimal(11,8);not null;uniqueIndex:idx_name_lat_lon"`
	Country   string  `json:"country,omitempty" gorm:"type:varchar(100);index"`
	Location  string  `json:"location,omitempty" gorm:"type:varchar(255);index"`
	// DataPoints []DataPoint `json:"data_points,omitempty" gorm:"foreignKey:StationID"` // Removed Has many DataPoints relationship
}
