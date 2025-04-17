package entity

import (
	core_entity "golang-microservices-boilerplate/pkg/core/entity" // Import the base entity
	"time"                                                         // Add time import

	"github.com/google/uuid"
	"gorm.io/gorm"
)

type AIModel struct {
	core_entity.BaseEntity
	Name         string    `json:"name,omitempty" gorm:"uniqueIndex:idx_name_version;not null"`
	Version      string    `json:"version,omitempty" gorm:"uniqueIndex:idx_name_version;not null"`
	FilePath     string    `json:"file_path,omitempty" gorm:"not null"` // Path in storage (set by the system)
	Description  string    `json:"description,omitempty" gorm:"type:text"`
	TrainedAt    time.Time `json:"trained_at,omitempty" gorm:"not null"`
	StationID    uuid.UUID `json:"station_id,omitempty" gorm:"type:uuid;not null;index"`
	Availability bool      `json:"availability,omitempty" gorm:"type:bool;default:true"`
}

// GORM hook to set UUID before creating
func (m *AIModel) BeforeCreate(tx *gorm.DB) (err error) {
	if m.ID == uuid.Nil {
		m.ID = uuid.New()
	}
	return
}
