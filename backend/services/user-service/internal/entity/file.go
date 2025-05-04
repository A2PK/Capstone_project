package entity

import (
	coreEntity "golang-microservices-boilerplate/pkg/core/entity"

	"github.com/google/uuid"
)

type File struct {
	coreEntity.BaseEntity
	Name              string    `json:"name" gorm:"not null"`
	ServiceInternalID string    `json:"service_internal_id" gorm:"not null"`
	Type              string    `json:"type" gorm:"not null"`
	Size              int64     `json:"size" gorm:"not null"` // in bytes
	UploaderID        uuid.UUID `json:"uploader_id" gorm:"not null"`
	URL               string    `json:"url" gorm:"not null"`
}
