package entity

import (
	coreEntity "golang-microservices-boilerplate/pkg/core/entity"

	"github.com/google/uuid"
)

type Notification struct {
	coreEntity.BaseEntity
	Title       string    `json:"title" gorm:"not null"`
	Description string    `json:"description" gorm:"not null"`
	Read        bool      `json:"read" gorm:"default:false"`
	UserID      uuid.UUID `json:"user_id" gorm:"not null"`
	User        User      `json:"user" gorm:"foreignKey:UserID"`
}
