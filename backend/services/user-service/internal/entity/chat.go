package entity

import (
	coreEntity "golang-microservices-boilerplate/pkg/core/entity"

	"github.com/google/uuid"
)

type ChatMessage struct {
	coreEntity.BaseEntity
	SenderID   uuid.UUID `json:"sender_id" gorm:"not null"`
	ReceiverID uuid.UUID `json:"receiver_id" gorm:"not null"`
	Message    string    `json:"message" gorm:"not null"`
	Read       bool      `json:"read" gorm:"default:false"`
}
