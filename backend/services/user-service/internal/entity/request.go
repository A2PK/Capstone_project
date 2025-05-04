package entity

import (
	coreEntity "golang-microservices-boilerplate/pkg/core/entity"

	"github.com/google/uuid"
)

type Request struct {
	coreEntity.BaseEntity
	Title       string    `json:"title" gorm:"not null"`
	Description string    `json:"description" gorm:"not null"`
	SenderID    uuid.UUID `json:"sender_id" gorm:"not null"`
	ReceiverID  uuid.UUID `json:"receiver_id" gorm:"not null"`
	Status      string    `json:"status" gorm:"not null"`
	Files       []File    `json:"files" gorm:"many2many:request_files;constraint:OnUpdate:CASCADE,OnDelete:CASCADE;"`
}
