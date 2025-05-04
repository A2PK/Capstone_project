package entity

import (
	coreEntity "golang-microservices-boilerplate/pkg/core/entity"
	// "github.com/google/uuid" // No longer needed directly here
)

type Article struct {
	coreEntity.BaseEntity
	Title      string `json:"title" gorm:"not null"`
	Content    string `json:"content" gorm:"not null"`
	AuthorID   string `json:"author_id" gorm:"not null"`
	PictureURL string `json:"picture_url" gorm:"not null"`
	Files      []File `json:"files" gorm:"many2many:article_files;constraint:OnUpdate:CASCADE,OnDelete:CASCADE;"`
	Badge      string `json:"badge" gorm:"not null;default:'common';check:chk_article_badge,badge IN ('good', 'danger', 'common')"`
}
