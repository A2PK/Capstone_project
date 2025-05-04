package repository

import (
	// Add if specific methods need it, BaseRepository methods have it

	"context"
	"errors"
	core_repo "golang-microservices-boilerplate/pkg/core/repository"
	core_types "golang-microservices-boilerplate/pkg/core/types"
	"golang-microservices-boilerplate/services/user-service/internal/entity"

	"fmt"

	"github.com/google/uuid"
	"gorm.io/gorm"
	// "github.com/google/uuid" // Add if needed for specific finders
)

// ArticleRepository defines specific persistence operations for Article entities.
type ArticleRepository interface {
	core_repo.BaseRepository[entity.Article] // Embed base CRUD
	// Add specific finders if needed, e.g.:
	// FindByAuthor(ctx context.Context, authorID uuid.UUID, opts core_types.FilterOptions) (*core_types.PaginationResult[entity.Article], error)
}

// gormArticleRepository implements ArticleRepository using GORM.
type gormArticleRepository struct {
	*core_repo.GormBaseRepository[entity.Article]
}

// NewArticleRepository creates a new ArticleRepository.
func NewArticleRepository(db *gorm.DB) ArticleRepository {
	return &gormArticleRepository{
		GormBaseRepository: core_repo.NewGormBaseRepository[entity.Article](db),
	}
}

// --- Implement ArticleRepository Specific Methods (if any) ---

// Override FindByID to preload Files
func (r *gormArticleRepository) FindByID(ctx context.Context, id uuid.UUID) (*entity.Article, error) {
	var article entity.Article
	// Use the embedded GormBaseRepository's DB instance and Preload
	result := r.DB.WithContext(ctx).Preload("Files").First(&article, id)
	if result.Error != nil {
		// Convert GORM error to a more generic one if needed
		if errors.Is(result.Error, gorm.ErrRecordNotFound) {
			// Use the standard gorm error for not found
			return nil, gorm.ErrRecordNotFound
		}
		return nil, result.Error
	}
	return &article, nil
}

// Override FindAll to preload Files
func (r *gormArticleRepository) FindAll(ctx context.Context, opts core_types.FilterOptions) (*core_types.PaginationResult[entity.Article], error) {
	var articles []*entity.Article
	var total int64

	// Base query for counting (applies filters)
	countQuery := r.DB.WithContext(ctx).Model(&entity.Article{})
	if !opts.IncludeDeleted {
		countQuery = countQuery.Where("deleted_at IS NULL")
	}
	if len(opts.Filters) > 0 {
		countQuery = countQuery.Where(opts.Filters) // Apply filters only for count
	}
	if err := countQuery.Count(&total).Error; err != nil {
		return nil, fmt.Errorf("failed to count articles: %w", err)
	}

	// Base query for finding data
	query := r.DB.WithContext(ctx).Model(&entity.Article{})
	if !opts.IncludeDeleted {
		query = query.Where("deleted_at IS NULL")
	}

	// Preload Files relationship
	query = query.Preload("Files")

	// Apply all options (filters, sorting, pagination) using the base helper
	query = r.ApplyFiltersOptions(query, opts)

	// Find the results
	if err := query.Find(&articles).Error; err != nil {
		return nil, fmt.Errorf("failed to find articles: %w", err)
	}

	// Use values from opts, providing defaults if necessary
	limit := opts.Limit
	if limit <= 0 {
		limit = 50 // Default limit
	}
	offset := opts.Offset
	if offset < 0 {
		offset = 0 // Default offset
	}

	return &core_types.PaginationResult[entity.Article]{
		Items:      articles,
		TotalItems: total,
		Limit:      limit,
		Offset:     offset,
	}, nil
}

// Example specific finder:
/*
func (r *gormArticleRepository) FindByAuthor(ctx context.Context, authorID uuid.UUID, opts core_types.FilterOptions) (*core_types.PaginationResult[entity.Article], error) {
	filter := map[string]interface{}{"author_id": authorID}
	// Add other filters from opts.Filters if necessary
	for k, v := range opts.Filters {
		if k != "author_id" { // Avoid overwriting
			filter[k] = v
		}
	}
	return r.FindWithFilter(ctx, filter, opts) // Assumes FindWithFilter exists in BaseRepo or needs implementation
}
*/
