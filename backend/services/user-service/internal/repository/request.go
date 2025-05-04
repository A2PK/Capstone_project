package repository

import (
	"context" // Add if specific methods need it, BaseRepository methods have it
	"errors"
	"fmt"
	core_repo "golang-microservices-boilerplate/pkg/core/repository"
	core_types "golang-microservices-boilerplate/pkg/core/types"
	"golang-microservices-boilerplate/services/user-service/internal/entity"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

// RequestRepository defines specific persistence operations for Request entities.
type RequestRepository interface {
	core_repo.BaseRepository[entity.Request] // Embed base CRUD

	// FindByUserParticipant retrieves requests where the user is either the sender or receiver, potentially filtering by status.
	FindByUserParticipant(ctx context.Context, userID uuid.UUID, status *string, opts core_types.FilterOptions) (*core_types.PaginationResult[entity.Request], error)
}

// gormRequestRepository implements RequestRepository using GORM.
type gormRequestRepository struct {
	*core_repo.GormBaseRepository[entity.Request]
}

// NewRequestRepository creates a new RequestRepository.
func NewRequestRepository(db *gorm.DB) RequestRepository {
	return &gormRequestRepository{
		GormBaseRepository: core_repo.NewGormBaseRepository[entity.Request](db),
	}
}

// --- Implement RequestRepository Specific Methods ---

// Renamed from FindByUser and updated logic
func (r *gormRequestRepository) FindByUserParticipant(ctx context.Context, userID uuid.UUID, status *string, opts core_types.FilterOptions) (*core_types.PaginationResult[entity.Request], error) {
	var requests []*entity.Request
	var total int64

	// Base query for both counting and finding data
	baseQuery := r.DB.WithContext(ctx).Model(&entity.Request{}).Where("sender_id = ? OR receiver_id = ?", userID, userID)

	// Apply status filter if provided
	if status != nil {
		baseQuery = baseQuery.Where("status = ?", *status)
	}

	// Count total matching records before applying pagination/sorting
	countQuery := baseQuery
	if err := countQuery.Count(&total).Error; err != nil {
		return nil, fmt.Errorf("failed to count requests by participant: %w", err)
	}

	// Apply pagination and sorting using base helper
	// Preload Files relationship
	findQuery := baseQuery.Preload("Files")
	findQuery = r.ApplyFiltersOptions(findQuery, opts) // Apply sorting/limit/offset

	// Find the results
	if err := findQuery.Find(&requests).Error; err != nil {
		return nil, fmt.Errorf("failed to find requests by participant: %w", err)
	}

	limit := opts.Limit
	if limit <= 0 {
		limit = 50 // Default limit
	}
	offset := opts.Offset
	if offset < 0 {
		offset = 0 // Default offset
	}

	return &core_types.PaginationResult[entity.Request]{
		Items:      requests,
		TotalItems: total,
		Limit:      limit,
		Offset:     offset,
	}, nil
}

// --- Override Base Methods to Preload Files ---

// Override FindByID to preload Files
func (r *gormRequestRepository) FindByID(ctx context.Context, id uuid.UUID) (*entity.Request, error) {
	var request entity.Request
	result := r.DB.WithContext(ctx).Preload("Files").First(&request, id)
	if result.Error != nil {
		if errors.Is(result.Error, gorm.ErrRecordNotFound) {
			return nil, gorm.ErrRecordNotFound // Use standard GORM error
		}
		return nil, fmt.Errorf("failed to find request by ID: %w", result.Error)
	}
	return &request, nil
}

// Override FindAll to preload Files
func (r *gormRequestRepository) FindAll(ctx context.Context, opts core_types.FilterOptions) (*core_types.PaginationResult[entity.Request], error) {
	var requests []*entity.Request
	var total int64

	// Base query for counting (applies filters)
	countQuery := r.DB.WithContext(ctx).Model(&entity.Request{})
	if !opts.IncludeDeleted {
		countQuery = countQuery.Where("deleted_at IS NULL")
	}
	if len(opts.Filters) > 0 {
		countQuery = countQuery.Where(opts.Filters)
	}
	if err := countQuery.Count(&total).Error; err != nil {
		return nil, fmt.Errorf("failed to count requests: %w", err)
	}

	// Base query for finding data
	query := r.DB.WithContext(ctx).Model(&entity.Request{})
	if !opts.IncludeDeleted {
		query = query.Where("deleted_at IS NULL")
	}

	// Preload Files relationship
	query = query.Preload("Files")

	// Apply all options (filters, sorting, pagination) using the base helper
	query = r.ApplyFiltersOptions(query, opts)

	// Find the results
	if err := query.Find(&requests).Error; err != nil {
		return nil, fmt.Errorf("failed to find requests: %w", err)
	}

	limit := opts.Limit
	if limit <= 0 {
		limit = 50 // Default limit
	}
	offset := opts.Offset
	if offset < 0 {
		offset = 0 // Default offset
	}

	return &core_types.PaginationResult[entity.Request]{
		Items:      requests,
		TotalItems: total,
		Limit:      limit,
		Offset:     offset,
	}, nil
}
