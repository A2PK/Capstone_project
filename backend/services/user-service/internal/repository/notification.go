package repository

import (
	"context" // Add if specific methods need it, BaseRepository methods have it

	core_repo "golang-microservices-boilerplate/pkg/core/repository"
	core_types "golang-microservices-boilerplate/pkg/core/types"
	"golang-microservices-boilerplate/services/user-service/internal/entity"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

// NotificationRepository defines specific persistence operations for Notification entities.
type NotificationRepository interface {
	core_repo.BaseRepository[entity.Notification] // Embed base CRUD

	// FindByUser retrieves notifications for a specific user, potentially filtering by read status.
	FindByUser(ctx context.Context, userID uuid.UUID, read *bool, opts core_types.FilterOptions) (*core_types.PaginationResult[entity.Notification], error)

	// MarkAsRead marks specific notifications as read.
	MarkAsRead(ctx context.Context, notificationIDs []uuid.UUID) error
}

// gormNotificationRepository implements NotificationRepository using GORM.
type gormNotificationRepository struct {
	*core_repo.GormBaseRepository[entity.Notification]
}

// NewNotificationRepository creates a new NotificationRepository.
func NewNotificationRepository(db *gorm.DB) NotificationRepository {
	return &gormNotificationRepository{
		GormBaseRepository: core_repo.NewGormBaseRepository[entity.Notification](db),
	}
}

// --- Implement NotificationRepository Specific Methods ---

func (r *gormNotificationRepository) FindByUser(ctx context.Context, userID uuid.UUID, read *bool, opts core_types.FilterOptions) (*core_types.PaginationResult[entity.Notification], error) {
	var notifications []*entity.Notification
	var total int64

	query := r.DB.WithContext(ctx).Model(&entity.Notification{}).
		Where("user_id = ?", userID)

	// Apply read status filter if provided
	if read != nil {
		query = query.Where("read = ?", *read)
	}

	// Count total matching records before applying pagination
	if err := query.Count(&total).Error; err != nil {
		return nil, err
	}

	// Apply pagination and sorting (default to created_at DESC for notifications)
	if opts.Limit > 0 {
		query = query.Limit(opts.Limit)
	}
	query = query.Offset(opts.Offset)

	if opts.SortBy != "" {
		order := opts.SortBy
		if opts.SortDesc {
			order += " DESC"
		}
		query = query.Order(order)
	} else {
		// Default sort order
		query = query.Order("created_at DESC")
	}

	// Find the results
	if err := query.Find(&notifications).Error; err != nil {
		return nil, err
	}

	return &core_types.PaginationResult[entity.Notification]{
		Items:      notifications,
		TotalItems: total,
		Limit:      opts.Limit,
		Offset:     opts.Offset,
	}, nil
}

func (r *gormNotificationRepository) MarkAsRead(ctx context.Context, notificationIDs []uuid.UUID) error {
	if len(notificationIDs) == 0 {
		return nil // Nothing to mark
	}
	err := r.DB.WithContext(ctx).
		Model(&entity.Notification{}).
		Where("id IN ?", notificationIDs).
		Update("read", true).Error
	return err
}
