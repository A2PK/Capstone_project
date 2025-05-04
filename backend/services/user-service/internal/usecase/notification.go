package usecase

import (
	"context"
	"fmt"

	core_logger "golang-microservices-boilerplate/pkg/core/logger"
	core_types "golang-microservices-boilerplate/pkg/core/types"
	core_usecase "golang-microservices-boilerplate/pkg/core/usecase"
	"golang-microservices-boilerplate/services/user-service/internal/entity"
	"golang-microservices-boilerplate/services/user-service/internal/repository"

	"github.com/google/uuid"
)

// NotificationUsecase defines the business logic operations for notifications.
type NotificationUsecase interface {
	core_usecase.BaseUseCase[entity.Notification] // Embed base CRUD

	// ListNotificationsForUser retrieves notifications for a specific user, optionally filtered by read status.
	ListNotificationsForUser(ctx context.Context, userID uuid.UUID, read *bool, opts core_types.FilterOptions) (*core_types.PaginationResult[entity.Notification], error)

	// MarkNotificationsAsRead marks specific notifications as read.
	MarkNotificationsAsRead(ctx context.Context, notificationIDs []uuid.UUID, readerID uuid.UUID) error

	// CreateNotification creates a new notification (might have specific logic beyond base Create).
	CreateNotification(ctx context.Context, userID uuid.UUID, title, description string, link *string) (*entity.Notification, error)

	// CreateMany creates multiple notification entities at once.
	CreateMany(ctx context.Context, notifications []*entity.Notification) ([]*entity.Notification, error)
}

// notificationUseCaseImpl implements the NotificationUsecase interface.
type notificationUseCaseImpl struct {
	*core_usecase.BaseUseCaseImpl[entity.Notification]
	repo   repository.NotificationRepository // Specific repo type
	logger core_logger.Logger
}

// NewNotificationUsecase creates a new instance of NotificationUsecase.
func NewNotificationUsecase(
	repo repository.NotificationRepository,
	logger core_logger.Logger,
) NotificationUsecase {
	baseUseCase := core_usecase.NewBaseUseCase(repo, logger)
	return &notificationUseCaseImpl{
		BaseUseCaseImpl: baseUseCase,
		repo:            repo,
		logger:          logger,
	}
}

// --- Implement Specific NotificationUsecase Methods ---

func (uc *notificationUseCaseImpl) ListNotificationsForUser(ctx context.Context, userID uuid.UUID, read *bool, opts core_types.FilterOptions) (*core_types.PaginationResult[entity.Notification], error) {
	uc.logger.Info("Listing notifications for user", "user_id", userID, "read_filter", read)
	result, err := uc.repo.FindByUser(ctx, userID, read, opts)
	if err != nil {
		uc.logger.Error("Failed to list notifications", "user_id", userID, "error", err)
		return nil, core_usecase.NewUseCaseError(core_usecase.ErrInternal, fmt.Sprintf("failed to retrieve notifications: %v", err))
	}
	uc.logger.Info("Successfully listed notifications", "user_id", userID, "count", len(result.Items))
	return result, nil
}

func (uc *notificationUseCaseImpl) MarkNotificationsAsRead(ctx context.Context, notificationIDs []uuid.UUID, readerID uuid.UUID) error {
	uc.logger.Info("Marking notifications as read", "notification_count", len(notificationIDs), "reader_id", readerID)
	if len(notificationIDs) == 0 {
		return nil
	}

	// Optional: Verify that the readerID owns these notifications before marking.

	err := uc.repo.MarkAsRead(ctx, notificationIDs)
	if err != nil {
		uc.logger.Error("Failed to mark notifications as read", "notification_count", len(notificationIDs), "reader_id", readerID, "error", err)
		return core_usecase.NewUseCaseError(core_usecase.ErrInternal, fmt.Sprintf("failed to mark notifications as read: %v", err))
	}

	uc.logger.Info("Successfully marked notifications as read", "notification_count", len(notificationIDs), "reader_id", readerID)
	return nil
}

func (uc *notificationUseCaseImpl) CreateNotification(ctx context.Context, userID uuid.UUID, title, description string, link *string) (*entity.Notification, error) {
	uc.logger.Info("Creating notification", "user_id", userID, "title", title)
	if description == "" || title == "" {
		return nil, core_usecase.NewUseCaseError(core_usecase.ErrInvalidInput, "notification title and description cannot be empty")
	}

	notification := &entity.Notification{
		UserID:      userID,
		Title:       title,
		Description: description,
		Read:        false,
		// Link field doesn't exist in entity
		// CreatedAt/UpdatedAt handled by BaseEntity hook
	}

	if err := uc.Create(ctx, notification); err != nil {
		uc.logger.Error("Failed to create notification", "user_id", userID, "title", title, "error", err)
		return nil, err // Create already returns UseCaseError
	}

	uc.logger.Info("Notification created successfully", "notification_id", notification.ID)
	return notification, nil
}

func (uc *notificationUseCaseImpl) CreateMany(ctx context.Context, notifications []*entity.Notification) ([]*entity.Notification, error) {
	uc.logger.Info("Creating multiple notifications", "notification_count", len(notifications))
	if len(notifications) == 0 {
		return nil, core_usecase.NewUseCaseError(core_usecase.ErrInvalidInput, "no notifications to create")
	}

	// Correctly handle both return values from repo.CreateMany
	createdNotifications, err := uc.repo.CreateMany(ctx, notifications)
	if err != nil {
		uc.logger.Error("Failed to create notifications", "notification_count", len(notifications), "error", err)
		// Return nil slice for entities on error
		return nil, core_usecase.NewUseCaseError(core_usecase.ErrInternal, fmt.Sprintf("failed to create notifications: %v", err))
	}

	uc.logger.Info("Successfully created notifications", "notification_count", len(createdNotifications))
	// Return the slice returned by the repository (which should have IDs populated)
	return createdNotifications, nil
}
