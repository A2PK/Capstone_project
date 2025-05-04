package repository

import (
	"context"

	core_repo "golang-microservices-boilerplate/pkg/core/repository"
	core_types "golang-microservices-boilerplate/pkg/core/types"
	"golang-microservices-boilerplate/services/user-service/internal/entity"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

// ChatRepository defines specific persistence operations for ChatMessage entities.
type ChatRepository interface {
	core_repo.BaseRepository[entity.ChatMessage] // Use ChatMessage

	// FindUnseenByReceiver retrieves all chat messages for a receiver that are marked as unread.
	FindUnseenByReceiver(ctx context.Context, receiverID uuid.UUID) ([]*entity.ChatMessage, error) // Use ChatMessage

	// FindConversation retrieves all messages between two users, ordered by SentAt.
	FindConversation(ctx context.Context, userID1, userID2 uuid.UUID, opts core_types.FilterOptions) (*core_types.PaginationResult[entity.ChatMessage], error) // Use ChatMessage

	// MarkAsRead marks specific chat messages as read.
	MarkAsRead(ctx context.Context, messageIDs []uuid.UUID) error
}

// gormChatRepository implements ChatRepository using GORM.
type gormChatRepository struct {
	*core_repo.GormBaseRepository[entity.ChatMessage] // Use ChatMessage
}

// NewChatRepository creates a new ChatRepository.
func NewChatRepository(db *gorm.DB) ChatRepository {
	return &gormChatRepository{
		GormBaseRepository: core_repo.NewGormBaseRepository[entity.ChatMessage](db), // Use ChatMessage
	}
}

// --- Implement ChatRepository Specific Methods ---

func (r *gormChatRepository) FindUnseenByReceiver(ctx context.Context, receiverID uuid.UUID) ([]*entity.ChatMessage, error) { // Use ChatMessage
	var chats []*entity.ChatMessage // Use ChatMessage
	err := r.DB.WithContext(ctx).
		Where("receiver_id = ? AND read = ?", receiverID.String(), false). // Use correct field name 'Read' and convert UUID to string
		Order("created_at ASC").                                           // Order by created_at (assuming BaseEntity)
		Find(&chats).Error
	if err != nil {
		return nil, err
	}
	return chats, nil
}

func (r *gormChatRepository) FindConversation(ctx context.Context, userID1, userID2 uuid.UUID, opts core_types.FilterOptions) (*core_types.PaginationResult[entity.ChatMessage], error) { // Use ChatMessage
	var chats []*entity.ChatMessage // Use ChatMessage
	var total int64

	// Convert UUIDs to strings for query
	userID1Str := userID1.String()
	userID2Str := userID2.String()

	query := r.DB.WithContext(ctx).Model(&entity.ChatMessage{}). // Use ChatMessage
									Where("(sender_id = ? AND receiver_id = ?) OR (sender_id = ? AND receiver_id = ?)", userID1Str, userID2Str, userID2Str, userID1Str)

	// Count total matching records before applying pagination
	if err := query.Count(&total).Error; err != nil {
		return nil, err
	}

	// Apply pagination and sorting (default to created_at ASC)
	// Replace ApplyGormOptions with direct GORM methods
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
		query = query.Order("created_at ASC")
	}

	// Find the results
	if err := query.Find(&chats).Error; err != nil {
		return nil, err
	}

	return &core_types.PaginationResult[entity.ChatMessage]{ // Use ChatMessage
		Items:      chats,
		TotalItems: total,
		Limit:      opts.Limit,
		Offset:     opts.Offset,
	}, nil
}

func (r *gormChatRepository) MarkAsRead(ctx context.Context, messageIDs []uuid.UUID) error {
	if len(messageIDs) == 0 {
		return nil // Nothing to mark
	}
	err := r.DB.WithContext(ctx).
		Model(&entity.ChatMessage{}). // Use ChatMessage
		Where("id IN ?", messageIDs).
		Update("read", true).Error // Use correct field name 'Read'
	return err
}
