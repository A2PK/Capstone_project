package usecase

import (
	"context"
	"fmt"

	// Import time for LastUpdated

	core_logger "golang-microservices-boilerplate/pkg/core/logger"
	core_types "golang-microservices-boilerplate/pkg/core/types"
	core_usecase "golang-microservices-boilerplate/pkg/core/usecase"

	// corePb "golang-microservices-boilerplate/proto/core" // No longer needed for PaginationInfo here
	// pb "golang-microservices-boilerplate/proto/user-service" // No longer needed
	"golang-microservices-boilerplate/services/user-service/internal/entity"
	"golang-microservices-boilerplate/services/user-service/internal/repository"

	// "golang-microservices-boilerplate/services/user-service/internal/schema" // Remove unused schema import

	"github.com/google/uuid"
	// "google.golang.org/protobuf/types/known/timestamppb" // No longer needed
)

// ChatUsecase defines the business logic operations for chat messages.
type ChatUsecase interface {
	core_usecase.BaseUseCase[entity.ChatMessage] // Use ChatMessage

	// ListUnseenMessagesForUser retrieves all unread messages for a specific user.
	ListUnseenMessagesForUser(ctx context.Context, userID uuid.UUID) ([]*entity.ChatMessage, error) // Use ChatMessage

	// ListMessagesBetweenUsers retrieves the conversation history between two users.
	ListMessagesBetweenUsers(ctx context.Context, userID1, userID2 uuid.UUID, opts core_types.FilterOptions) (*core_types.PaginationResult[entity.ChatMessage], error) // Use ChatMessage

	// SendMessage creates a new chat message.
	SendMessage(ctx context.Context, senderID, receiverID uuid.UUID, message string) (*entity.ChatMessage, error) // Use ChatMessage

	// MarkMessagesAsRead marks specific messages as read.
	MarkMessagesAsRead(ctx context.Context, messageIDs []uuid.UUID, readerID uuid.UUID) error
}

// chatUseCaseImpl implements the ChatUsecase interface.
type chatUseCaseImpl struct {
	*core_usecase.BaseUseCaseImpl[entity.ChatMessage]                           // Use ChatMessage
	repo                                              repository.ChatRepository // Specific repo type
	userRepo                                          repository.UserRepository // Inject UserRepository to fetch user details
	logger                                            core_logger.Logger
}

// NewChatUsecase creates a new instance of ChatUsecase.
func NewChatUsecase(
	repo repository.ChatRepository,
	userRepo repository.UserRepository, // Add userRepo dependency
	logger core_logger.Logger,
) ChatUsecase {
	baseUseCase := core_usecase.NewBaseUseCase(repo, logger)
	return &chatUseCaseImpl{
		BaseUseCaseImpl: baseUseCase,
		repo:            repo,
		userRepo:        userRepo, // Store userRepo
		logger:          logger,
	}
}

// --- Implement Specific ChatUsecase Methods ---

func (uc *chatUseCaseImpl) ListUnseenMessagesForUser(ctx context.Context, userID uuid.UUID) ([]*entity.ChatMessage, error) { // Use ChatMessage
	uc.logger.Info("Listing unseen messages", "user_id", userID)
	messages, err := uc.repo.FindUnseenByReceiver(ctx, userID)
	if err != nil {
		uc.logger.Error("Failed to list unseen messages", "user_id", userID, "error", err)
		// Consider wrapping error using core_usecase helpers if not already done by repo
		return nil, err // Return repo error directly or wrap it
	}
	uc.logger.Info("Successfully listed unseen messages", "user_id", userID, "count", len(messages))
	return messages, nil
}

func (uc *chatUseCaseImpl) ListMessagesBetweenUsers(ctx context.Context, userID1, userID2 uuid.UUID, opts core_types.FilterOptions) (*core_types.PaginationResult[entity.ChatMessage], error) { // Use ChatMessage
	uc.logger.Info("Listing messages between users", "user_id1", userID1, "user_id2", userID2)
	result, err := uc.repo.FindConversation(ctx, userID1, userID2, opts)
	if err != nil {
		uc.logger.Error("Failed to list messages between users", "user_id1", userID1, "user_id2", userID2, "error", err)
		// Consider wrapping error
		return nil, err // Return repo error directly or wrap it
	}
	uc.logger.Info("Successfully listed messages between users", "user_id1", userID1, "user_id2", userID2, "count", len(result.Items))
	return result, nil
}

func (uc *chatUseCaseImpl) SendMessage(ctx context.Context, senderID, receiverID uuid.UUID, message string) (*entity.ChatMessage, error) { // Use ChatMessage
	uc.logger.Info("Sending message", "sender_id", senderID, "receiver_id", receiverID)
	if message == "" {
		// Reverted: Use fmt.Errorf for validation until ErrValidation is confirmed/fixed
		return nil, fmt.Errorf("validation error: message content cannot be empty")
	}
	if senderID == receiverID {
		// Reverted: Use fmt.Errorf for validation
		return nil, fmt.Errorf("validation error: sender and receiver cannot be the same")
	}

	// TODO: Check if receiverID exists? Depends on requirements.

	chat := &entity.ChatMessage{ // Use ChatMessage
		SenderID:   senderID,   // Use string UUID
		ReceiverID: receiverID, // Use string UUID
		Message:    message,
		Read:       false, // Default to unread
	}

	if err := uc.Create(ctx, chat); err != nil {
		uc.logger.Error("Failed to create chat message", "sender_id", senderID, "receiver_id", receiverID, "error", err)
		return nil, err // Return wrapped error
	}

	uc.logger.Info("Message sent successfully", "chat_id", chat.ID)
	return chat, nil
}

func (uc *chatUseCaseImpl) MarkMessagesAsRead(ctx context.Context, messageIDs []uuid.UUID, readerID uuid.UUID) error {
	uc.logger.Info("Attempting to mark messages as read", "message_count", len(messageIDs), "reader_id", readerID)
	if len(messageIDs) == 0 {
		uc.logger.Info("No message IDs provided to mark as read")
		return nil
	}

	// Optional: Verify that the messages actually belong to the readerID as the receiver
	// This would involve fetching the messages first. For performance, skipping for now.
	// filter := map[string]interface{}{"id": messageIDs, "receiver_id": readerID.String()}
	// messages, err := uc.FindWithFilter(ctx, filter, core_types.FilterOptions{Limit: len(messageIDs)})
	// if err != nil { ... }
	// if len(messages.Items) != len(messageIDs) { // Some messages don't exist or don't belong to the reader }
	// idsToUpdate := make([]uuid.UUID, len(messages.Items))
	// for i, msg := range messages.Items { idsToUpdate[i] = msg.ID }
	// err = uc.repo.MarkAsRead(ctx, idsToUpdate)

	err := uc.repo.MarkAsRead(ctx, messageIDs)
	if err != nil {
		uc.logger.Error("Failed to mark messages as read in repository", "message_count", len(messageIDs), "reader_id", readerID, "error", err)
		// Consider wrapping error
		return err
	}

	uc.logger.Info("Successfully marked messages as read", "message_count", len(messageIDs), "reader_id", readerID)
	return nil
}
