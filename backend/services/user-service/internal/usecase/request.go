package usecase

import (
	"context"
	"fmt"

	core_logger "golang-microservices-boilerplate/pkg/core/logger"
	core_repo "golang-microservices-boilerplate/pkg/core/repository"
	core_types "golang-microservices-boilerplate/pkg/core/types"
	core_usecase "golang-microservices-boilerplate/pkg/core/usecase"
	"golang-microservices-boilerplate/services/user-service/internal/entity"
	"golang-microservices-boilerplate/services/user-service/internal/repository"

	"github.com/google/uuid"
)

// RequestUsecase defines the business logic operations for user requests.
type RequestUsecase interface {
	core_usecase.BaseUseCase[entity.Request] // Embed base CRUD

	// CreateUserRequest creates a new request initiated by a specific user.
	// DEPRECATED: Use CreateRequestWithFiles instead to handle associations correctly.
	// CreateUserRequest(ctx context.Context, senderID uuid.UUID, req *entity.Request) error

	// CreateRequestWithFiles creates a new request and associates provided file IDs.
	CreateRequestWithFiles(ctx context.Context, request *entity.Request, fileIDs []string) (*entity.Request, error)

	// ListUserRequests retrieves requests relevant to a user (sent or received).
	ListUserRequests(ctx context.Context, userID uuid.UUID, direction string, status *string, opts core_types.FilterOptions) (*core_types.PaginationResult[entity.Request], error)

	// RespondToUserRequest allows a receiver to accept or reject a request.
	RespondToUserRequest(ctx context.Context, requestID uuid.UUID, receiverID uuid.UUID, accept bool) error

	// DeleteUserRequest allows the sender to delete (cancel) their request.
	DeleteUserRequest(ctx context.Context, requestID uuid.UUID, senderID uuid.UUID, hardDelete bool) error

	FindByUserParticipant(ctx context.Context, userID uuid.UUID, status *string, opts core_types.FilterOptions) (*core_types.PaginationResult[entity.Request], error)
}

// requestUseCaseImpl implements the RequestUsecase interface.
type requestUseCaseImpl struct {
	*core_usecase.BaseUseCaseImpl[entity.Request]
	repo     repository.RequestRepository // Specific repo type
	fileRepo repository.FileRepository    // Add FileRepository
	logger   core_logger.Logger
}

// NewRequestUsecase creates a new instance of RequestUsecase.
func NewRequestUsecase(
	repo repository.RequestRepository,
	fileRepo repository.FileRepository, // Add FileRepository
	logger core_logger.Logger,
) RequestUsecase {
	baseUseCase := core_usecase.NewBaseUseCase(repo, logger)
	return &requestUseCaseImpl{
		BaseUseCaseImpl: baseUseCase,
		repo:            repo,
		fileRepo:        fileRepo, // Store FileRepository
		logger:          logger,
	}
}

// --- Implement Specific RequestUsecase Methods ---

// CreateRequestWithFiles handles the creation including file associations.
func (uc *requestUseCaseImpl) CreateRequestWithFiles(ctx context.Context, request *entity.Request, fileIDs []string) (*entity.Request, error) {
	if request == nil {
		return nil, core_usecase.NewUseCaseError(core_usecase.ErrInvalidInput, "request data cannot be nil")
	}
	if request.SenderID == uuid.Nil {
		return nil, core_usecase.NewUseCaseError(core_usecase.ErrInvalidInput, "sender ID cannot be nil")
	}
	if request.ReceiverID == uuid.Nil {
		return nil, core_usecase.NewUseCaseError(core_usecase.ErrInvalidInput, "receiver ID cannot be nil")
	}
	if request.Title == "" {
		return nil, core_usecase.NewUseCaseError(core_usecase.ErrInvalidInput, "request title cannot be empty")
	}
	if request.Status == "" { // Default status if not provided
		request.Status = "pending"
	}

	uc.logger.Info("Attempting to create request with files", "title", request.Title, "sender_id", request.SenderID, "receiver_id", request.ReceiverID, "file_count", len(fileIDs))

	err := uc.Repository.Transaction(ctx, func(txRepo core_repo.BaseRepository[entity.Request]) error {
		// 1. Create the request within the transaction
		if err := txRepo.Create(ctx, request); err != nil {
			uc.logger.Error("Failed to create request in transaction", "error", err)
			return err // Rollback
		}
		uc.logger.Debug("Request created in transaction", "request_id", request.ID)

		// 2. Handle file associations if IDs are provided
		if len(fileIDs) > 0 {
			var filesToAssociate []*entity.File
			for _, idStr := range fileIDs {
				fileID, parseErr := uuid.Parse(idStr)
				if parseErr != nil {
					uc.logger.Warn("Invalid file ID format provided", "file_id_str", idStr)
					return core_usecase.NewUseCaseError(core_usecase.ErrInvalidInput, fmt.Sprintf("invalid file ID format: %s", idStr)) // Rollback
				}

				// Find the file using the FileRepository (needs access within Tx context ideally)
				file, findErr := uc.fileRepo.FindByID(ctx, fileID) // Use the injected fileRepo
				if findErr != nil {
					uc.logger.Error("Failed to find file for association", "file_id", fileID, "error", findErr)
					if findErr.Error() == "entity not found" { // Adjust error check
						return core_usecase.NewUseCaseError(core_usecase.ErrInvalidInput, fmt.Sprintf("file with ID %s not found", fileID))
					}
					return findErr // Rollback on other errors
				}
				filesToAssociate = append(filesToAssociate, file)
			}

			// 3. Associate found files with the request
			gormTxRepo, ok := txRepo.(*core_repo.GormBaseRepository[entity.Request])
			if !ok {
				return fmt.Errorf("unexpected repository type in transaction")
			}
			txDB := gormTxRepo.DB

			uc.logger.Debug("Associating files with request", "request_id", request.ID, "file_count", len(filesToAssociate))
			if assocErr := txDB.Model(request).Association("Files").Append(filesToAssociate); assocErr != nil {
				uc.logger.Error("Failed to append file association", "request_id", request.ID, "error", assocErr)
				return fmt.Errorf("failed to associate files: %w", assocErr) // Rollback
			}
		}
		return nil // Commit transaction
	})

	if err != nil {
		return nil, err // Transaction failed
	}

	uc.logger.Info("Successfully created request with files", "request_id", request.ID)
	return request, nil
}

func (uc *requestUseCaseImpl) FindByUserParticipant(ctx context.Context, userID uuid.UUID, status *string, opts core_types.FilterOptions) (*core_types.PaginationResult[entity.Request], error) {
	uc.logger.Info("Finding requests by user participant", "user_id", userID, "status", status)

	result, err := uc.repo.FindByUserParticipant(ctx, userID, status, opts)
	if err != nil {
		uc.logger.Error("Failed to find requests by user participant", "user_id", userID, "error", err)
		return nil, err
	}

	uc.logger.Info("Successfully found requests by user participant", "user_id", userID, "count", len(result.Items))
	return result, nil
}

// ListUserRequests uses the specific repository method to find requests involving the user.
func (uc *requestUseCaseImpl) ListUserRequests(ctx context.Context, userID uuid.UUID, direction string, status *string, opts core_types.FilterOptions) (*core_types.PaginationResult[entity.Request], error) {
	uc.logger.Info("Listing user requests", "user_id", userID, "direction", direction, "status", status)

	var result *core_types.PaginationResult[entity.Request]
	var err error

	// TODO: Refine filtering logic based on 'direction' if needed.
	// The FindByUserParticipant already finds where user is sender OR receiver.
	// Add specific filters based on 'direction' if FindByUserParticipant doesn't suffice.
	if direction == "sent" {
		// Modify opts or use a different repo method if only sent needed
		filter := map[string]interface{}{"sender_id": userID}
		if status != nil {
			filter["status"] = *status
		}
		result, err = uc.FindWithFilter(ctx, filter, opts) // Use base FindWithFilter for now
	} else if direction == "received" {
		// Modify opts or use a different repo method if only received needed
		filter := map[string]interface{}{"receiver_id": userID}
		if status != nil {
			filter["status"] = *status
		}
		result, err = uc.FindWithFilter(ctx, filter, opts) // Use base FindWithFilter for now
	} else {
		// Default: Use the repo method that finds participation as sender OR receiver
		result, err = uc.repo.FindByUserParticipant(ctx, userID, status, opts)
	}

	if err != nil {
		uc.logger.Error("Failed to list user requests", "user_id", userID, "error", err)
		return nil, err // Return original error
	}

	uc.logger.Info("Successfully listed user requests", "user_id", userID, "count", len(result.Items))
	return result, nil
}

// RespondToUserRequest updates the status of a request.
func (uc *requestUseCaseImpl) RespondToUserRequest(ctx context.Context, requestID uuid.UUID, receiverID uuid.UUID, accept bool) error {
	uc.logger.Info("Responding to user request", "request_id", requestID, "receiver_id", receiverID, "accept", accept)

	// 1. Get the request
	req, err := uc.GetByID(ctx, requestID)
	if err != nil {
		// Handles not found error via base GetByID
		uc.logger.Error("Failed to get request for response", "request_id", requestID, "error", err)
		return err
	}

	// 2. Check permissions: Is the user the intended receiver?
	if req.ReceiverID != receiverID {
		uc.logger.Warn("Permission denied to respond to request", "request_id", requestID, "actual_receiver", req.ReceiverID, "requester", receiverID)
		return core_usecase.NewUseCaseError(core_usecase.ErrForbidden, "you are not the recipient of this request")
	}

	// 3. Check status: Can only respond to pending requests?
	if req.Status != "pending" {
		uc.logger.Warn("Attempted to respond to non-pending request", "request_id", requestID, "current_status", req.Status)
		return core_usecase.NewUseCaseError(core_usecase.ErrConflict, fmt.Sprintf("request is already in '%s' status", req.Status))
	}

	// 4. Update status
	if accept {
		req.Status = "accepted"
	} else {
		req.Status = "rejected"
	}

	// Use the embedded Update method
	err = uc.Update(ctx, req)
	if err != nil {
		uc.logger.Error("Failed to update request status", "request_id", requestID, "error", err)
		return err
	}

	uc.logger.Info("Successfully responded to user request", "request_id", requestID, "new_status", req.Status)
	return nil
}

// DeleteUserRequest allows the sender to delete their own request.
func (uc *requestUseCaseImpl) DeleteUserRequest(ctx context.Context, requestID uuid.UUID, senderID uuid.UUID, hardDelete bool) error {
	uc.logger.Info("Deleting user request", "request_id", requestID, "sender_id", senderID, "hardDelete", hardDelete)

	// 1. Get the request to verify ownership
	req, err := uc.GetByID(ctx, requestID)
	if err != nil {
		// Handles not found error via base GetByID
		uc.logger.Error("Failed to get request for deletion", "request_id", requestID, "error", err)
		return err
	}

	// 2. Check permissions: Is the user the sender?
	if req.SenderID != senderID {
		uc.logger.Warn("Permission denied to delete request", "request_id", requestID, "actual_sender", req.SenderID, "requester", senderID)
		return core_usecase.NewUseCaseError(core_usecase.ErrForbidden, "you did not send this request")
	}

	// 3. Perform deletion using the embedded Delete method
	err = uc.Delete(ctx, requestID, hardDelete)
	if err != nil {
		uc.logger.Error("Failed to delete user request", "request_id", requestID, "error", err)
		return err
	}

	uc.logger.Info("Successfully deleted user request", "request_id", requestID)
	return nil
}
