package usecase

import (
	"context"
	"fmt"
	"io" // Use structured logger ideally

	core_types "golang-microservices-boilerplate/pkg/core/types" // Import core types for FilterOptions

	"github.com/google/uuid"
	"google.golang.org/api/drive/v3"
	"google.golang.org/api/option"

	coreLogger "golang-microservices-boilerplate/pkg/core/logger"
	coreUsecase "golang-microservices-boilerplate/pkg/core/usecase"
	"golang-microservices-boilerplate/services/user-service/internal/entity"
	"golang-microservices-boilerplate/services/user-service/internal/repository"
)

// FileUsecase defines the business logic operations for files, including cloud storage.
type FileUsecase interface {
	coreUsecase.BaseUseCase[entity.File] // Embed base CRUD
	UploadFile(ctx context.Context, fileName string, fileContent io.Reader, fileSize int64, mimeType string, uploaderID uuid.UUID) (*entity.File, error)
	// GetFileLink is implicitly handled by GetByID returning the entity with the URL
	// Delete needs to be overridden to handle cloud deletion
	DeleteFile(ctx context.Context, fileID uuid.UUID, hardDelete bool) error // Renamed to avoid conflict
	// ListUserFiles retrieves files uploaded by a specific user.
	ListUserFiles(ctx context.Context, userID uuid.UUID, opts core_types.FilterOptions) (*core_types.PaginationResult[entity.File], error)

	// GetByIDs retrieves multiple files by their IDs.
	GetByIDs(ctx context.Context, ids []uuid.UUID) ([]*entity.File, error)
}

// fileUseCaseImpl implements the FileUsecase interface.
type fileUseCaseImpl struct {
	*coreUsecase.BaseUseCaseImpl[entity.File]                           // Embed base implementation
	repo                                      repository.FileRepository // Specific repo type
	logger                                    coreLogger.Logger
	driveService                              *drive.Service
	driveFolderID                             string // Optional: Specific folder to upload files into
}

// NewFileUsecase creates a new instance of FileUsecase.
// It expects credentialsPath to point to the Google Cloud service account JSON key file or OAuth credentials.
// driveFolderID is optional; if provided, files will be uploaded to this specific Drive folder.
func NewFileUsecase(
	repo repository.FileRepository,
	logger coreLogger.Logger,
	credentialsPath string, // Path to credentials.json
	driveFolderID string, // Optional: Drive Folder ID to upload into
) (FileUsecase, error) {

	ctx := context.Background()

	// --- Temporary Service Account assumption for compilation ---
	// Replace this with proper OAuth or Service Account setup
	// driveService, err := drive.NewService(ctx, option.WithCredentialsFile(credentialsPath), option.WithScopes(drive.DriveScope))

	driveService, err := drive.NewService(ctx, option.WithCredentialsFile(credentialsPath), option.WithScopes(drive.DriveScope))
	if err != nil {
		logger.Error("Unable to retrieve Drive client", "error", err)
		return nil, fmt.Errorf("unable to retrieve Drive client: %v", err)
	}
	// --- End Temporary ---

	baseUseCase := coreUsecase.NewBaseUseCase(repo, logger)

	return &fileUseCaseImpl{
		BaseUseCaseImpl: baseUseCase, // Directly use the returned concrete type
		repo:            repo,
		logger:          logger,
		driveService:    driveService,
		driveFolderID:   driveFolderID,
	}, nil
}

// --- Implement Specific FileUsecase Methods ---

// UploadFile uploads the file content to Google Drive and saves metadata to the database.
func (uc *fileUseCaseImpl) UploadFile(ctx context.Context, fileName string, fileContent io.Reader, fileSize int64, mimeType string, uploaderID uuid.UUID) (*entity.File, error) {
	uc.logger.Info("Attempting to upload file to Google Drive", "fileName", fileName, "mimeType", mimeType, "size", fileSize)

	// --- Google Drive Upload Logic ---
	driveFile := &drive.File{
		Name:     fileName,
		MimeType: mimeType,
		// Set parent folder if specified
	}
	if uc.driveFolderID != "" {
		driveFile.Parents = []string{uc.driveFolderID}
	}

	// Create the file on Google Drive
	// TODO: Handle context cancellation during upload?
	createdDriveFile, err := uc.driveService.Files.Create(driveFile).Media(fileContent).Context(ctx).Do()
	if err != nil {
		uc.logger.Error("Failed to create file in Google Drive", "fileName", fileName, "error", err)
		// Convert Drive API error to a use case error?
		return nil, coreUsecase.NewUseCaseError(coreUsecase.ErrInternal, fmt.Sprintf("failed to upload file to cloud storage: %v", err))
	}
	uc.logger.Info("File uploaded successfully to Google Drive", "driveFileId", createdDriveFile.Id, "fileName", fileName)

	// --- Set Public Read Permission ---
	uc.logger.Info("Setting 'anyone with link can read' permission", "driveFileId", createdDriveFile.Id)
	perm := &drive.Permission{Type: "anyone", Role: "reader"}
	_, err = uc.driveService.Permissions.Create(createdDriveFile.Id, perm).Context(ctx).Do()
	if err != nil {
		// Log the error but continue, the file is uploaded but might not be publicly accessible via the link.
		uc.logger.Error("Failed to set public read permission on Drive file", "driveFileId", createdDriveFile.Id, "error", err)
		// Depending on requirements, you might choose to return an error here or attempt to delete the file.
	} else {
		uc.logger.Info("Successfully set public read permission", "driveFileId", createdDriveFile.Id)
	}

	size := fileSize // Default to provided size

	// --- Get Web Content Link (Direct Download Link) ---
	// Fetch the file again to get the webContentLink (direct download link).
	// Note: webContentLink requires the file to be accessible to the caller.
	fetchedDriveFile, err := uc.driveService.Files.Get(createdDriveFile.Id).Fields("id, webContentLink, size").Context(ctx).Do()
	if err != nil {
		uc.logger.Error("Failed to fetch Drive file details after creation", "driveFileId", createdDriveFile.Id, "error", err)
		// Proceed, but URL might be missing
	}
	downloadLink := ""
	if fetchedDriveFile != nil {
		downloadLink = fetchedDriveFile.WebContentLink
		size = fetchedDriveFile.Size
		if downloadLink == "" {
			uc.logger.Warn("Drive file webContentLink is empty, check file permissions or API configuration", "driveFileId", createdDriveFile.Id)
		}
	}

	// --- Create Database Entity ---
	dbFile := &entity.File{
		// BaseEntity will be populated by GORM BeforeCreate hook
		Name:              fileName,
		ServiceInternalID: createdDriveFile.Id, // Store Drive File ID
		Type:              mimeType,
		Size:              size,
		UploaderID:        uploaderID,
		URL:               downloadLink, // Store the direct download link
	}

	// Use the embedded BaseUseCaseImpl's Create method
	if err := uc.BaseUseCaseImpl.Create(ctx, dbFile); err != nil {
		uc.logger.Error("Failed to save file metadata to database after Drive upload", "driveFileId", createdDriveFile.Id, "fileName", fileName, "error", err)
		// CRITICAL: File uploaded to Drive but DB record failed.
		// Consider attempting to delete the Drive file here for consistency.
		// Or implement a background job for cleanup.
		// --- Cleanup attempt (optional) ---
		/*
			deleteErr := uc.driveService.Files.Delete(createdDriveFile.Id).Context(context.Background()).Do() // Use background context for cleanup
			if deleteErr != nil {
				uc.logger.Error("Failed to delete orphaned Drive file after DB save failure", "driveFileId", createdDriveFile.Id, "deleteError", deleteErr)
			} else {
				uc.logger.Info("Successfully deleted orphaned Drive file", "driveFileId", createdDriveFile.Id)
			}
		*/
		// Return the DB error
		return nil, err // ConvertRepositoryError(err) if needed
	}

	uc.logger.Info("File metadata saved to database", "fileId", dbFile.ID, "driveFileId", dbFile.ServiceInternalID)
	// dbFile now contains the ID generated by the database
	return dbFile, nil
}

// DeleteFile overrides the base Delete to also remove the file from Google Drive on hard delete.
func (uc *fileUseCaseImpl) DeleteFile(ctx context.Context, fileID uuid.UUID, hardDelete bool) error {
	uc.logger.Info("Attempting to delete file", "fileId", fileID, "hardDelete", hardDelete)

	// If hard deleting, we need to delete from Drive first (or after, with rollback?)
	if hardDelete {
		// 1. Get the file entity from DB to find the Drive ID
		fileEntity, err := uc.BaseUseCaseImpl.GetByID(ctx, fileID) // Use embedded GetByID
		if err != nil {
			// Handle not found error specifically
			if ucErr, ok := err.(*coreUsecase.UseCaseError); ok && ucErr.Type == coreUsecase.ErrNotFound {
				uc.logger.Warn("File not found in DB for hard delete", "fileId", fileID)
				return err // Return the not found error
			}
			uc.logger.Error("Failed to get file entity for hard delete", "fileId", fileID, "error", err)
			return err // Return DB error
		}

		// 2. Delete from Google Drive
		driveFileID := fileEntity.ServiceInternalID
		if driveFileID != "" {
			uc.logger.Info("Attempting to delete file from Google Drive", "driveFileId", driveFileID, "fileId", fileID)
			err = uc.driveService.Files.Delete(driveFileID).Context(ctx).Do()
			// Handle Drive deletion errors
			// If file not found in Drive, maybe just log and continue?
			if err != nil {
				// Check for googleapi.Error type for specific status codes (e.g., 404 Not Found)
				uc.logger.Error("Failed to delete file from Google Drive", "driveFileId", driveFileID, "fileId", fileID, "error", err)
				// Decide whether to proceed with DB deletion or return error.
				// Returning error might be safer to indicate inconsistency.
				return coreUsecase.NewUseCaseError(coreUsecase.ErrInternal, fmt.Sprintf("failed to delete file from cloud storage: %v", err))
			}
			uc.logger.Info("File deleted successfully from Google Drive", "driveFileId", driveFileID, "fileId", fileID)
		} else {
			uc.logger.Warn("File entity found in DB but has no ServiceInternalID (Drive ID), skipping Drive deletion.", "fileId", fileID)
		}
	}

	// 3. Delete from Database (soft or hard) using the embedded BaseUseCaseImpl's Delete
	// This will handle the "not found" case again if the record was already deleted between steps (unlikely but possible)
	err := uc.BaseUseCaseImpl.Delete(ctx, fileID, hardDelete)
	if err != nil {
		// If Drive deletion succeeded but DB deletion failed, we have an orphaned Drive deletion.
		// This state is complex to automatically reconcile. Log it prominently.
		if hardDelete {
			// Attempt to retrieve DriveFileID again if possible, or use the one from above if still in scope
			driveFileID := ""                                                                                                       // Re-fetch or use variable if available
			if fileEntity, getErr := uc.BaseUseCaseImpl.GetByID(context.Background(), fileID); getErr == nil && fileEntity != nil { // Use background context for logging
				driveFileID = fileEntity.ServiceInternalID
			}
			uc.logger.Error("CRITICAL: File deleted from Drive, but failed to delete from DB", "fileId", fileID, "driveFileId", driveFileID, "error", err)
		} else {
			uc.logger.Error("Failed to soft delete file from DB", "fileId", fileID, "error", err)
		}
		return err // Return DB error
	}

	uc.logger.Info("File deleted successfully from database", "fileId", fileID, "hardDelete", hardDelete)
	return nil
}

// ListUserFiles retrieves files uploaded by a specific user.
func (uc *fileUseCaseImpl) ListUserFiles(ctx context.Context, userID uuid.UUID, opts core_types.FilterOptions) (*core_types.PaginationResult[entity.File], error) {
	uc.logger.Info("Listing files for user", "user_id", userID)

	filter := map[string]interface{}{
		"uploader_id": userID, // Filter by the UploaderID field
	}

	// Use the embedded FindWithFilter from BaseUseCaseImpl
	// Ensure BaseRepository implementation correctly handles uuid.UUID type in filter maps.
	result, err := uc.FindWithFilter(ctx, filter, opts)
	if err != nil {
		uc.logger.Error("Failed to list files for user", "user_id", userID, "error", err)
		// Consider wrapping error
		return nil, err
	}

	uc.logger.Info("Successfully listed files for user", "user_id", userID, "count", len(result.Items))
	return result, nil
}

// GetByIDs retrieves multiple files by their IDs.
func (uc *fileUseCaseImpl) GetByIDs(ctx context.Context, ids []uuid.UUID) ([]*entity.File, error) {
	uc.logger.Info("Getting files by IDs", "id_count", len(ids))
	if len(ids) == 0 {
		return []*entity.File{}, nil // Return empty slice, not an error
	}

	// Call the repository method
	files, err := uc.repo.FindManyByIDs(ctx, ids)
	if err != nil {
		uc.logger.Error("Failed to get files by IDs from repository", "id_count", len(ids), "error", err)
		// Don't wrap repository errors here unless adding specific use case context
		return nil, err
	}

	uc.logger.Info("Successfully retrieved files by IDs", "found_count", len(files))
	return files, nil
}

// --- Override Base Methods if Cloud Interaction is Needed ---

// Delete wraps the specific DeleteFile method to ensure cloud cleanup logic is called.
// This overrides the embedded Delete method from BaseUseCaseImpl.
func (uc *fileUseCaseImpl) Delete(ctx context.Context, id uuid.UUID, hardDelete bool) error {
	return uc.DeleteFile(ctx, id, hardDelete)
}

// TODO: Implement GetClient function for OAuth flow based on Google Drive API examples
// This function handles token retrieval, storage, and refresh.
// func getClient(config *oauth2.Config) *http.Client { ... }

// --- Helper Function for Usecase Errors (if not already central) ---
// This is already defined in pkg/core/usecase/usecase.go
// func NewUseCaseError(errorType coreUsecase.UseCaseErrorType, message string) error {
// 	return &coreUsecase.UseCaseError{
// 		Type:    errorType,
// 		Message: message,
// 	}
// }
