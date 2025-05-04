package usecase

import (
	"context"
	"fmt"

	"github.com/google/uuid"

	// "context" // Add if specific methods need it, BaseUseCase methods have it

	core_logger "golang-microservices-boilerplate/pkg/core/logger"
	core_usecase "golang-microservices-boilerplate/pkg/core/usecase"

	// core_types "golang-microservices-boilerplate/pkg/core/types" // Add if needed for specific methods
	core_repo "golang-microservices-boilerplate/pkg/core/repository"
	"golang-microservices-boilerplate/services/user-service/internal/entity"
	"golang-microservices-boilerplate/services/user-service/internal/repository"
	// "github.com/google/uuid" // Add if needed for specific methods
)

// ArticleUsecase defines the business logic operations for articles.
type ArticleUsecase interface {
	core_usecase.BaseUseCase[entity.Article] // Embed base CRUD
	// Add specific methods if needed, e.g.:
	// PublishArticle(ctx context.Context, articleID uuid.UUID) error
	CreateArticleWithFiles(ctx context.Context, article *entity.Article, fileIDs []string) (*entity.Article, error)
}

// articleUseCaseImpl implements the ArticleUsecase interface.
type articleUseCaseImpl struct {
	*core_usecase.BaseUseCaseImpl[entity.Article]
	repo     repository.ArticleRepository
	fileRepo repository.FileRepository
	logger   core_logger.Logger
}

// NewArticleUsecase creates a new instance of ArticleUsecase.
func NewArticleUsecase(
	repo repository.ArticleRepository,
	fileRepo repository.FileRepository,
	logger core_logger.Logger,
) ArticleUsecase {
	baseUseCase := core_usecase.NewBaseUseCase(repo, logger)
	return &articleUseCaseImpl{
		BaseUseCaseImpl: baseUseCase,
		repo:            repo,
		fileRepo:        fileRepo,
		logger:          logger,
	}
}

func (uc *articleUseCaseImpl) CreateArticleWithFiles(ctx context.Context, article *entity.Article, fileIDs []string) (*entity.Article, error) {
	uc.logger.Info("Attempting to create article with files", "title", article.Title, "file_count", len(fileIDs))

	// Use the base repository's transaction capability
	err := uc.Repository.Transaction(ctx, func(txRepo core_repo.BaseRepository[entity.Article]) error {
		// Need access to the specific FileRepository within the transaction as well.
		// This might require adjusting the Transaction helper or passing the *gorm.DB down.
		// Simpler approach for now: Assume fileRepo can use the same DB context if transactions are managed globally,
		// or enhance the Transaction helper. Let's assume fileRepo works for simplicity here.

		// 1. Create the article within the transaction
		if err := txRepo.Create(ctx, article); err != nil {
			uc.logger.Error("Failed to create article in transaction", "error", err)
			return err // Rollback
		}
		uc.logger.Debug("Article created in transaction", "article_id", article.ID)

		// 2. Handle file associations if IDs are provided
		if len(fileIDs) > 0 {
			var filesToAssociate []*entity.File
			for _, idStr := range fileIDs {
				fileID, parseErr := uuid.Parse(idStr)
				if parseErr != nil {
					uc.logger.Warn("Invalid file ID format provided", "file_id_str", idStr)
					return core_usecase.NewUseCaseError(core_usecase.ErrInvalidInput, fmt.Sprintf("invalid file ID format: %s", idStr)) // Rollback
				}

				// Find the file using the FileRepository
				file, findErr := uc.fileRepo.FindByID(ctx, fileID) // Use the injected fileRepo
				if findErr != nil {
					uc.logger.Error("Failed to find file for association", "file_id", fileID, "error", findErr)
					// Check if it's a not found error specifically
					if findErr.Error() == "entity not found" { // Adjust error check as needed
						return core_usecase.NewUseCaseError(core_usecase.ErrInvalidInput, fmt.Sprintf("file with ID %s not found", fileID))
					}
					return findErr // Rollback on other errors
				}
				filesToAssociate = append(filesToAssociate, file)
			}

			// 3. Associate found files with the article
			// Access the underlying *gorm.DB from the transactional repo if needed
			gormTxRepo, ok := txRepo.(*core_repo.GormBaseRepository[entity.Article])
			if !ok {
				return fmt.Errorf("unexpected repository type in transaction") // Should not happen
			}
			txDB := gormTxRepo.DB // Get the *gorm.DB with the transaction

			uc.logger.Debug("Associating files with article", "article_id", article.ID, "file_count", len(filesToAssociate))
			if assocErr := txDB.Model(article).Association("Files").Append(filesToAssociate); assocErr != nil {
				uc.logger.Error("Failed to append file association", "article_id", article.ID, "error", assocErr)
				return fmt.Errorf("failed to associate files: %w", assocErr) // Rollback
			}
		}
		// If all steps succeed, the transaction commits automatically
		return nil
	})

	if err != nil {
		// Transaction failed and rolled back
		return nil, err
	}

	// Transaction succeeded, article should have ID.
	// Optionally: Reload the article with Files preloaded if needed downstream
	// uc.Repository.FindByID(ctx, article.ID) ... with preload option

	uc.logger.Info("Successfully created article with files", "article_id", article.ID)
	return article, nil
}
