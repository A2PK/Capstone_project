package main

import (
	"fmt"
	"log"
	"time"

	"golang-microservices-boilerplate/pkg/core/database"
	coreGrpc "golang-microservices-boilerplate/pkg/core/grpc" // Renamed alias to avoid conflict
	"golang-microservices-boilerplate/pkg/core/logger"
	"golang-microservices-boilerplate/pkg/utils"
	"golang-microservices-boilerplate/services/user-service/internal/controller"
	"golang-microservices-boilerplate/services/user-service/internal/entity"
	"golang-microservices-boilerplate/services/user-service/internal/repository"
	"golang-microservices-boilerplate/services/user-service/internal/usecase"
	// pb "golang-microservices-boilerplate/proto/user-service" // Import generated proto package - Needed in SetupGrpcServer
)

// Define struct to hold all dependencies
type ServiceDependencies struct {
	Logger              logger.Logger
	UserUseCase         usecase.UserUsecase
	FileUseCase         usecase.FileUsecase
	ArticleUseCase      usecase.ArticleUsecase
	RequestUseCase      usecase.RequestUsecase
	ChatUseCase         usecase.ChatUsecase
	NotificationUseCase usecase.NotificationUsecase
	Mapper              controller.Mapper
}

// SetupDependencies initializes all repositories, use cases, and mappers.
func SetupDependencies() (*ServiceDependencies, error) {
	// Initialize logger
	logConfig := logger.LoadLogConfigFromEnv()
	logConfig.AppName = utils.GetEnv("SERVER_APP_NAME", "User Service")
	appLogger, err := logger.NewLogger(logConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize logger: %w", err)
	}
	appLogger.Info("Logger initialized")

	// Initialize database connection
	db, err := database.NewDatabaseConnection(database.DefaultDBConfig())
	if err != nil {
		appLogger.Error("Failed to connect to database", "error", err)
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}
	appLogger.Info("Database connection established")

	// Auto migrate all models
	appLogger.Info("Running database migrations...")
	err = db.MigrateModels(
		&entity.User{},
		&entity.File{},
		&entity.Article{},
		&entity.Request{},
		&entity.ChatMessage{},
		&entity.Notification{},
		// Add other entities if any
	)
	if err != nil {
		appLogger.Error("Failed to auto-migrate models", "error", err)
		return nil, fmt.Errorf("failed to auto-migrate models: %w", err)
	}
	appLogger.Info("Database migrations completed")

	// Initialize repositories
	userRepo := repository.NewUserRepository(db.DB)
	fileRepo := repository.NewFileRepository(db.DB)
	articleRepo := repository.NewArticleRepository(db.DB)
	requestRepo := repository.NewRequestRepository(db.DB)
	chatRepo := repository.NewChatRepository(db.DB)
	notificationRepo := repository.NewNotificationRepository(db.DB)
	appLogger.Info("Repositories initialized")

	// --- Initialize Use Cases ---

	// User UseCase
	accessTokenDuration := 7 * 24 * time.Hour
	refreshTokenDuration := 30 * 24 * time.Hour
	userUseCase := usecase.NewUserUseCase(userRepo, appLogger, &accessTokenDuration, &refreshTokenDuration)

	// File UseCase (Requires credentials path and optional folder ID from env/config)
	credentialsPath := utils.GetEnv("GOOGLE_APPLICATION_CREDENTIALS_PATH", "credentials.json") // Example env var
	driveFolderID := utils.GetEnv("GOOGLE_DRIVE_FOLDER_ID", "")                                // Example env var
	fileUseCase, err := usecase.NewFileUsecase(fileRepo, appLogger, credentialsPath, driveFolderID)
	if err != nil {
		appLogger.Error("Failed to initialize FileUseCase", "error", err)
		return nil, fmt.Errorf("failed to initialize FileUseCase: %w", err)
	}

	// Article UseCase
	articleUseCase := usecase.NewArticleUsecase(articleRepo, fileRepo, appLogger)

	// Request UseCase
	requestUseCase := usecase.NewRequestUsecase(requestRepo, fileRepo, appLogger)

	// Chat UseCase (Depends on UserRepo)
	chatUseCase := usecase.NewChatUsecase(chatRepo, userRepo, appLogger)

	// Notification UseCase
	notificationUseCase := usecase.NewNotificationUsecase(notificationRepo, appLogger) // Assuming simple constructor

	appLogger.Info("Use cases initialized")

	// Initialize the general mapper
	mapper := controller.NewMapper()
	appLogger.Info("Mapper initialized")

	deps := &ServiceDependencies{
		Logger:              appLogger,
		UserUseCase:         userUseCase,
		FileUseCase:         fileUseCase,
		ArticleUseCase:      articleUseCase,
		RequestUseCase:      requestUseCase,
		ChatUseCase:         chatUseCase,
		NotificationUseCase: notificationUseCase,
		Mapper:              mapper,
	}

	appLogger.Info("Dependency setup completed successfully")
	return deps, nil
}

// SetupGrpcServer creates the gRPC server and registers all service handlers.
func SetupGrpcServer(deps *ServiceDependencies) (*coreGrpc.BaseGrpcServer, error) {
	if deps == nil {
		return nil, fmt.Errorf("service dependencies cannot be nil")
	}
	appLogger := deps.Logger

	appLogger.Info("Setting up gRPC server...")
	// Initialize gRPC server with interceptors
	grpcServer := coreGrpc.NewBaseGrpcServer(appLogger)

	// Register all service implementations
	// The Register...ServiceServer functions now correctly accept the use case and the general mapper
	controller.RegisterUserServiceServer(grpcServer.Server(), deps.UserUseCase, deps.Mapper)
	controller.RegisterFileServiceServer(grpcServer.Server(), deps.FileUseCase, deps.Mapper)
	controller.RegisterArticleServiceServer(grpcServer.Server(), deps.ArticleUseCase, deps.Mapper)
	controller.RegisterRequestServiceServer(grpcServer.Server(), deps.RequestUseCase, deps.Mapper)
	controller.RegisterDirectMessageServiceServer(grpcServer.Server(), deps.ChatUseCase, deps.Mapper) // Ensure correct registration name from chat.pb.go
	controller.RegisterNotificationServiceServer(grpcServer.Server(), deps.NotificationUseCase, deps.Mapper)

	appLogger.Info("gRPC services registered successfully")

	log.Printf("gRPC server setup completed successfully") // Keep using standard log here? Or logger?
	return grpcServer, nil
}
