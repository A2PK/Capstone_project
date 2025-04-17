package main

import (
	"golang-microservices-boilerplate/pkg/core/database"
	"golang-microservices-boilerplate/pkg/core/grpc"
	"golang-microservices-boilerplate/pkg/core/logger"
	"golang-microservices-boilerplate/pkg/utils"

	// Update these imports to point to model-service internal packages
	modelpb "golang-microservices-boilerplate/proto/model-service" // Import generated protobuf code
	controller "golang-microservices-boilerplate/services/model-service/internal/controller"
	"golang-microservices-boilerplate/services/model-service/internal/entity"
	"golang-microservices-boilerplate/services/model-service/internal/repository"
	"golang-microservices-boilerplate/services/model-service/internal/usecase"
	// entity "golang-microservices-boilerplate/services/model-service/internal/entity"
	// repository "golang-microservices-boilerplate/services/model-service/internal/repository"
	// usecase "golang-microservices-boilerplate/services/model-service/internal/usecase"
)

// SetupServices initializes all the services needed by the application
func SetupServices() (*grpc.BaseGrpcServer, error) {
	// Initialize logger
	logConfig := logger.LoadLogConfigFromEnv()
	// Update AppName for Model Service
	logConfig.AppName = utils.GetEnv("SERVER_APP_NAME", "Model Service")
	appLogger, err := logger.NewLogger(logConfig)
	if err != nil {
		return nil, err
	}

	appLogger.Info("Setting up model service")

	// Initialize database connection
	dbConn, err := database.NewDatabaseConnection(database.DefaultDBConfig())
	if err != nil {
		appLogger.Error("Failed to connect to database", "error", err)
		return nil, err
	}
	appLogger.Info("Connected to database")

	// Auto migrate models
	// Add model service entities here
	if err := dbConn.MigrateModels(&entity.AIModel{}); err != nil {
		appLogger.Error("Failed to auto-migrate models", "error", err)
		return nil, err
	}
	appLogger.Info("Database models migrated")

	// Initialize repositories (Pass the gorm.DB instance from the connection)
	modelRepo := repository.NewGormModelRepository(dbConn.DB)
	appLogger.Info("Repositories initialized")

	// Initialize use cases
	// This constructor is defined in services/model-service/internal/usecase/model.go
	modelUseCase := usecase.NewModelUseCase(
		modelRepo,
		appLogger,
	)
	appLogger.Info("Use cases initialized")

	// Initialize mapper
	mapper := controller.NewModelMapper()
	appLogger.Info("Mappers initialized")

	// Initialize gRPC server with interceptors
	grpcServer := grpc.NewBaseGrpcServer(appLogger)

	// Register gRPC service implementation
	modelpb.RegisterModelServiceServer(
		grpcServer.Server(),
		controller.NewModelServiceServer(appLogger, mapper, modelUseCase), // Pass all dependencies
	)
	appLogger.Info("gRPC server registered")

	appLogger.Info("Model service setup completed successfully")
	return grpcServer, nil
}
