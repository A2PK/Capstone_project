package main

import (
	"golang-microservices-boilerplate/pkg/core/database"
	"golang-microservices-boilerplate/pkg/core/grpc"
	"golang-microservices-boilerplate/pkg/core/logger"
	"golang-microservices-boilerplate/pkg/utils"
	controller "golang-microservices-boilerplate/services/water-quality-service/internal/controller"
	entity "golang-microservices-boilerplate/services/water-quality-service/internal/entity"
	repository "golang-microservices-boilerplate/services/water-quality-service/internal/repository"
	usecase "golang-microservices-boilerplate/services/water-quality-service/internal/usecase"
)

// SetupServices initializes all the services needed by the application
func SetupServices() (*grpc.BaseGrpcServer, error) {
	// Initialize logger
	logConfig := logger.LoadLogConfigFromEnv()
	logConfig.AppName = utils.GetEnv("SERVER_APP_NAME", "Water Quality Service")
	appLogger, err := logger.NewLogger(logConfig)
	if err != nil {
		return nil, err
	}

	appLogger.Info("Setting up water quality service")

	// Initialize database connection
	db, err := database.NewDatabaseConnection(database.DefaultDBConfig())
	if err != nil {
		appLogger.Error("Failed to connect to database", "error", err)
		return nil, err
	}
	appLogger.Info("Connected to database")

	// Auto migrate models (Remove Indicator)
	if err := db.MigrateModels(&entity.Station{}, &entity.DataPoint{} /*&entity.Indicator{},*/, &entity.DataSourceSchema{}); err != nil {
		appLogger.Error("Failed to auto-migrate models", "error", err)
		return nil, err
	}

	// Initialize repositories (Remove IndicatorRepo)
	stationRepo := repository.NewGormStationRepository(db.DB)
	dataPointRepo := repository.NewGormDataPointRepository(db.DB)
	// indicatorRepo := repository.NewGormIndicatorRepository(db.DB) // Removed
	schemaRepo := repository.NewGormDataSourceSchemaRepository(db.DB)

	// Initialize use cases
	stationUseCase := usecase.NewStationUsecase(
		stationRepo,
		appLogger,
	)

	dataPointUseCase := usecase.NewDataPointUsecase(
		dataPointRepo,
		// indicatorRepo, // Removed
		appLogger,
	)

	// indicatorUseCase := usecase.NewIndicatorUsecase( // Removed
	// 	indicatorRepo,
	// 	appLogger,
	// )

	importUseCase := usecase.NewImportService(
		stationRepo,
		dataPointRepo,
		// indicatorRepo, // Removed
		schemaRepo,
		appLogger,
	)

	// Initialize DataSourceSchema usecase
	schemaUseCase := usecase.NewDataSourceSchemaUsecase(
		schemaRepo,
		appLogger,
	)

	// Initialize mapper
	mapper := controller.NewWaterQualityMapper()

	// Initialize gRPC server with interceptors
	grpcServer := grpc.NewBaseGrpcServer(appLogger)

	// Register gRPC service implementation (Remove indicatorUseCase)
	controller.RegisterWaterQualityServiceServer(
		grpcServer.Server(),
		stationUseCase,
		dataPointUseCase,
		// indicatorUseCase, // Removed
		importUseCase,
		schemaUseCase,
		mapper,
	)

	appLogger.Info("Water Quality service setup completed successfully")
	return grpcServer, nil
}
