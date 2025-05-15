package gateway

import (
	"golang-microservices-boilerplate/pkg/core/logger"
	"golang-microservices-boilerplate/pkg/middleware"

	"github.com/gofiber/fiber/v2"
)

// setupAuthMiddleware configures and applies JWT authentication middleware selectively to API routes.
func setupAuthMiddleware(app *fiber.App, logger logger.Logger) {

	// useage:
	// app.Use("/api/v1/auth/refresh", middleware.AuthMiddleware())
	app.Use("/api/v1/notifications", middleware.AuthMiddleware())
	// app.Use("/api/v1/files", middleware.AuthMiddleware())
	app.Use("/api/v1/requests", middleware.AuthMiddleware())
	app.Use("/api/v1/messages", middleware.AuthMiddleware())
	app.Use("/api/v1/conversations", middleware.AuthMiddleware())
	// app.Use("/api/v1/articles", middleware.AuthMiddleware())

	logger.Info("Auth middleware configured for apis")
}
