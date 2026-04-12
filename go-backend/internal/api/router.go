// Package api provides route definitions for the Go backend.
package api

import (
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"

	"github.com/coinsight/go-backend/internal/websocket"
)

// SetupRouter configures all routes and middleware on the Gin engine.
func SetupRouter(handler *Handler, wsHub *websocket.Hub) *gin.Engine {
	r := gin.Default()

	// CORS middleware — allow frontend access
	r.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"*"},
		AllowMethods:     []string{"GET", "POST", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept"},
		AllowCredentials: true,
	}))

	// API routes
	api := r.Group("/api")
	{
		api.GET("/historical", handler.GetHistorical)
		api.GET("/predict", handler.GetPredictions)
		api.GET("/predict/export", handler.ExportPredictions)
		api.POST("/retrain", handler.PostRetrain)
		api.GET("/health", handler.HealthCheck)
	}

	// WebSocket route
	r.GET("/ws/ticker", wsHub.HandleConnection)

	return r
}
