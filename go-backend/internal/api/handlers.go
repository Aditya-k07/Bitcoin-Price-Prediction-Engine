// Package api provides HTTP handlers for the Go backend.
package api

import (
	"log"
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"

	"github.com/coinsight/go-backend/internal/cache"
	"github.com/coinsight/go-backend/internal/coingecko"
	"github.com/coinsight/go-backend/internal/mlclient"
	"github.com/coinsight/go-backend/internal/models"
)

// Handler holds dependencies for HTTP handlers.
type Handler struct {
	CoinGecko *coingecko.Client
	MLClient  *mlclient.Client
	Cache     *cache.RedisCache
}

// NewHandler creates a handler with all dependencies injected.
func NewHandler(cg *coingecko.Client, ml *mlclient.Client, rc *cache.RedisCache) *Handler {
	return &Handler{
		CoinGecko: cg,
		MLClient:  ml,
		Cache:     rc,
	}
}

// GetHistorical serves OHLC candlestick data from CoinGecko.
//
// This endpoint has ZERO dependency on the ML service — it works
// even if the Python service is down (Logic Isolation).
//
// GET /api/historical?days=90
func (h *Handler) GetHistorical(c *gin.Context) {
	daysStr := c.DefaultQuery("days", "90")
	days, err := strconv.Atoi(daysStr)
	if err != nil || days < 1 || days > 365 {
		c.JSON(http.StatusBadRequest, models.ErrorResponse{
			Error:   "invalid_parameter",
			Message: "days must be between 1 and 365",
		})
		return
	}

	currency := c.DefaultQuery("currency", "usd")
	if currency != "usd" {
		c.JSON(http.StatusBadRequest, models.ErrorResponse{
			Error:   "invalid_parameter",
			Message: "currency must be 'usd'",
		})
		return
	}

	candles, err := h.CoinGecko.GetOHLC(days, currency)
	if err != nil {
		log.Printf("[Handler] CoinGecko OHLC error: %v", err)
		c.JSON(http.StatusServiceUnavailable, models.ErrorResponse{
			Error:   "upstream_error",
			Message: "Failed to fetch historical data from CoinGecko",
		})
		return
	}

	c.JSON(http.StatusOK, models.HistoricalResponse{
		Data:   candles,
		Source: "coingecko",
		Cached: false,
	})
}

// GetPredictions proxies prediction requests to the ML service with fresh CoinGecko data.
// Fetches 1 year of historical OHLC data from CoinGecko and sends it with the prediction request.
// Results are cached in Redis for 5 minutes.
//
// GET /api/predict?model=xgboost&days=30
func (h *Handler) GetPredictions(c *gin.Context) {
	model := c.DefaultQuery("model", "xgboost")
	if model != "xgboost" && model != "lstm_xgboost" {
		c.JSON(http.StatusBadRequest, models.ErrorResponse{
			Error:   "invalid_parameter",
			Message: "model must be 'xgboost' or 'lstm_xgboost'",
		})
		return
	}

	currency := c.DefaultQuery("currency", "usd")
	if currency != "usd" {
		c.JSON(http.StatusBadRequest, models.ErrorResponse{
			Error:   "invalid_parameter",
			Message: "currency must be 'usd'",
		})
		return
	}

	daysStr := c.DefaultQuery("days", "30")
	days, err := strconv.Atoi(daysStr)
	if err != nil || days < 1 || days > 365 {
		c.JSON(http.StatusBadRequest, models.ErrorResponse{
			Error:   "invalid_parameter",
			Message: "days must be between 1 and 365",
		})
		return
	}

	// Check Redis cache first
	if h.Cache != nil {
		cached, err := h.Cache.GetPrediction(c.Request.Context(), model, days, currency)
		if err != nil {
			log.Printf("[Handler] Cache read error (non-fatal): %v", err)
		}
		if cached != nil {
			c.JSON(http.StatusOK, cached)
			return
		}
	}

	// Fetch 1 year of historical data from CoinGecko (365 days)
	log.Printf("[Handler] Fetching 1 year of historical data from CoinGecko...")
	candles, err := h.CoinGecko.GetOHLC(365, currency)
	if err != nil {
		log.Printf("[Handler] CoinGecko OHLC error: %v", err)
		c.JSON(http.StatusServiceUnavailable, models.ErrorResponse{
			Error:   "upstream_error",
			Message: "Failed to fetch historical data from CoinGecko",
		})
		return
	}

	log.Printf("[Handler] Got %d candles from CoinGecko, sending to ML service...", len(candles))

	// Send fresh data to ML service for prediction
	prediction, err := h.MLClient.GetPredictionsWithData(model, days, candles)
	if err != nil {
		log.Printf("[Handler] ML prediction error: %v", err)
		c.JSON(http.StatusServiceUnavailable, models.ErrorResponse{
			Error:   "ml_service_error",
			Message: "Failed to get predictions. The ML service may be unavailable or retraining.",
		})
		return
	}

	// Store in cache
	if h.Cache != nil {
		if err := h.Cache.SetPrediction(c.Request.Context(), model, days, currency, prediction); err != nil {
			log.Printf("[Handler] Cache write error (non-fatal): %v", err)
		}
	}

	prediction.Cached = false
	c.JSON(http.StatusOK, prediction)
}

// PostRetrain triggers model retraining via the ML service.
// Invalidates the Redis cache for the retrained model on success.
//
// POST /api/retrain?model=xgboost
func (h *Handler) PostRetrain(c *gin.Context) {
	model := c.DefaultQuery("model", "xgboost")
	if model != "xgboost" && model != "lstm_xgboost" {
		c.JSON(http.StatusBadRequest, models.ErrorResponse{
			Error:   "invalid_parameter",
			Message: "model must be 'xgboost' or 'lstm_xgboost'",
		})
		return
	}

	result, err := h.MLClient.Retrain(model)
	if err != nil {
		log.Printf("[Handler] ML retrain error: %v", err)
		c.JSON(http.StatusServiceUnavailable, models.ErrorResponse{
			Error:   "ml_service_error",
			Message: "Failed to retrain model. The ML service may be unavailable.",
		})
		return
	}

	// Invalidate cache for this model
	if h.Cache != nil {
		if err := h.Cache.InvalidateModel(c.Request.Context(), model); err != nil {
			log.Printf("[Handler] Cache invalidation error (non-fatal): %v", err)
		}
	}

	c.JSON(http.StatusOK, result)
}

// HealthCheck returns the backend's health status plus ML service connectivity.
//
// GET /api/health
func (h *Handler) HealthCheck(c *gin.Context) {
	status := "healthy"

	// Check ML service connectivity
	mlHealth, mlErr := h.MLClient.Health()
	mlStatus := "unavailable"
	var modelsLoaded []string
	if mlErr == nil && mlHealth != nil {
		mlStatus = mlHealth.Status
		modelsLoaded = mlHealth.ModelsLoaded
	}

	// Check Redis connectivity
	redisStatus := "unavailable"
	if h.Cache != nil {
		if err := h.Cache.Ping(c.Request.Context()); err == nil {
			redisStatus = "connected"
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"status": status,
		"services": gin.H{
			"ml_service": gin.H{
				"status":        mlStatus,
				"models_loaded": modelsLoaded,
			},
			"redis": gin.H{
				"status": redisStatus,
			},
		},
	})
}
