// CoinSight Go Backend — Entry Point
//
// This is the orchestrator service that ties together:
// - CoinGecko API (historical data)
// - ML Service (predictions)
// - Redis (caching)
// - WebSocket (live ticker)
//
// Usage:
//   go run cmd/server/main.go
//
// Environment variables (see config/config.go for all options):
//   PORT=8080
//   ML_SERVICE_URL=http://localhost:8000
//   REDIS_ADDR=localhost:6379

package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/coinsight/go-backend/config"
	"github.com/coinsight/go-backend/internal/api"
	"github.com/coinsight/go-backend/internal/cache"
	"github.com/coinsight/go-backend/internal/coingecko"
	"github.com/coinsight/go-backend/internal/mlclient"
	"github.com/coinsight/go-backend/internal/websocket"
)

func main() {
	// Load configuration
	cfg := config.Load()

	log.Println("🚀 CoinSight Go Backend starting...")
	log.Printf("   Port:           %s", cfg.Port)
	log.Printf("   ML Service:     %s", cfg.MLServiceURL)
	log.Printf("   Redis:          %s", cfg.RedisAddr)
	log.Printf("   Ticker Interval: %ds", cfg.TickerInterval)

	// Initialize CoinGecko client
	cgClient := coingecko.NewClient(cfg.CoinGeckoBaseURL, cfg.CoinGeckoTimeout)

	// Initialize ML service client (with circuit breaker)
	mlClient := mlclient.NewClient(cfg.MLServiceURL, cfg.MLServiceTimeout)

	// Initialize Redis cache
	redisCache := cache.NewRedisCache(cfg.RedisAddr, cfg.RedisPassword, cfg.RedisDB, cfg.CacheTTL)

	// Test Redis connection
	ctx := context.Background()
	if err := redisCache.Ping(ctx); err != nil {
		log.Printf("⚠️  Redis connection failed: %v (caching disabled)", err)
	} else {
		log.Println("✅ Redis connected")
	}

	// Initialize WebSocket hub
	wsHub := websocket.NewHub(cgClient, cfg.TickerInterval)
	wsHub.Start()

	// Initialize HTTP handlers
	handler := api.NewHandler(cgClient, mlClient, redisCache)

	// Setup router with all routes
	router := api.SetupRouter(handler, wsHub)

	// Create HTTP server with timeouts
	srv := &http.Server{
		Addr:         fmt.Sprintf(":%s", cfg.Port),
		Handler:      router,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 60 * time.Second, // Long timeout for ML predictions
		IdleTimeout:  120 * time.Second,
	}

	// Start server in a goroutine
	go func() {
		log.Printf("✅ Server listening on :%s", cfg.Port)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server error: %v", err)
		}
	}()

	// Graceful shutdown on SIGINT/SIGTERM
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("🛑 Shutting down server...")

	// Give active requests 10 seconds to complete
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	wsHub.Stop()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("Server forced to shutdown: %v", err)
	}

	if err := redisCache.Close(); err != nil {
		log.Printf("Redis close error: %v", err)
	}

	log.Println("👋 Server stopped")
}
