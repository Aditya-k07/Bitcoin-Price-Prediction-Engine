// Package config provides environment-based configuration for the Go backend.
package config

import (
	"os"
	"strconv"
)

// Config holds all configuration values for the application.
type Config struct {
	// Server settings
	Port string

	// ML Service settings
	MLServiceURL     string
	MLServiceTimeout int // seconds

	// Redis settings
	RedisAddr     string
	RedisPassword string
	RedisDB       int
	CacheTTL      int // seconds

	// CoinGecko settings
	CoinGeckoBaseURL string
	CoinGeckoTimeout int // seconds

	// WebSocket settings
	TickerInterval int // seconds
}

// Load reads configuration from environment variables with sensible defaults.
func Load() *Config {
	return &Config{
		Port: getEnv("PORT", "8080"),

		MLServiceURL:     getEnv("ML_SERVICE_URL", "http://localhost:8000"),
<<<<<<< HEAD
		// Training XGBoost (3 estimators) often exceeds 30s; align with frontend retrain timeout.
		MLServiceTimeout: getEnvInt("ML_SERVICE_TIMEOUT", 600),
=======
		MLServiceTimeout: getEnvInt("ML_SERVICE_TIMEOUT", 30),
>>>>>>> 3bba824c0d1d9f1b3d9d9f10848532f480acc103

		RedisAddr:     getEnv("REDIS_ADDR", "localhost:6379"),
		RedisPassword: getEnv("REDIS_PASSWORD", ""),
		RedisDB:       getEnvInt("REDIS_DB", 0),
		CacheTTL:      getEnvInt("CACHE_TTL", 300), // 5 minutes

		CoinGeckoBaseURL: getEnv("COINGECKO_BASE_URL", "https://api.coingecko.com/api/v3"),
		CoinGeckoTimeout: getEnvInt("COINGECKO_TIMEOUT", 10),

		TickerInterval: getEnvInt("TICKER_INTERVAL", 10),
	}
}

func getEnv(key, fallback string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return fallback
}

func getEnvInt(key string, fallback int) int {
	if val := os.Getenv(key); val != "" {
		if i, err := strconv.Atoi(val); err == nil {
			return i
		}
	}
	return fallback
}
