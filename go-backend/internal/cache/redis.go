// Package cache provides a Redis-based caching layer for predictions.
// Predictions are cached with a configurable TTL (default 5 minutes)
// to reduce load on the ML service.
package cache

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/coinsight/go-backend/internal/models"
)

// RedisCache wraps go-redis for prediction caching.
type RedisCache struct {
	client *redis.Client
	ttl    time.Duration
}

// NewRedisCache creates a new Redis cache client.
func NewRedisCache(addr, password string, db, ttlSeconds int) *RedisCache {
	rdb := redis.NewClient(&redis.Options{
		Addr:     addr,
		Password: password,
		DB:       db,
	})

	return &RedisCache{
		client: rdb,
		ttl:    time.Duration(ttlSeconds) * time.Second,
	}
}

// Ping checks the Redis connection.
func (rc *RedisCache) Ping(ctx context.Context) error {
	return rc.client.Ping(ctx).Err()
}

// predictionKey builds the Redis key for a cached prediction.
// Format: "prediction:{model}:{days}"
func predictionKey(model string, days int) string {
	return fmt.Sprintf("prediction:%s:%d", model, days)
}

// GetPrediction retrieves a cached prediction response.
// Returns nil if the cache is empty or expired.
func (rc *RedisCache) GetPrediction(ctx context.Context, model string, days int) (*models.PredictionResponse, error) {
	key := predictionKey(model, days)

	data, err := rc.client.Get(ctx, key).Bytes()
	if err == redis.Nil {
		return nil, nil // Cache miss — not an error
	}
	if err != nil {
		return nil, fmt.Errorf("redis get failed: %w", err)
	}

	var prediction models.PredictionResponse
	if err := json.Unmarshal(data, &prediction); err != nil {
		return nil, fmt.Errorf("failed to unmarshal cached prediction: %w", err)
	}

	log.Printf("[Cache] HIT for %s", key)
	prediction.Cached = true
	return &prediction, nil
}

// SetPrediction stores a prediction response in the cache.
func (rc *RedisCache) SetPrediction(ctx context.Context, model string, days int, prediction *models.PredictionResponse) error {
	key := predictionKey(model, days)

	data, err := json.Marshal(prediction)
	if err != nil {
		return fmt.Errorf("failed to marshal prediction for cache: %w", err)
	}

	if err := rc.client.Set(ctx, key, data, rc.ttl).Err(); err != nil {
		return fmt.Errorf("redis set failed: %w", err)
	}

	log.Printf("[Cache] SET %s (TTL: %v)", key, rc.ttl)
	return nil
}

// InvalidateModel removes all cached predictions for a given model.
// Called after a retrain to ensure fresh predictions are served.
func (rc *RedisCache) InvalidateModel(ctx context.Context, model string) error {
	pattern := fmt.Sprintf("prediction:%s:*", model)
	iter := rc.client.Scan(ctx, 0, pattern, 0).Iterator()

	count := 0
	for iter.Next(ctx) {
		if err := rc.client.Del(ctx, iter.Val()).Err(); err != nil {
			log.Printf("[Cache] Warning: failed to delete key %s: %v", iter.Val(), err)
		}
		count++
	}

	if err := iter.Err(); err != nil {
		return fmt.Errorf("redis scan failed: %w", err)
	}

	log.Printf("[Cache] Invalidated %d keys for model '%s'", count, model)
	return nil
}

// Close gracefully closes the Redis connection.
func (rc *RedisCache) Close() error {
	return rc.client.Close()
}
