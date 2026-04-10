// Package mlclient provides an HTTP client for communicating with the Python ML service.
// Implements timeout handling, error resilience, and a simple circuit breaker.
package mlclient

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/coinsight/go-backend/internal/models"
)

// CircuitState represents the state of the circuit breaker.
type CircuitState int

const (
	CircuitClosed   CircuitState = iota // Normal operation
	CircuitOpen                         // Requests blocked
	CircuitHalfOpen                     // Testing with a single request
)

// Client wraps HTTP communication with the ML service.
type Client struct {
	baseURL    string
	httpClient *http.Client

	// Circuit breaker state
	mu              sync.RWMutex
	state           CircuitState
	failures        int
	maxFailures     int
	cooldownPeriod  time.Duration
	lastFailureTime time.Time
}

// NewClient creates an ML service client with circuit breaker.
func NewClient(baseURL string, timeoutSec int) *Client {
	return &Client{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: time.Duration(timeoutSec) * time.Second,
		},
		state:          CircuitClosed,
		failures:       0,
		maxFailures:    3,
		cooldownPeriod: 30 * time.Second,
	}
}

// isAvailable checks if the circuit breaker allows requests.
func (c *Client) isAvailable() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	switch c.state {
	case CircuitClosed:
		return true
	case CircuitOpen:
		// Check if cooldown period has passed
		if time.Since(c.lastFailureTime) > c.cooldownPeriod {
			return true // Will transition to half-open
		}
		return false
	case CircuitHalfOpen:
		return true
	}
	return false
}

// recordSuccess resets the circuit breaker on a successful call.
func (c *Client) recordSuccess() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.failures = 0
	c.state = CircuitClosed
}

// recordFailure increments the failure counter and potentially opens the circuit.
func (c *Client) recordFailure() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.failures++
	c.lastFailureTime = time.Now()

	if c.failures >= c.maxFailures {
		c.state = CircuitOpen
		log.Printf("[MLClient] Circuit breaker OPEN after %d failures. Cooldown: %v",
			c.failures, c.cooldownPeriod)
	}
}

// GetPredictions fetches price predictions from the ML service.
func (c *Client) GetPredictions(model string, days int) (*models.PredictionResponse, error) {
	if !c.isAvailable() {
		return nil, fmt.Errorf("ML service unavailable (circuit breaker open)")
	}

	url := fmt.Sprintf("%s/predict?model=%s&days=%d", c.baseURL, model, days)
	log.Printf("[MLClient] Fetching predictions: model=%s, days=%d", model, days)

	resp, err := c.httpClient.Get(url)
	if err != nil {
		c.recordFailure()
		return nil, fmt.Errorf("ML service request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		c.recordFailure()
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ML service returned status %d: %s", resp.StatusCode, string(body))
	}

	var prediction models.PredictionResponse
	if err := json.NewDecoder(resp.Body).Decode(&prediction); err != nil {
		c.recordFailure()
		return nil, fmt.Errorf("failed to parse ML prediction response: %w", err)
	}

	c.recordSuccess()
	return &prediction, nil
}

// GetPredictionsWithData sends fresh OHLC data to the ML service for predictions.
func (c *Client) GetPredictionsWithData(model string, days int, candles []models.CandleData) (*models.PredictionResponse, error) {
	if !c.isAvailable() {
		return nil, fmt.Errorf("ML service unavailable (circuit breaker open)")
	}

	url := fmt.Sprintf("%s/predict_with_data", c.baseURL)
	log.Printf("[MLClient] Fetching predictions with data: model=%s, days=%d, candles=%d", model, days, len(candles))

	// Build request payload
	requestData := map[string]interface{}{
		"model": model,
		"days": days,
		"data": candles,
	}

	payload, err := json.Marshal(requestData)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request data: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, url, bytes.NewBuffer(payload))
	if err != nil {
		return nil, fmt.Errorf("failed to build prediction request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		c.recordFailure()
		return nil, fmt.Errorf("ML service prediction request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		c.recordFailure()
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ML service returned status %d: %s", resp.StatusCode, string(body))
	}

	var prediction models.PredictionResponse
	if err := json.NewDecoder(resp.Body).Decode(&prediction); err != nil {
		c.recordFailure()
		return nil, fmt.Errorf("failed to parse ML prediction response: %w", err)
	}

	c.recordSuccess()
	return &prediction, nil
}

// Retrain triggers model retraining on the ML service.
func (c *Client) Retrain(model string) (*models.RetrainResponse, error) {
	if !c.isAvailable() {
		return nil, fmt.Errorf("ML service unavailable (circuit breaker open)")
	}

	url := fmt.Sprintf("%s/retrain?model=%s", c.baseURL, model)
	log.Printf("[MLClient] Triggering retrain: model=%s", model)

	req, err := http.NewRequest(http.MethodPost, url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to build retrain request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		c.recordFailure()
		return nil, fmt.Errorf("ML service retrain request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		c.recordFailure()
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ML service retrain returned status %d: %s", resp.StatusCode, string(body))
	}

	var result models.RetrainResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		c.recordFailure()
		return nil, fmt.Errorf("failed to parse retrain response: %w", err)
	}

	c.recordSuccess()
	return &result, nil
}

// Health checks if the ML service is healthy.
func (c *Client) Health() (*models.HealthResponse, error) {
	url := fmt.Sprintf("%s/health", c.baseURL)

	resp, err := c.httpClient.Get(url)
	if err != nil {
		return nil, fmt.Errorf("ML service health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ML service health returned status %d", resp.StatusCode)
	}

	var health models.HealthResponse
	if err := json.NewDecoder(resp.Body).Decode(&health); err != nil {
		return nil, fmt.Errorf("failed to parse health response: %w", err)
	}

	return &health, nil
}
