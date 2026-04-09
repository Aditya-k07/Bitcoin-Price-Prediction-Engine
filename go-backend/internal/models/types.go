// Package models defines shared data types used across the Go backend.
package models

// CandleData represents a single OHLC candlestick data point.
type CandleData struct {
	Timestamp int64   `json:"timestamp"`
	Open      float64 `json:"open"`
	High      float64 `json:"high"`
	Low       float64 `json:"low"`
	Close     float64 `json:"close"`
}

// HistoricalResponse is the API response for the /api/historical endpoint.
type HistoricalResponse struct {
	Data   []CandleData `json:"data"`
	Source string       `json:"source"`
	Cached bool         `json:"cached"`
}

// PredictionPoint represents a single price prediction as OHLC candlestick.
type PredictionPoint struct {
	Date  string  `json:"date"`
	Open  float64 `json:"open"`
	High  float64 `json:"high"`
	Low   float64 `json:"low"`
	Close float64 `json:"close"`
}

// PredictionResponse is the response from the ML service /predict endpoint.
type PredictionResponse struct {
	Model                string            `json:"model"`
	Predictions          []PredictionPoint `json:"predictions"`
	RMSE                 float64           `json:"rmse"`
	MAE                  *float64          `json:"mae,omitempty"`
	R2Score              *float64          `json:"r2_score,omitempty"`
	MAPE                 *float64          `json:"mape,omitempty"`
	F1Score              *float64          `json:"f1_score,omitempty"`
	Accuracy             *float64          `json:"accuracy,omitempty"`
	DirectionalAccuracy  *float64          `json:"directional_accuracy,omitempty"`
	TrainedAt            string            `json:"trained_at"`
	ArchitectureDetails  map[string]interface{} `json:"architecture_details,omitempty"`
	Cached               bool              `json:"cached"`
}

// RetrainResponse is the response from the ML service /retrain endpoint.
type RetrainResponse struct {
	Model                string            `json:"model"`
	RMSE                 float64           `json:"rmse"`
	MAE                  *float64          `json:"mae,omitempty"`
	R2Score              *float64          `json:"r2_score,omitempty"`
	MAPE                 *float64          `json:"mape,omitempty"`
	F1Score              *float64          `json:"f1_score,omitempty"`
	Accuracy             *float64          `json:"accuracy,omitempty"`
	DirectionalAccuracy  *float64          `json:"directional_accuracy,omitempty"`
	TrainedAt            string            `json:"trained_at"`
	Message              string            `json:"message"`
	ArchitectureDetails  map[string]interface{} `json:"architecture_details,omitempty"`
}

// HealthResponse from ML service.
type HealthResponse struct {
	Status       string   `json:"status"`
	ModelsLoaded []string `json:"models_loaded"`
}

// TickerMessage is sent over WebSocket with live price data.
type TickerMessage struct {
	Price     float64 `json:"price"`
	Currency  string  `json:"currency"`
	Timestamp string  `json:"timestamp"`
	Change24h float64 `json:"change_24h"`
}

// ErrorResponse is a standard API error response.
type ErrorResponse struct {
	Error   string `json:"error"`
	Message string `json:"message"`
}
