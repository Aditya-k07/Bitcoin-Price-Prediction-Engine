// Package coingecko provides an HTTP client for the CoinGecko API.
// Fetches OHLC candlestick data and simple price for live ticker.
package coingecko

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"

	"github.com/coinsight/go-backend/internal/models"
)

// Client wraps HTTP calls to the CoinGecko public API.
type Client struct {
	baseURL    string
	httpClient *http.Client
}

// NewClient creates a CoinGecko API client with the given base URL and timeout.
func NewClient(baseURL string, timeoutSec int) *Client {
	return &Client{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: time.Duration(timeoutSec) * time.Second,
		},
	}
}

// GetOHLC fetches OHLC candlestick data for Bitcoin.
//
// CoinGecko OHLC endpoint returns arrays of [timestamp, open, high, low, close].
// Supported days values: 1, 7, 14, 30, 90, 180, 365, max.
// Candle granularity is automatic:
//   - 1-2 days: 30-minute candles
//   - 3-30 days: 4-hour candles
//   - 31+ days: daily candles
func (c *Client) GetOHLC(days int) ([]models.CandleData, error) {
	url := fmt.Sprintf("%s/coins/bitcoin/ohlc?vs_currency=usd&days=%d", c.baseURL, days)

	log.Printf("[CoinGecko] Fetching OHLC data for %d days...", days)

	resp, err := c.httpClient.Get(url)
	if err != nil {
		return nil, fmt.Errorf("coingecko OHLC request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("coingecko returned status %d: %s", resp.StatusCode, string(body))
	}

	// CoinGecko returns: [[timestamp, open, high, low, close], ...]
	var rawData [][]float64
	if err := json.NewDecoder(resp.Body).Decode(&rawData); err != nil {
		return nil, fmt.Errorf("failed to parse OHLC response: %w", err)
	}

	candles := make([]models.CandleData, 0, len(rawData))
	for _, row := range rawData {
		if len(row) < 5 {
			continue // skip malformed rows
		}
		candles = append(candles, models.CandleData{
			Timestamp: int64(row[0]),
			Open:      row[1],
			High:      row[2],
			Low:       row[3],
			Close:     row[4],
		})
	}

	log.Printf("[CoinGecko] Received %d candles", len(candles))
	return candles, nil
}

// SimplePrice holds the response from CoinGecko simple/price endpoint.
type SimplePrice struct {
	Bitcoin struct {
		USD            float64 `json:"usd"`
		USDChange24h   float64 `json:"usd_24h_change"`
		LastUpdatedAt  int64   `json:"last_updated_at"`
	} `json:"bitcoin"`
}

// GetSimplePrice fetches the current BTC/USD price with 24h change.
// Used by the WebSocket ticker for live price updates.
func (c *Client) GetSimplePrice() (*models.TickerMessage, error) {
	url := fmt.Sprintf(
		"%s/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true&include_last_updated_at=true",
		c.baseURL,
	)

	resp, err := c.httpClient.Get(url)
	if err != nil {
		return nil, fmt.Errorf("coingecko simple price request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("coingecko returned status %d: %s", resp.StatusCode, string(body))
	}

	var priceData SimplePrice
	if err := json.NewDecoder(resp.Body).Decode(&priceData); err != nil {
		return nil, fmt.Errorf("failed to parse simple price response: %w", err)
	}

	return &models.TickerMessage{
		Price:     priceData.Bitcoin.USD,
		Currency:  "usd",
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		Change24h: priceData.Bitcoin.USDChange24h,
	}, nil
}
