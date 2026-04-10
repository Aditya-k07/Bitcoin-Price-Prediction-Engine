// Package coingecko provides an HTTP client for the CoinGecko API.
// Fetches OHLC candlestick data and simple price for live ticker.
package coingecko

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
<<<<<<< HEAD
	"sync"
=======
>>>>>>> 3bba824c0d1d9f1b3d9d9f10848532f480acc103
	"time"

	"github.com/coinsight/go-backend/internal/models"
)

// Client wraps HTTP calls to the CoinGecko public API.
type Client struct {
	baseURL    string
	httpClient *http.Client
<<<<<<< HEAD
	mu         sync.RWMutex
	cache      map[string][]models.CandleData
=======
>>>>>>> 3bba824c0d1d9f1b3d9d9f10848532f480acc103
}

// NewClient creates a CoinGecko API client with the given base URL and timeout.
func NewClient(baseURL string, timeoutSec int) *Client {
	return &Client{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: time.Duration(timeoutSec) * time.Second,
		},
<<<<<<< HEAD
		cache: make(map[string][]models.CandleData),
	}
}

// GetOHLC fetches OHLC-like daily candles for Bitcoin.
//
// Instead of relying on CoinGecko's /ohlc endpoint (which can be limited),
// this method uses /market_chart and aggregates price points into daily OHLC.
=======
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
>>>>>>> 3bba824c0d1d9f1b3d9d9f10848532f480acc103
func (c *Client) GetOHLC(days int, currency string) ([]models.CandleData, error) {
	if currency == "" {
		currency = "usd"
	}
<<<<<<< HEAD
	url := fmt.Sprintf("%s/coins/bitcoin/market_chart?vs_currency=%s&days=%d", c.baseURL, currency, days)

	log.Printf("[CoinGecko] Fetching market chart data for %d days in %s...", days, currency)

	resp, err := c.httpClient.Get(url)
	if err != nil {
		if cached := c.getCachedCandles(days, currency); len(cached) > 0 {
			log.Printf("[CoinGecko] Using cached candles after request error: %v", err)
			return cached, nil
		}
		return nil, fmt.Errorf("coingecko market_chart request failed: %w", err)
=======
	url := fmt.Sprintf("%s/coins/bitcoin/ohlc?vs_currency=%s&days=%d", c.baseURL, currency, days)

	log.Printf("[CoinGecko] Fetching OHLC data for %d days in %s...", days, currency)

	resp, err := c.httpClient.Get(url)
	if err != nil {
		return nil, fmt.Errorf("coingecko OHLC request failed: %w", err)
>>>>>>> 3bba824c0d1d9f1b3d9d9f10848532f480acc103
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
<<<<<<< HEAD
		if cached := c.getCachedCandles(days, currency); len(cached) > 0 {
			log.Printf("[CoinGecko] Using cached candles after status %d", resp.StatusCode)
			return cached, nil
		}
		return nil, fmt.Errorf("coingecko returned status %d: %s", resp.StatusCode, string(body))
	}

	// CoinGecko market_chart returns: {"prices": [[timestamp_ms, price], ...], ...}
	var payload struct {
		Prices [][]float64 `json:"prices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("failed to parse market_chart response: %w", err)
	}
	if len(payload.Prices) == 0 {
		return nil, fmt.Errorf("coingecko market_chart returned empty prices")
	}

	type dayCandle struct {
		tsMs int64
		open float64
		high float64
		low  float64
		close float64
	}

	dayOrder := make([]string, 0, len(payload.Prices))
	dayMap := make(map[string]*dayCandle, len(payload.Prices))

	for _, row := range payload.Prices {
		if len(row) < 2 {
			continue
		}
		tsMs := int64(row[0])
		price := row[1]
		day := time.UnixMilli(tsMs).UTC().Format("2006-01-02")

		if dc, ok := dayMap[day]; ok {
			if price > dc.high {
				dc.high = price
			}
			if price < dc.low {
				dc.low = price
			}
			dc.close = price
			continue
		}

		dayMap[day] = &dayCandle{
			tsMs: tsMs,
			open: price,
			high: price,
			low:  price,
			close: price,
		}
		dayOrder = append(dayOrder, day)
	}

	candles := make([]models.CandleData, 0, len(dayOrder))
	for _, day := range dayOrder {
		dc := dayMap[day]
		candles = append(candles, models.CandleData{
			Timestamp: dc.tsMs,
			Open:      dc.open,
			High:      dc.high,
			Low:       dc.low,
			Close:     dc.close,
		})
	}

	// Keep only the last requested number of daily candles.
	if len(candles) > days {
		candles = candles[len(candles)-days:]
	}
	c.setCachedCandles(days, currency, candles)

=======
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

>>>>>>> 3bba824c0d1d9f1b3d9d9f10848532f480acc103
	log.Printf("[CoinGecko] Received %d candles", len(candles))
	return candles, nil
}

<<<<<<< HEAD
func cacheKey(days int, currency string) string {
	return fmt.Sprintf("%s:%d", currency, days)
}

func (c *Client) getCachedCandles(days int, currency string) []models.CandleData {
	c.mu.RLock()
	defer c.mu.RUnlock()
	key := cacheKey(days, currency)
	if candles, ok := c.cache[key]; ok && len(candles) > 0 {
		out := make([]models.CandleData, len(candles))
		copy(out, candles)
		return out
	}
	// Fallback: use any larger cached range and trim to requested days.
	for _, candidate := range []int{365, 180, 90, 30, 14, 7} {
		if candidate < days {
			continue
		}
		k := cacheKey(candidate, currency)
		if candles, ok := c.cache[k]; ok && len(candles) >= days {
			start := len(candles) - days
			out := make([]models.CandleData, days)
			copy(out, candles[start:])
			return out
		}
	}
	return nil
}

func (c *Client) setCachedCandles(days int, currency string, candles []models.CandleData) {
	c.mu.Lock()
	defer c.mu.Unlock()
	key := cacheKey(days, currency)
	out := make([]models.CandleData, len(candles))
	copy(out, candles)
	c.cache[key] = out
}

=======
>>>>>>> 3bba824c0d1d9f1b3d9d9f10848532f480acc103
// SimplePrice holds the response from CoinGecko simple/price endpoint.
type SimplePrice struct {
	Bitcoin struct {
		USD            float64 `json:"usd"`
		USDChange24h   float64 `json:"usd_24h_change"`
		INR            float64 `json:"inr"`
		INRChange24h   float64 `json:"inr_24h_change"`
		LastUpdatedAt  int64   `json:"last_updated_at"`
	} `json:"bitcoin"`
}

// GetSimplePrice fetches the current BTC price in USD and INR with 24h change.
// Used by the WebSocket ticker for live price updates.
func (c *Client) GetSimplePrice() (map[string]*models.TickerMessage, error) {
	url := fmt.Sprintf(
		"%s/simple/price?ids=bitcoin&vs_currencies=usd,inr&include_24hr_change=true&include_last_updated_at=true",
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

	timestamp := time.Now().UTC().Format(time.RFC3339)

	return map[string]*models.TickerMessage{
		"usd": {
			Price:     priceData.Bitcoin.USD,
			Currency:  "usd",
			Timestamp: timestamp,
			Change24h: priceData.Bitcoin.USDChange24h,
		},
		"inr": {
			Price:     priceData.Bitcoin.INR,
			Currency:  "inr",
			Timestamp: timestamp,
			Change24h: priceData.Bitcoin.INRChange24h,
		},
	}, nil
}
