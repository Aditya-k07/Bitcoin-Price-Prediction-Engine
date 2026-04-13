// Package coingecko provides an HTTP client for the CoinGecko API.
// Fetches OHLC candlestick data and simple price for live ticker.
package coingecko

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/coinsight/go-backend/internal/models"
)

// Client wraps HTTP calls to the CoinGecko public API.
type Client struct {
	baseURL    string
	httpClient *http.Client
	mu         sync.RWMutex
	cache      map[string][]models.CandleData
}

// NewClient creates a CoinGecko API client with the given base URL and timeout.
func NewClient(baseURL string, timeoutSec int) *Client {
	return &Client{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: time.Duration(timeoutSec) * time.Second,
		},
		cache: make(map[string][]models.CandleData),
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
func (c *Client) GetOHLC(days string, currency string) ([]models.CandleData, error) {
	if currency == "" {
		currency = "usd"
	}
	if days == "" {
		days = "31" // Minimum for daily candles
	}

	actualDays := days
	var daysVal int
	fmt.Sscanf(days, "%d", &daysVal)

	// Use Binance for long-term history (365+ days or "max") because CoinGecko free tier is restricted.
	if days == "max" || daysVal >= 365 {
		log.Printf("[MarketData] Switching to Binance for long-term history (%s days)", days)
		return c.getOHLCFromBinance(days, currency)
	}

	if daysVal < 31 && days != "max" {
		actualDays = "31"
	}

	// Try cache first for CoinGecko
	if cached := c.getCachedCandles(actualDays, currency); len(cached) > 0 {
		log.Printf("[CoinGecko] Using cached candles for %s days", actualDays)
		return cached, nil
	}

	url := fmt.Sprintf("%s/coins/bitcoin/ohlc?vs_currency=%s&days=%s", c.baseURL, currency, actualDays)
	log.Printf("[CoinGecko] Fetching OHLC data for %s days in %s...", actualDays, currency)

	resp, err := c.httpClient.Get(url)
	if err != nil {
		return nil, fmt.Errorf("coingecko OHLC request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("coingecko returned status %d: %s", resp.StatusCode, string(body))
	}

	var rawData [][]float64
	if err := json.NewDecoder(resp.Body).Decode(&rawData); err != nil {
		return nil, fmt.Errorf("failed to parse OHLC response: %w", err)
	}

	candles := make([]models.CandleData, 0, len(rawData))
	for _, row := range rawData {
		if len(row) < 5 {
			continue
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
	c.setCachedCandles(actualDays, currency, candles)
	return candles, nil
}

func (c *Client) getOHLCFromBinance(days string, currency string) ([]models.CandleData, error) {
	symbol := "BTCUSDT"
	interval := "1d"
	limit := 1000

	var allCandles []models.CandleData
	// We'll fetch 3 chunks of 1000 daily candles (~8 years) to cover "All Time"
	// Binance has data for BTCUSDT since 2017.
	
	now := time.Now().UnixMilli()
	// Chunk 1: most recent
	allCandles, err := c.fetchBinanceChunk(symbol, interval, limit, 0)
	if err != nil {
		return nil, err
	}

	if days == "max" && len(allCandles) > 0 {
		// Chunk 2: before chunk 1
		startTime := allCandles[0].Timestamp - (int64(limit) * 24 * 60 * 60 * 1000)
		chunk2, err := c.fetchBinanceChunk(symbol, interval, limit, startTime)
		if err == nil && len(chunk2) > 0 {
			// Prepend chunk2 to allCandles
			allCandles = append(chunk2, allCandles...)
		}
		
		// Chunk 3: even earlier (approx 2017/2018)
		if len(chunk2) > 0 {
			startTime = chunk2[0].Timestamp - (int64(limit) * 24 * 60 * 60 * 1000)
			chunk3, err := c.fetchBinanceChunk(symbol, interval, limit, startTime)
			if err == nil && len(chunk3) > 0 {
				allCandles = append(chunk3, allCandles...)
			}
		}
	}

	// De-duplicate and sort just in case
	unique := make(map[int64]models.CandleData)
	for _, c := range allCandles {
		unique[c.Timestamp] = c
	}
	
	sorted := make([]models.CandleData, 0, len(unique))
	for _, c := range unique {
		sorted = append(sorted, c)
	}
	// Sort by timestamp
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i].Timestamp > sorted[j].Timestamp {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	log.Printf("[Binance] Total history compiled: %d candles", len(sorted))
	return sorted, nil
}

func (c *Client) fetchBinanceChunk(symbol, interval string, limit int, startTime int64) ([]models.CandleData, error) {
	url := fmt.Sprintf("https://api.binance.com/api/v3/klines?symbol=%s&interval=%s&limit=%d", symbol, interval, limit)
	if startTime > 0 {
		url = fmt.Sprintf("%s&startTime=%d", url, startTime)
	}

	resp, err := c.httpClient.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("binance status %d", resp.StatusCode)
	}

	var rawData [][]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&rawData); err != nil {
		return nil, err
	}

	candles := make([]models.CandleData, 0, len(rawData))
	for _, row := range rawData {
		if len(row) < 6 {
			continue
		}
		candles = append(candles, models.CandleData{
			Timestamp: int64(row[0].(float64)),
			Open:      parsePrice(row[1]),
			High:      parsePrice(row[2]),
			Low:       parsePrice(row[3]),
			Close:     parsePrice(row[4]),
		})
	}
	return candles, nil
}

func parsePrice(v interface{}) float64 {
	s, ok := v.(string)
	if !ok {
		return 0
	}
	var f float64
	fmt.Sscanf(s, "%f", &f)
	return f
}



func cacheKey(days string, currency string) string {
	return fmt.Sprintf("%s:%s", currency, days)
}

func (c *Client) getCachedCandles(days string, currency string) []models.CandleData {
	c.mu.RLock()
	defer c.mu.RUnlock()
	key := cacheKey(days, currency)
	if candles, ok := c.cache[key]; ok && len(candles) > 0 {
		out := make([]models.CandleData, len(candles))
		copy(out, candles)
		return out
	}
	
	// Optional: recursive fallback logic for numeric days could stay here, 
	// but let's keep it simple for now as string-based cache is more precise.
	return nil
}

func (c *Client) setCachedCandles(days string, currency string, candles []models.CandleData) {
	c.mu.Lock()
	defer c.mu.Unlock()
	key := cacheKey(days, currency)
	out := make([]models.CandleData, len(candles))
	copy(out, candles)
	c.cache[key] = out
}

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
