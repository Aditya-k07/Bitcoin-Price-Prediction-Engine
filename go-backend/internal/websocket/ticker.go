// Package websocket provides a WebSocket hub for broadcasting live BTC price updates.
// Uses the hub pattern: one goroutine manages all connections and broadcasts.
package websocket

import (
	"encoding/json"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	ws "github.com/gorilla/websocket"

	"github.com/coinsight/go-backend/internal/coingecko"
	"github.com/coinsight/go-backend/internal/models"
)

// upgrader configures the WebSocket upgrade from HTTP.
var upgrader = ws.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	// Allow all origins for development (tighten in production)
	CheckOrigin: func(r *http.Request) bool { return true },
}

// Hub manages WebSocket connections and broadcasts price updates.
type Hub struct {
	clients    map[*ws.Conn]bool
	mu         sync.RWMutex
	cgClient   *coingecko.Client
	interval   time.Duration
	stopCh     chan struct{}
	lastTicker map[string]*models.TickerMessage
}

// NewHub creates a new WebSocket hub.
func NewHub(cgClient *coingecko.Client, intervalSec int) *Hub {
	return &Hub{
		clients:  make(map[*ws.Conn]bool),
		cgClient: cgClient,
		interval: time.Duration(intervalSec) * time.Second,
		stopCh:   make(chan struct{}),
	}
}

// Start begins the background price polling and broadcasting loop.
func (h *Hub) Start() {
	go h.pollAndBroadcast()
	log.Printf("[WebSocket] Hub started, polling every %v", h.interval)
}

// Stop terminates the background polling loop.
func (h *Hub) Stop() {
	close(h.stopCh)

	h.mu.Lock()
	defer h.mu.Unlock()
	for conn := range h.clients {
		conn.Close()
		delete(h.clients, conn)
	}
	log.Println("[WebSocket] Hub stopped")
}

// HandleConnection upgrades an HTTP connection to WebSocket and registers it.
func (h *Hub) HandleConnection(c *gin.Context) {
	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Printf("[WebSocket] Upgrade failed: %v", err)
		return
	}

	h.mu.Lock()
	h.clients[conn] = true
	clientCount := len(h.clients)
	h.mu.Unlock()

	log.Printf("[WebSocket] Client connected (%d total)", clientCount)

	// Send the last known price immediately on connect
	h.mu.RLock()
	lastTicker := h.lastTicker
	h.mu.RUnlock()

	if lastTicker != nil {
		data, _ := json.Marshal(lastTicker)
		conn.WriteMessage(ws.TextMessage, data)
	}

	// Read loop (handles disconnection detection)
	go func() {
		defer func() {
			h.mu.Lock()
			delete(h.clients, conn)
			remaining := len(h.clients)
			h.mu.Unlock()
			conn.Close()
			log.Printf("[WebSocket] Client disconnected (%d remaining)", remaining)
		}()

		for {
			// We don't expect messages from clients, but we need to read
			// to detect disconnection
			if _, _, err := conn.ReadMessage(); err != nil {
				break
			}
		}
	}()
}

// pollAndBroadcast fetches the latest price from CoinGecko and sends it to all clients.
func (h *Hub) pollAndBroadcast() {
	ticker := time.NewTicker(h.interval)
	defer ticker.Stop()

	for {
		select {
		case <-h.stopCh:
			return
		case <-ticker.C:
			h.fetchAndBroadcast()
		}
	}
}

// fetchAndBroadcast performs a single fetch-and-broadcast cycle.
func (h *Hub) fetchAndBroadcast() {
	h.mu.RLock()
	clientCount := len(h.clients)
	h.mu.RUnlock()

	if clientCount == 0 {
		return // No clients connected, skip the API call
	}

	priceMsg, err := h.cgClient.GetSimplePrice()
	if err != nil {
		log.Printf("[WebSocket] Failed to fetch price: %v", err)
		return
	}

	// Update last known ticker
	h.mu.Lock()
	h.lastTicker = priceMsg
	h.mu.Unlock()

	data, err := json.Marshal(priceMsg)
	if err != nil {
		log.Printf("[WebSocket] Failed to marshal ticker: %v", err)
		return
	}

	// Broadcast to all connected clients
	h.mu.RLock()
	defer h.mu.RUnlock()

	for conn := range h.clients {
		if err := conn.WriteMessage(ws.TextMessage, data); err != nil {
			log.Printf("[WebSocket] Write failed, will disconnect: %v", err)
			conn.Close()
			go func(c *ws.Conn) {
				h.mu.Lock()
				delete(h.clients, c)
				h.mu.Unlock()
			}(conn)
		}
	}
}
