/**
 * WebSocket service — manages connection to the live price ticker.
 * Falls back to CoinGecko API if WebSocket unavailable.
 */

class WebSocketService {
  constructor() {
    this.ws = null;
    this.listeners = new Set();
    this.statusListeners = new Set();
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 3; // Lower for this approach
    this.baseDelay = 1000; // 1 second
    this.isConnecting = false;
    this.fallbackPollInterval = null;
    this.lastPrice = null;
  }

  /**
   * Try WebSocket first, fall back to CoinGecko polling if unavailable.
   */
  connect() {
    if (this.ws?.readyState === WebSocket.OPEN || this.isConnecting) return;

    this.isConnecting = true;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/ticker`;

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log('[WS] Connected to ticker');
        this.reconnectAttempts = 0;
        this.isConnecting = false;
        if (this.fallbackPollInterval) clearInterval(this.fallbackPollInterval);
        this._notifyStatus('connected');
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.listeners.forEach((cb) => cb(data));
        } catch (err) {
          console.error('[WS] Failed to parse message:', err);
        }
      };

      this.ws.onclose = (event) => {
        console.log('[WS] Disconnected, falling back to CoinGecko polling');
        this.isConnecting = false;
        this._notifyStatus('disconnected');
        this._startFallbackPolling();
      };

      this.ws.onerror = (error) => {
        console.error('[WS] Error:', error);
        this.isConnecting = false;
        this._startFallbackPolling();
      };
    } catch (err) {
      console.error('[WS] Connection failed:', err);
      this.isConnecting = false;
      this._startFallbackPolling();
    }
  }

  /**
   * Fallback: poll CoinGecko API for current BTC price.
   */
  _startFallbackPolling() {
    if (this.fallbackPollInterval) return; // Already polling

    console.log('[Fallback] Starting CoinGecko polling for live price');
    this._notifyStatus('connected'); // Show as connected (via API fallback)

    // Fetch immediately
    this._fetchPriceFromCoinGecko();

    // Then poll every 10 seconds
    this.fallbackPollInterval = setInterval(() => {
      this._fetchPriceFromCoinGecko();
    }, 10000);
  }

  /**
   * Fetch current BTC price from CoinGecko.
   */
  async _fetchPriceFromCoinGecko() {
    try {
      const response = await fetch(
        'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd,inr&include_24hr_change=true'
      );
      const data = await response.json();

      if (data.bitcoin) {
        const phoneFormat = {
          usd: {
            price: data.bitcoin.usd,
            change_24h: data.bitcoin.usd_24h_change,
          },
          inr: {
            price: data.bitcoin.inr,
            change_24h: data.bitcoin.inr_24h_change,
          },
        };

        // Notify listeners in the same format as WebSocket
        this.listeners.forEach((cb) => cb(phoneFormat));
        this.lastPrice = data.bitcoin.usd;
      }
    } catch (err) {
      console.error('[Fallback] Failed to fetch from CoinGecko:', err);
    }
  }

  /**
   * Notify all status listeners.
   */
  _notifyStatus(status) {
    this.statusListeners.forEach((cb) => cb(status));
  }

  /**
   * Subscribe to price ticker messages.
   * @param {Function} callback - Called with {usd: {price, change_24h}, inr: {price, change_24h}}
   * @returns {Function} Unsubscribe function
   */
  onMessage(callback) {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }

  /**
   * Subscribe to connection status changes.
   * @param {Function} callback - Called with 'connected' or 'disconnected'
   * @returns {Function} Unsubscribe function
   */
  onStatus(callback) {
    this.statusListeners.add(callback);
    return () => this.statusListeners.delete(callback);
  }

  /**
   * Disconnect and clean up.
   */
  disconnect() {
    this.maxReconnectAttempts = 0; // Prevent reconnection
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    if (this.fallbackPollInterval) {
      clearInterval(this.fallbackPollInterval);
      this.fallbackPollInterval = null;
    }
    this.listeners.clear();
    this.statusListeners.clear();
  }
}

// Singleton instance
const wsService = new WebSocketService();
export default wsService;
