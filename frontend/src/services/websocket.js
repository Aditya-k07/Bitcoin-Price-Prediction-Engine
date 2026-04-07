/**
 * WebSocket service — manages connection to the live price ticker.
 * Auto-reconnects on disconnection with exponential backoff.
 */

class WebSocketService {
  constructor() {
    this.ws = null;
    this.listeners = new Set();
    this.statusListeners = new Set();
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 10;
    this.baseDelay = 1000; // 1 second
    this.isConnecting = false;
  }

  /**
   * Connect to the WebSocket ticker endpoint.
   * In dev, Vite proxy forwards /ws to the Go backend.
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
        console.log('[WS] Disconnected:', event.code, event.reason);
        this.isConnecting = false;
        this._notifyStatus('disconnected');
        this._scheduleReconnect();
      };

      this.ws.onerror = (error) => {
        console.error('[WS] Error:', error);
        this.isConnecting = false;
      };
    } catch (err) {
      console.error('[WS] Connection failed:', err);
      this.isConnecting = false;
      this._scheduleReconnect();
    }
  }

  /**
   * Schedule a reconnection attempt with exponential backoff.
   */
  _scheduleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log('[WS] Max reconnect attempts reached');
      return;
    }

    const delay = this.baseDelay * Math.pow(2, this.reconnectAttempts);
    this.reconnectAttempts++;

    console.log(`[WS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    setTimeout(() => this.connect(), delay);
  }

  /**
   * Notify status listeners of connection state changes.
   */
  _notifyStatus(status) {
    this.statusListeners.forEach((cb) => cb(status));
  }

  /**
   * Subscribe to price ticker messages.
   * @param {Function} callback - Called with {price, currency, timestamp, change_24h}
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
    this.listeners.clear();
    this.statusListeners.clear();
  }
}

// Singleton instance
const wsService = new WebSocketService();
export default wsService;
