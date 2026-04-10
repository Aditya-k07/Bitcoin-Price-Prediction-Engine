/**
 * LiveTicker — WebSocket-powered real-time BTC/USD price display.
 * Shows current price, 24h change, and connection status.
 * Price flashes green/red on updates.
 */

import { useState, useEffect, useRef } from 'react';
import wsService from '../services/websocket';

export default function LiveTicker({ currency = 'usd' }) {
  const [price, setPrice] = useState(null);
  const [change24h, setChange24h] = useState(null);
  const [status, setStatus] = useState('disconnected');
  const [flash, setFlash] = useState('');
  const prevPrice = useRef(null);

  useEffect(() => {
    // Connect to WebSocket
    wsService.connect();

    // Subscribe to price updates
    const unsubMsg = wsService.onMessage((data) => {
      // data is now { "usd": {...}, "inr": {...} }
      const currData = data[currency];
      if (!currData) return;

      // Determine flash direction
      if (prevPrice.current !== null) {
        if (currData.price > prevPrice.current) setFlash('up');
        else if (currData.price < prevPrice.current) setFlash('down');
      }
      prevPrice.current = currData.price;

      setPrice(currData.price);
      setChange24h(currData.change_24h);

      // Clear flash after animation
      setTimeout(() => setFlash(''), 600);
    });

    // Subscribe to connection status
    const unsubStatus = wsService.onStatus((s) => setStatus(s));

    return () => {
      unsubMsg();
      unsubStatus();
    };
  }, []);

  const formatPrice = (p) => {
    if (p == null) return '—';
    return new Intl.NumberFormat(currency === 'inr' ? 'en-IN' : 'en-US', {
      style: 'currency',
      currency: currency.toUpperCase(),
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(p);
  };

  const formatChange = (c) => {
    if (c == null) return '';
    const sign = c >= 0 ? '+' : '';
    return `${sign}${c.toFixed(2)}%`;
  };

  return (
    <div className="ticker">
      <div className="ticker__dot" />
      <span className="ticker__label">BTC/{currency.toUpperCase()}</span>
      <span className={`ticker__price ${flash ? `ticker__price--${flash}` : ''}`}>
        {formatPrice(price)}
      </span>
      {change24h !== null && (
        <span
          className={`ticker__change ${
            change24h >= 0 ? 'ticker__change--positive' : 'ticker__change--negative'
          }`}
        >
          {formatChange(change24h)}
        </span>
      )}
    </div>
  );
}
