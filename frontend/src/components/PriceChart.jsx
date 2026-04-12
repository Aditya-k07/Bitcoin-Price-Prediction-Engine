/**
 * PriceChart — Combined view with candlestick for both historical and predicted prices.
 *
 * Uses ApexCharts with multiple series:
 * 1. Candlestick series for historical OHLC data
 * 2. Candlestick series for predicted OHLC data (dashed, different color)
 *
 * The chart seamlessly stitches historical data to prediction data
 * on a shared time axis.
 */

import { useMemo } from 'react';
import Chart from 'react-apexcharts';

export default function PriceChart({ historicalData, predictions, isLoading }) {
  // Transform data for ApexCharts
  const { historicalCandles, predictionCandles } = useMemo(() => {
    // Historical candlestick data: [{x: timestamp, y: [O, H, L, C]}]
    const historicalCandles = (historicalData || []).map((d) => {
      // Backend (simple_backend.py) sends 'date', but some older versions or components might use 'timestamp'
      const dateVal = d.date || d.timestamp;
      const x = dateVal instanceof Date ? dateVal : new Date(dateVal);
      const y = [
        parseFloat(d.open) || 0,
        parseFloat(d.high) || 0,
        parseFloat(d.low) || 0,
        parseFloat(d.close) || 0,
      ];
      return { x, y };
    });

    // Prediction candlestick data: [{x: date, y: [O, H, L, C]}]
    // Date format from ML service: "2026-04-06"
    const predictionCandles = (predictions || []).map((p) => {
      const x = new Date(p.date); // Parse "2026-04-06" as ISO date
      const y = [
        parseFloat(p.open) || 0,
        parseFloat(p.high) || 0,
        parseFloat(p.low) || 0,
        parseFloat(p.close) || 0,
      ];
      return { x, y };
    });

    if (historicalCandles.length > 0 && predictionCandles.length > 0) {
      console.log('[Chart] Historical:', historicalCandles.length, 'Predicted:', predictionCandles.length);
      console.log('[Chart] Sample historical:', historicalCandles[0]);
      console.log('[Chart] Sample prediction:', predictionCandles[0]);
    }

    return {
      historicalCandles,
      predictionCandles,
    };
  }, [historicalData, predictions]);

  const series = [
    {
      name: 'BTC/USD (Historical)',
      type: 'candlestick',
      data: historicalCandles,
    },
    {
      name: 'BTC/USD (Predicted)',
      type: 'line',
      data: predictionCandles.map(p => ({
        x: p.x,
        y: p.y[3], // close price
      })),
    },
  ];

  console.log('[PriceChart] Data:', {
    historical: historicalCandles.length,
    predicted: predictionCandles.length,
  });

  const options = {
    chart: {
      type: 'candlestick',
      height: 500,
      background: 'transparent',
      toolbar: {
        show: true,
        tools: {
          download: true,
          selection: true,
          zoom: true,
          zoomin: true,
          zoomout: true,
          pan: true,
          reset: true,
        },
      },
      zoom: {
        enabled: true,
      },
      animations: {
        enabled: true,
        easing: 'easeinout',
        speed: 500,
      },
    },
    theme: {
      mode: 'dark',
    },
    grid: {
      borderColor: 'rgba(255, 255, 255, 0.06)',
      strokeDashArray: 3,
      xaxis: { lines: { show: false } },
      yaxis: { lines: { show: true } },
    },
    xaxis: {
      type: 'datetime',
      labels: {
        style: {
          colors: '#64748b',
          fontFamily: 'Inter, sans-serif',
          fontSize: '11px',
        },
        datetimeFormatter: {
          year: 'yyyy',
          month: "MMM 'yy",
          day: 'dd MMM',
          hour: 'HH:mm',
        },
      },
      axisBorder: { show: false },
      axisTicks: { show: false },
    },
    yaxis: {
      tooltip: { enabled: true },
      labels: {
        style: {
          colors: '#64748b',
          fontFamily: 'Inter, sans-serif',
          fontSize: '11px',
        },
        formatter: (val) => {
          if (val >= 1000) return `$${(val / 1000).toFixed(1)}k`;
          return `$${val.toFixed(0)}`;
        },
      },
    },
    tooltip: {
      theme: 'dark',
      style: {
        fontFamily: 'Inter, sans-serif',
      },
      x: {
        format: 'dd MMM yyyy',
      },
    },
    plotOptions: {
      candlestick: {
        colors: {
          upward: '#10b981',
          downward: '#ef4444',
        },
        wick: {
          useFillColor: true,
        },
      },
    },
    stroke: {
      width: [1, 2.5],
      dashArray: [0, 0],
      curve: 'smooth',
      lineCap: 'round',
      lineJoin: 'round',
    },
    // Colors for candlestick and line series - cyan for predictions
    colors: ['#10b981', '#00d4ff'],
    markers: {
      size: [0, 5],
      shape: ['square', 'circle'],
      hover: {
        size: [0, 7],
      },
    },
    fill: {
      opacity: [1, 0],
    },
    legend: {
      show: true,
      position: 'top',
      horizontalAlign: 'left',
      labels: {
        colors: '#94a3b8',
      },
      fontFamily: 'Inter, sans-serif',
      fontSize: '12px',
      markers: {
        width: 10,
        height: 10,
        radius: 3,
      },
    },
    responsive: [
      {
        breakpoint: 768,
        options: {
          chart: { height: 350 },
        },
      },
    ],
  };

  if (isLoading && historicalCandles.length === 0) {
    return (
      <div className="loading-overlay">
        <div className="loading-overlay__spinner" />
        Loading chart data...
      </div>
    );
  }

  return (
    <div className="chart-wrapper">
      <Chart
        options={options}
        series={series}
        type="candlestick"
        height={500}
        width="100%"
      />
    </div>
  );
}
