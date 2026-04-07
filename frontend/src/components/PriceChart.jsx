/**
 * PriceChart — Combined view with candlestick, prediction line, and confidence bands.
 *
 * Uses ApexCharts with multiple series:
 * 1. Candlestick series for historical OHLC data
 * 2. Line series for predicted prices (dashed cyan)
 * 3. RangeArea series for 95% confidence interval bands
 *
 * The chart seamlessly stitches historical data to prediction data
 * on a shared time axis.
 */

import { useMemo } from 'react';
import Chart from 'react-apexcharts';

export default function PriceChart({ historicalData, predictions, isLoading }) {
  // Transform data for ApexCharts
  const { candlestickData, predictionLine, confidenceBands } = useMemo(() => {
    // Candlestick data: [{x: timestamp, y: [O, H, L, C]}]
    const candles = (historicalData || []).map((d) => ({
      x: new Date(d.timestamp),
      y: [
        parseFloat(d.open.toFixed(2)),
        parseFloat(d.high.toFixed(2)),
        parseFloat(d.low.toFixed(2)),
        parseFloat(d.close.toFixed(2)),
      ],
    }));

    // Prediction line: [{x: date, y: price}]
    const predLine = (predictions || []).map((p) => ({
      x: new Date(p.date).getTime(),
      y: parseFloat(p.price.toFixed(2)),
    }));

    // Add a connecting point from the last candle to smooth the transition
    if (candles.length > 0 && predLine.length > 0) {
      const lastCandle = candles[candles.length - 1];
      predLine.unshift({
        x: lastCandle.x,
        y: lastCandle.y[3], // Close price
      });
    }

    // Confidence bands: [{x: date, y: [lower, upper]}]
    const bands = (predictions || []).map((p) => ({
      x: new Date(p.date).getTime(),
      y: [parseFloat(p.lower.toFixed(2)), parseFloat(p.upper.toFixed(2))],
    }));

    // Connect bands to last candle too
    if (candles.length > 0 && bands.length > 0) {
      const lastCandle = candles[candles.length - 1];
      const closePrice = lastCandle.y[3];
      bands.unshift({
        x: lastCandle.x,
        y: [closePrice, closePrice],
      });
    }

    return {
      candlestickData: candles,
      predictionLine: predLine,
      confidenceBands: bands,
    };
  }, [historicalData, predictions]);

  const series = [
    {
      name: 'BTC/USD',
      type: 'candlestick',
      data: candlestickData,
    },
    {
      name: 'Predicted Price',
      type: 'line',
      data: predictionLine,
    },
    {
      name: 'Confidence Interval',
      type: 'rangeArea',
      data: confidenceBands,
    },
  ];

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
          upward: '#22c55e',
          downward: '#ef4444',
        },
        wick: {
          useFillColor: true,
        },
      },
    },
    stroke: {
      width: [1, 2.5, 0],
      curve: 'smooth',
      dashArray: [0, 6, 0],
    },
    colors: ['#64748b', '#06b6d4', 'rgba(6, 182, 212, 0.15)'],
    fill: {
      type: ['solid', 'solid', 'solid'],
      opacity: [1, 1, 0.3],
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

  if (isLoading && candlestickData.length === 0) {
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
      />
    </div>
  );
}
