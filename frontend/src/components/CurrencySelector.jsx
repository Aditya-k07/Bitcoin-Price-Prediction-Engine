import React from 'react';

export default function CurrencySelector({ currency, setCurrency }) {
  return (
    <div className="currency-selector">
      <div 
        className={`currency-pill ${currency === 'usd' ? 'active' : ''}`}
        onClick={() => setCurrency('usd')}
      >
        <span className="currency-symbol">$</span> USD
      </div>
    </div>
  );
}
