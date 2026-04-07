/**
 * ModelSelector — Toggle between Conservative (Prophet) and Aggressive (XGBoost).
 * Visually indicates active model with a glowing button style.
 */

export default function ModelSelector({ activeModel, onModelChange, disabled }) {
  const models = [
    { id: 'xgboost', label: 'XGBoost', tag: 'Aggressive' },
    { id: 'prophet', label: 'Prophet', tag: 'Conservative' },
  ];

  return (
    <div className="model-selector">
      {models.map((m) => (
        <button
          key={m.id}
          className={`model-selector__btn ${
            activeModel === m.id ? 'model-selector__btn--active' : ''
          }`}
          onClick={() => onModelChange(m.id)}
          disabled={disabled}
        >
          {m.label}
          <span className="model-selector__tag">{m.tag}</span>
        </button>
      ))}
    </div>
  );
}
