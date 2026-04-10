/**
 * ModelSelector — Toggle between XGBoost (Accurate) and Ridge (Fast).
 * Visually indicates active model with a glowing button style.
 */

export default function ModelSelector({ activeModel, onModelChange, disabled }) {
  const models = [
    { id: 'xgboost', label: 'XGBoost', tag: 'Accurate' },
    { id: 'ridge', label: 'Ridge', tag: 'Fast' },
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
