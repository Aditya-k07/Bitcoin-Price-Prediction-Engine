/**
 * RetrainButton — Triggers model retrain flow: Frontend → Go → Python.
 * Shows loading spinner during retraining.
 */

export default function RetrainButton({ model, onRetrain, isLoading }) {
  return (
    <button
      className={`retrain-btn ${isLoading ? 'retrain-btn--loading' : ''}`}
      onClick={() => onRetrain(model)}
      disabled={isLoading}
      title={`Retrain the ${model} model with latest data`}
    >
      {isLoading ? (
        <>
          <span className="retrain-btn__spinner" />
          Retraining...
        </>
      ) : (
        <>
          🔄 Retrain Model
        </>
      )}
    </button>
  );
}
