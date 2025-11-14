import { useEffect, useState, useRef } from 'react';

function formatTimeAgo(date) {
  if (!date) return 'N/A';
  const now = new Date();
  const then = new Date(date);
  const seconds = Math.floor((now - then) / 1000);
  
  if (seconds < 60) return `${seconds} seconds ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} minute${minutes !== 1 ? 's' : ''} ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} hour${hours !== 1 ? 's' : ''} ago`;
  const days = Math.floor(hours / 24);
  return `${days} day${days !== 1 ? 's' : ''} ago`;
}

function PredictionCard({ prediction, lastUpdated, isRefreshing, hasChanged }) {
  const [isAnimating, setIsAnimating] = useState(false);
  const previousPredictionId = useRef(null);

  // Trigger animation when data changes
  useEffect(() => {
    if (prediction && prediction.predictionId !== previousPredictionId.current) {
      setIsAnimating(true);
      previousPredictionId.current = prediction.predictionId;
      const timer = setTimeout(() => setIsAnimating(false), 600);
      return () => clearTimeout(timer);
    }
  }, [prediction]);

  if (!prediction) {
    return (
      <div className="prediction-card">
        <div className="prediction-header">
          <h2>Prediction Status</h2>
        </div>
        <p className="no-data">No predictions available yet</p>
      </div>
    );
  }

  const isPumpOn = prediction.pumpStatus === 'ON';
  const pumpTimeMinutes = (prediction.pumpTimeSec / 60).toFixed(1);
  const pumpTimeHours = (prediction.pumpTimeSec / 3600).toFixed(2);

  return (
    <div className={`prediction-card ${isPumpOn ? 'pump-on' : 'pump-off'} ${isRefreshing ? 'refreshing' : ''}`}>
      <div className="prediction-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <h2>Irrigation Prediction</h2>
          <span className="live-indicator" title={isRefreshing ? 'Updating...' : 'Live'}>
            {isRefreshing ? '🔄' : '●'}
          </span>
        </div>
        <span className={`status-badge ${isPumpOn ? 'on' : 'off'} ${isAnimating || hasChanged ? 'data-updated' : ''}`}>
          {prediction.pumpStatus}
        </span>
      </div>

      {/* Prominent Water and Pump Time Display */}
      <div className="prediction-values">
        <div className="prediction-value-item">
          <p className="prediction-value-label">Water Required</p>
          <p className="prediction-value">
            {typeof prediction.waterMM === 'number' ? `${prediction.waterMM.toFixed(1)} mm` : '—'}
          </p>
        </div>
        <div className="prediction-value-item">
          <p className="prediction-value-label">Pump Time</p>
          <p className="prediction-value">
            {typeof prediction.pumpTimeSec === 'number' 
              ? prediction.pumpTimeSec >= 3600 
                ? `${pumpTimeHours} hrs`
                : `${pumpTimeMinutes} min`
              : '—'}
          </p>
        </div>
      </div>

      <div className={`prediction-details ${isAnimating || hasChanged ? 'data-updated' : ''}`}>
        <div className="detail-row">
          <span className="label">Status:</span>
          <span className={`value ${prediction.used ? 'used' : 'pending'}`}>
            {prediction.used ? 'Used' : isPumpOn ? 'Pending' : 'No irrigation needed'}
          </span>
        </div>

        <div className="detail-row">
          <span className="label">Device:</span>
          <span className="value">{prediction.deviceId || 'N/A'}</span>
        </div>

        {prediction.createdAt && (
          <div className="detail-row">
            <span className="label">Predicted:</span>
            <span className="value">
              {formatTimeAgo(prediction.createdAt)}
            </span>
          </div>
        )}

        {lastUpdated && (
          <div className="detail-row meta">
            <span className="label">Last updated:</span>
            <span className="value">
              {formatTimeAgo(lastUpdated)}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

export default PredictionCard;

