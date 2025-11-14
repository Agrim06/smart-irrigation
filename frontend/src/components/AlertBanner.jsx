import { useEffect, useState } from 'react';

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

function AlertBanner({ alert, onDismiss, onMarkRead }) {
  const [isVisible, setIsVisible] = useState(true);
  const [isAnimating, setIsAnimating] = useState(false);

  useEffect(() => {
    if (alert) {
      setIsVisible(true);
      setIsAnimating(true);
      const timer = setTimeout(() => setIsAnimating(false), 500);
      return () => clearTimeout(timer);
    }
  }, [alert]);

  if (!alert || !isVisible) {
    return null;
  }

  const isPumpOn = alert.type === 'PUMP_ON';
  const pumpTimeMinutes = alert.pumpTimeSec ? (alert.pumpTimeSec / 60).toFixed(1) : 0;

  const handleDismiss = () => {
    setIsVisible(false);
    if (onDismiss) {
      setTimeout(() => onDismiss(), 300);
    }
  };

  const handleMarkRead = () => {
    if (onMarkRead && !alert.read) {
      onMarkRead(alert._id);
    }
  };

  return (
    <div 
      className={`alert-banner ${isPumpOn ? 'alert-pump-on' : 'alert-pump-off'} ${isAnimating ? 'alert-enter' : ''} ${alert.read ? 'alert-read' : ''}`}
      onClick={handleMarkRead}
    >
      <div className="alert-content">
        <div className="alert-icon">
          {isPumpOn ? '🚰' : '✅'}
        </div>
        <div className="alert-text">
          <div className="alert-title">
            {isPumpOn ? 'Irrigation Required' : 'No Irrigation Needed'}
          </div>
          <div className="alert-message">
            {alert.message}
          </div>
          {isPumpOn && alert.waterMM && (
            <div className="alert-details">
              Water: {alert.waterMM.toFixed(1)} mm • Duration: {pumpTimeMinutes} min
            </div>
          )}
          <div className="alert-meta">
            Device: {alert.deviceId || 'N/A'} • {formatTimeAgo(alert.createdAt)}
          </div>
        </div>
      </div>
      <button 
        className="alert-dismiss" 
        onClick={(e) => {
          e.stopPropagation();
          handleDismiss();
        }}
        aria-label="Dismiss alert"
      >
        ×
      </button>
    </div>
  );
}

export default AlertBanner;

