const getStatus = (value, { min, max }) => {
  if (value < min) return { label: 'Needs water', tone: 'low' };
  if (value > max) return { label: 'Soil saturated', tone: 'high' };
  return { label: 'In optimal range', tone: 'safe' };
};

import { formatTimestampRaw } from '../utils/date.js';

function MoistureChart({
  moisture,
  temperature,
  humidity,
  timestamp,
  deviceId,
  safeRange
}) {
  const status = getStatus(moisture ?? 0, safeRange);
  const timestampParts = formatTimestampRaw(timestamp);

  return (
    <article className="card">
      <h2 style={{ textAlign: 'center', fontSize: '1.5rem' }}>Current Moisture</h2>
      <p className="moisture-value">
        {typeof moisture === 'number' ? `${moisture.toFixed(1)}%` : '—'}
      </p>
      <p className={`status-pill ${status.tone}`}>{status.label}</p>

      <div className="meta">
        <p>
          <strong>Device:</strong> {deviceId || '—'}
        </p>
        <p>
          <strong>Measured:</strong>{' '}
          {timestamp
            ? `${timestampParts.time} / ${timestampParts.date}`
            : '—'}
        </p>
        <p>
          <strong>Temperature:</strong>{' '}
          {typeof temperature === 'number' ? `${temperature.toFixed(1)} °C` : '—'}
        </p>
        <p>
          <strong>Humidity:</strong>{' '}
          {typeof humidity === 'number' ? `${humidity.toFixed(1)} %` : '—'}
        </p>
      </div>
    </article>
  );
}

export default MoistureChart;
