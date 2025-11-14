import { useState } from 'react';
import MoistureChart from '../components/MoistureChart.jsx';
import SensorHistoryTable from '../components/SensorHistoryTable.jsx';
import PredictionCard from '../components/PredictionCard.jsx';
import ThemeToggle from '../components/ThemeToggle.jsx';
import { useLiveSensorData } from '../hooks/useLiveSensorData.js';
import { useSensorHistory } from '../hooks/useSensorHistory.js';
import { useLivePredictions } from '../hooks/useLivePredictions.js';
import { useAlerts } from '../hooks/useAlerts.js';

const safeRange = { min: 20, max: 45 };

function Dashboard() {
  const [deviceId, setDeviceId] = useState('');
  const { data, error, loading, lastUpdated, refresh } = useLiveSensorData({
    deviceId,
    pollInterval: 10000
  });
  const {
    data: history,
    error: historyError,
    loading: historyLoading,
    refresh: refreshHistory
  } = useSensorHistory({
    deviceId,
    limit: 5,
    pollInterval: 15000
  });

  const {
    data: prediction,
    error: predictionError,
    loading: predictionLoading,
    lastUpdated: predictionLastUpdated,
    isRefreshing: predictionRefreshing,
    hasChanged: predictionChanged,
    refresh: refreshPrediction
  } = useLivePredictions({
    deviceId,
    pollInterval: 10000
  });

  // Use alerts hook only for push notifications (no UI display)
  useAlerts({
    deviceId,
    pollInterval: 10000,
    enableNotifications: true
  });

  const handleRefresh = () => {
    refresh();
    refreshHistory();
    refreshPrediction();
  };

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <div className="header-content">
          <div>
            <h1 className="brand-name">Terafarm</h1>
            <p className="brand-tagline">Soil Moisture Monitor</p>
          </div>
          <ThemeToggle />
        </div>
      </header>

      {/* <div className="controls">
        <input
          value={deviceId}
          placeholder="Filter by Device ID (optional)"
          onChange={(event) => setDeviceId(event.target.value)}
        />
        <button type="button" onClick={handleRefresh}>
          Refresh now
        </button>
      </div> */}

      {loading && <p className="loading">Looking for the latest moisture reading…</p>}
      {error && <p className="error-banner">{error}</p>}
      {historyLoading && <p className="loading">Loading sensor history…</p>}
      {historyError && <p className="error-banner">{historyError}</p>}
      {predictionLoading && <p className="loading">Loading predictions…</p>}
      {predictionError && <p className="error-banner">{predictionError}</p>}

      <div className="card-grid">
        {data && (
          <MoistureChart
            moisture={data.moisture}
            temperature={data.temperature}
            humidity={data.humidity}
            timestamp={data.timestamp}
            deviceId={data.deviceId}
            safeRange={safeRange}
          />
        )}
        <PredictionCard 
          prediction={prediction} 
          lastUpdated={predictionLastUpdated}
          isRefreshing={predictionRefreshing}
          hasChanged={predictionChanged}
        />
      </div>

      {!loading && !error && !data && (
        <p className="error-banner">
          No readings found yet. Make sure your devices are posting data to `sensordatas`.
        </p>
      )}

      <SensorHistoryTable rows={history} />
    </div>
  );
}

export default Dashboard;
