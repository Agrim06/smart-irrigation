const API_BASE_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5000';

export async function fetchLatestSensorData(deviceId) {
  const params = deviceId ? `?deviceId=${encodeURIComponent(deviceId)}` : '';
  const response = await fetch(`${API_BASE_URL}/api/sensors/latest${params}`, {
    headers: { 'Content-Type': 'application/json' }
  });

  if (!response.ok) {
    throw new Error(`Backend error (${response.status})`);
  }

  const payload = await response.json();

  if (!payload.success) {
    if (payload.data === null) {
      return null;
    }
    throw new Error(payload.error || 'Unknown backend error');
  }

  return payload.data ?? null;
}

export async function fetchSensorHistory({ deviceId, limit = 100 } = {}) {
  const params = new URLSearchParams();
  if (deviceId) params.set('deviceId', deviceId);
  if (limit) params.set('limit', limit);

  const response = await fetch(`${API_BASE_URL}/api/sensors?${params.toString()}`, {
    headers: { 'Content-Type': 'application/json' }
  });

  if (!response.ok) {
    throw new Error(`Backend error (${response.status})`);
  }

  const payload = await response.json();

  if (!payload.success) {
    throw new Error(payload.error || 'Unknown backend error');
  }

  return payload.data ?? [];
}

export async function fetchLatestPrediction(deviceId) {
  const params = deviceId ? `?deviceId=${encodeURIComponent(deviceId)}` : '';
  const response = await fetch(`${API_BASE_URL}/api/predictions/latest${params}`, {
    headers: { 'Content-Type': 'application/json' }
  });

  if (!response.ok) {
    throw new Error(`Backend error (${response.status})`);
  }

  const payload = await response.json();

  if (!payload.success) {
    if (payload.data === null) {
      return null;
    }
    throw new Error(payload.error || 'Unknown backend error');
  }

  return payload.data ?? null;
}

export async function fetchPredictionHistory({ deviceId, limit = 50, days } = {}) {
  const params = new URLSearchParams();
  if (deviceId) params.set('deviceId', deviceId);
  if (limit) params.set('limit', limit);
  if (days) params.set('days', days);

  const response = await fetch(`${API_BASE_URL}/api/predictions?${params.toString()}`, {
    headers: { 'Content-Type': 'application/json' }
  });

  if (!response.ok) {
    throw new Error(`Backend error (${response.status})`);
  }

  const payload = await response.json();

  if (!payload.success) {
    throw new Error(payload.error || 'Unknown backend error');
  }

  return payload.data ?? [];
}

export async function fetchAlerts({ deviceId, limit = 50, unreadOnly = false } = {}) {
  const params = new URLSearchParams();
  if (deviceId) params.set('deviceId', deviceId);
  if (limit) params.set('limit', limit);
  if (unreadOnly) params.set('unreadOnly', 'true');

  const response = await fetch(`${API_BASE_URL}/api/alerts?${params.toString()}`, {
    headers: { 'Content-Type': 'application/json' }
  });

  if (!response.ok) {
    throw new Error(`Backend error (${response.status})`);
  }

  const payload = await response.json();

  if (!payload.success) {
    throw new Error(payload.error || 'Unknown backend error');
  }

  return payload.data ?? [];
}

export async function fetchLatestAlert(deviceId) {
  const params = deviceId ? `?deviceId=${encodeURIComponent(deviceId)}` : '';
  const response = await fetch(`${API_BASE_URL}/api/alerts/latest${params}`, {
    headers: { 'Content-Type': 'application/json' }
  });

  if (!response.ok) {
    throw new Error(`Backend error (${response.status})`);
  }

  const payload = await response.json();

  if (!payload.success) {
    if (payload.data === null) {
      return null;
    }
    throw new Error(payload.error || 'Unknown backend error');
  }

  return payload.data ?? null;
}

export async function markAlertAsRead(alertId) {
  const response = await fetch(`${API_BASE_URL}/api/alerts/${alertId}/read`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' }
  });

  if (!response.ok) {
    throw new Error(`Backend error (${response.status})`);
  }

  const payload = await response.json();
  return payload.success;
}

export async function markAllAlertsAsRead(deviceId) {
  const response = await fetch(`${API_BASE_URL}/api/alerts/read-all`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ deviceId })
  });

  if (!response.ok) {
    throw new Error(`Backend error (${response.status})`);
  }

  const payload = await response.json();
  return payload.success;
}