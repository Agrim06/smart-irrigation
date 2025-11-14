import { useEffect, useMemo, useRef } from 'react';
import { fetchLatestAlert, markAlertAsRead } from '../api/backend.js';

export function useAlerts({
  deviceId,
  pollInterval = 10000,
  enableNotifications = true
} = {}) {
  const previousAlertIdRef = useRef(null);
  const intervalRef = useRef(null);

  const activeDevice = useMemo(() => deviceId?.trim() || undefined, [deviceId]);

  // Request notification permission
  useEffect(() => {
    if (enableNotifications && 'Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
  }, [enableNotifications]);

  // Show browser notification
  const showNotification = (alertData) => {
    if (!enableNotifications || !('Notification' in window)) {
      console.log('[useAlerts] Notifications not available');
      return;
    }

    if (Notification.permission !== 'granted') {
      console.log('[useAlerts] Notification permission not granted:', Notification.permission);
      // Try requesting permission again
      if (Notification.permission === 'default') {
        Notification.requestPermission().then(permission => {
          if (permission === 'granted') {
            // Retry showing notification
            showNotification(alertData);
          }
        });
      }
      return;
    }

    const title = alertData.type === 'PUMP_ON' ? '🚰 Irrigation Required' : '✅ No Irrigation Needed';
    const options = {
      body: alertData.message,
      icon: '/favicon.ico',
      badge: '/favicon.ico',
      tag: `alert-${alertData._id}`,
      requireInteraction: false,
      silent: false
    };

    try {
      const notification = new Notification(title, options);
      console.log('[useAlerts] Notification shown:', title);
      
      // Auto-close after 5 seconds
      setTimeout(() => {
        notification.close();
      }, 5000);
    } catch (err) {
      console.error('[useAlerts] Failed to show notification:', err);
    }
  };

  const load = async () => {
    try {
      const latest = await fetchLatestAlert(activeDevice);
      
      if (latest) {
        const isNewAlert = latest._id !== previousAlertIdRef.current;
        
        if (isNewAlert) {
          console.log('[useAlerts] New alert found:', latest.type, latest.read ? '(read)' : '(unread)');
          previousAlertIdRef.current = latest._id;
          
          // Show notification for unread alerts
          if (!latest.read) {
            showNotification(latest);
            // Mark as read after showing notification (with small delay to ensure notification is shown)
            setTimeout(async () => {
              try {
                await markAlertAsRead(latest._id);
                console.log('[useAlerts] Alert marked as read');
              } catch (err) {
                console.warn('[useAlerts] Failed to mark alert as read:', err);
              }
            }, 1000);
          }
        }
      } else {
        // No alerts found - this is normal
        if (process.env.NODE_ENV === 'development') {
          console.log('[useAlerts] No alerts found');
        }
      }
    } catch (err) {
      // Log errors in development
      if (process.env.NODE_ENV === 'development') {
        console.log('[useAlerts] Error:', err.message || 'Unknown error');
      }
    }
  };

  useEffect(() => {
    // Reset when device changes
    previousAlertIdRef.current = null;
    
    // Initial load
    load();
    
    // Set up polling interval
    intervalRef.current = setInterval(load, pollInterval);
    
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeDevice, pollInterval]);

  // This hook doesn't return anything - it only handles notifications
  return {};
}

