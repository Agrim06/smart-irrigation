const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

// -------------------------------
// MongoDB
// -------------------------------
mongoose.connect('mongodb://localhost:27017/irrigation_db')
  .then(() => console.log("MongoDB Connected"))
  .catch(err => console.error("DB ERROR:", err));


// -------------------------------
// Sensor Data Schema
// -------------------------------
const sensorSchema = new mongoose.Schema(
  {
    deviceId: String,
    temperature: Number,
    humidity: Number,
    moisture: Number,
    timestamp: { type: Date, default: Date.now }
  },
  {
    collection: 'sensordatas'
  }
);
const SensorData = mongoose.model('SensorData', sensorSchema);


// -------------------------------
// Prediction Schema
// -------------------------------
const predictionSchema = new mongoose.Schema({
  deviceId: String,
  waterMM: Number,
  pumpTimeSec: Number,
  predictionId: String,
  used: { type: Boolean, default: false },
  createdAt: { type: Date, default: Date.now }
});
const Prediction = mongoose.model("Prediction", predictionSchema);


// -------------------------------
// Alert Schema
// -------------------------------
const alertSchema = new mongoose.Schema({
  deviceId: String,
  type: { type: String, enum: ['PUMP_ON', 'PUMP_OFF'], required: true },
  message: String,
  waterMM: Number,
  pumpTimeSec: Number,
  read: { type: Boolean, default: false },
  createdAt: { type: Date, default: Date.now }
});
const Alert = mongoose.model("Alert", alertSchema);


// =====================================================================
// 1️⃣ Save Sensor Data
// =====================================================================
app.post("/api/sensors/data", async (req, res) => {
  try {
    const entry = new SensorData({
      deviceId: req.body.deviceId,
      temperature: req.body.temperature,
      humidity: req.body.humidity,
      moisture: req.body.moisture,
      timestamp: req.body.timestamp ? new Date(req.body.timestamp) : new Date()
    });

    await entry.save();
    res.json({ success: true });
  } 
  catch (e) {
    res.status(500).json({ success: false, error: e.message });
  }
});


// =====================================================================
// 1b️⃣ Fetch Latest Sensor Reading
// =====================================================================
app.get("/api/sensors/latest", async (req, res) => {
  try {
    const { deviceId } = req.query;
    const filter = deviceId ? { deviceId } : {};

    const latest = await SensorData.findOne(filter).sort({ timestamp: -1 });

    if (!latest) {
      return res.json({
        success: false,
        error: "No sensor data available",
        data: null
      });
    }

    res.json({
      success: true,
      data: {
        deviceId: latest.deviceId,
        temperature: latest.temperature,
        humidity: latest.humidity,
        moisture: latest.moisture,
        timestamp: latest.timestamp
      }
    });
  }
  catch (e) {
    res.status(500).json({ success: false, error: e.message });
  }
});


// =====================================================================
// 1c️⃣ Fetch Sensor History
// =====================================================================
app.get("/api/sensors", async (req, res) => {
  try {
    const { deviceId, limit = 100 } = req.query;
    const filter = deviceId ? { deviceId } : {};
    const size = Math.min(Math.max(parseInt(limit, 10) || 100, 1), 500);

    const data = await SensorData.find(filter)
      .sort({ timestamp: -1 })
      .limit(size);

    res.json({ success: true, data });
  }
  catch (e) {
    res.status(500).json({ success: false, error: e.message });
  }
});


// =====================================================================
// 2️⃣ Add Prediction (from ML model or manual)
// =====================================================================
app.post("/api/prediction/:deviceId", async (req, res) => {
  try {
    const { deviceId } = req.params;
    const { waterMM, pumpTimeSec } = req.body;

    const prediction = new Prediction({
      deviceId,
      waterMM,
      pumpTimeSec,
      predictionId: new Date().toISOString(),
      used: false
    });

    await prediction.save();
    res.json({ success: true, prediction });
  }
  catch (e) {
    res.status(500).json({ success: false, error: e.message });
  }
});


// =====================================================================
// 3️⃣ ESP Fetches Latest *Unused* Prediction
// =====================================================================
app.get("/api/prediction/:deviceId", async (req, res) => {
  try {
    const { deviceId } = req.params;

    const latest = await Prediction.findOne({
      deviceId,
      used: false
    }).sort({ createdAt: -1 });

    if (!latest) {
      return res.json({ success: false, error: "No active prediction" });
    }

    res.json({
      success: true,
      waterMM: latest.waterMM,
      pumpTimeSec: latest.pumpTimeSec,
      predictionId: latest.predictionId
    });
  } 
  catch (e) {
    res.status(500).json({ success: false, error: e.message });
  }
});


// =====================================================================
// 4️⃣ Mark Prediction as Used (ESP calls after pump finishes)
// =====================================================================
app.post("/api/prediction/mark-used/:deviceId", async (req, res) => {
  try {
    const { deviceId } = req.params;

    const latest = await Prediction.findOne({
      deviceId,
      used: false
    }).sort({ createdAt: -1 });

    if (!latest)
      return res.json({ success: false, message: "No active prediction" });

    latest.used = true;
    await latest.save();

    res.json({ success: true });
  }
  catch (e) {
    res.status(500).json({ success: false, error: e.message });
  }
});


// =====================================================================
// 5️⃣ Get Latest Prediction (for frontend display)
// =====================================================================
app.get("/api/predictions/latest", async (req, res) => {
  try {
    const { deviceId } = req.query;
    const filter = deviceId ? { deviceId } : {};

    const latest = await Prediction.findOne(filter)
      .sort({ createdAt: -1 });

    if (!latest) {
      return res.json({
        success: false,
        error: "No predictions available",
        data: null
      });
    }

    res.json({
      success: true,
      data: {
        deviceId: latest.deviceId,
        waterMM: latest.waterMM,
        pumpTimeSec: latest.pumpTimeSec,
        predictionId: latest.predictionId,
        used: latest.used,
        createdAt: latest.createdAt,
        pumpStatus: latest.waterMM > 0 ? "ON" : "OFF"
      }
    });
  }
  catch (e) {
    res.status(500).json({ success: false, error: e.message });
  }
});


// =====================================================================
// 6️⃣ Get Prediction History
// =====================================================================
app.get("/api/predictions", async (req, res) => {
  try {
    const { deviceId, limit = 50, days } = req.query;
    const filter = deviceId ? { deviceId } : {};
    
    // Add date filter if days parameter is provided
    if (days) {
      const daysNum = parseInt(days, 10);
      if (!isNaN(daysNum) && daysNum > 0) {
        const cutoffDate = new Date();
        cutoffDate.setDate(cutoffDate.getDate() - daysNum);
        filter.createdAt = { $gte: cutoffDate };
      }
    }
    
    const size = Math.min(Math.max(parseInt(limit, 10) || 50, 1), 1000);

    const predictions = await Prediction.find(filter)
      .sort({ createdAt: -1 })
      .limit(size);

    res.json({
      success: true,
      data: predictions.map(p => ({
        deviceId: p.deviceId,
        waterMM: p.waterMM,
        pumpTimeSec: p.pumpTimeSec,
        predictionId: p.predictionId,
        used: p.used,
        createdAt: p.createdAt,
        pumpStatus: p.waterMM > 0 ? "ON" : "OFF"
      }))
    });
  }
  catch (e) {
    res.status(500).json({ success: false, error: e.message });
  }
});


// =====================================================================
// 7️⃣ Get Alerts (unread first, then by date)
// =====================================================================
app.get("/api/alerts", async (req, res) => {
  try {
    const { deviceId, limit = 50, unreadOnly = false } = req.query;
    const filter = {};
    
    if (deviceId) filter.deviceId = deviceId;
    if (unreadOnly === 'true') filter.read = false;
    
    const size = Math.min(Math.max(parseInt(limit, 10) || 50, 1), 200);

    const alerts = await Alert.find(filter)
      .sort({ read: 1, createdAt: -1 }) // Unread first, then by date
      .limit(size);

    res.json({
      success: true,
      data: alerts.map(a => ({
        _id: a._id,
        deviceId: a.deviceId,
        type: a.type,
        message: a.message,
        waterMM: a.waterMM,
        pumpTimeSec: a.pumpTimeSec,
        read: a.read,
        createdAt: a.createdAt
      }))
    });
  }
  catch (e) {
    res.status(500).json({ success: false, error: e.message });
  }
});


// =====================================================================
// 8️⃣ Get Latest Alert (unread first, then any latest)
// =====================================================================
app.get("/api/alerts/latest", async (req, res) => {
  try {
    const { deviceId, unreadOnly = 'false' } = req.query;
    const filter = {};
    if (deviceId) filter.deviceId = deviceId;
    
    // First try to get unread alert
    const unreadFilter = { ...filter, read: false };
    let latest = await Alert.findOne(unreadFilter)
      .sort({ createdAt: -1 });

    // If no unread alert and unreadOnly is false, get the latest alert (even if read)
    if (!latest && unreadOnly !== 'true') {
      latest = await Alert.findOne(filter)
        .sort({ createdAt: -1 });
    }

    if (!latest) {
      return res.json({
        success: false,
        error: "No alerts found",
        data: null
      });
    }

    res.json({
      success: true,
      data: {
        _id: latest._id,
        deviceId: latest.deviceId,
        type: latest.type,
        message: latest.message,
        waterMM: latest.waterMM,
        pumpTimeSec: latest.pumpTimeSec,
        read: latest.read,
        createdAt: latest.createdAt
      }
    });
  }
  catch (e) {
    res.status(500).json({ success: false, error: e.message });
  }
});


// =====================================================================
// 9️⃣ Mark Alert as Read
// =====================================================================
app.post("/api/alerts/:alertId/read", async (req, res) => {
  try {
    const { alertId } = req.params;
    const alert = await Alert.findById(alertId);
    
    if (!alert) {
      return res.status(404).json({ success: false, error: "Alert not found" });
    }

    alert.read = true;
    await alert.save();

    res.json({ success: true });
  }
  catch (e) {
    res.status(500).json({ success: false, error: e.message });
  }
});


// =====================================================================
// 🔟 Mark All Alerts as Read
// =====================================================================
app.post("/api/alerts/read-all", async (req, res) => {
  try {
    const { deviceId } = req.body;
    const filter = { read: false };
    if (deviceId) filter.deviceId = deviceId;

    await Alert.updateMany(filter, { read: true });

    res.json({ success: true });
  }
  catch (e) {
    res.status(500).json({ success: false, error: e.message });
  }
});


// =====================================================================
app.listen(5000, () => console.log("API running at http://localhost:5000"));
