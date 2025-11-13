// server.js
const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");

const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));

// --------------------
// Config
// --------------------
const MONGO_URI = process.env.MONGO_URI || "mongodb://127.0.0.1:27017/esp_input";
const PORT = process.env.PORT || 5000;

// --------------------
// MongoDB connect (single call)
// --------------------
mongoose
  .connect(MONGO_URI)
  .then(() => console.log("✅ MongoDB connected:", MONGO_URI))
  .catch((err) => {
    console.error("❌ MongoDB connection error (initial):", err.message || err);
    // do not exit — server can keep running for dev. If you want to exit, uncomment next line:
    // process.exit(1);
  });

// Optional: listen for connection events
mongoose.connection.on("connected", () => {
  console.log("Mongoose: connected");
});
mongoose.connection.on("error", (err) => {
  console.error("Mongoose: connection error:", err.message || err);
});
mongoose.connection.on("disconnected", () => {
  console.warn("Mongoose: disconnected");
});

// --------------------
// Helpers
// --------------------
function getDynamicModel(collectionName) {
  if (!collectionName) throw new Error("Collection name required");

  // sanitize: allow letters, numbers and underscores only
  const safeName = String(collectionName).replace(/[^a-zA-Z0-9_]/g, "");
  if (!safeName) throw new Error("Invalid collection name after sanitization");

  // reuse compiled model if exists
  if (mongoose.models[safeName]) {
    return mongoose.models[safeName];
  }

  const schema = new mongoose.Schema(
    {
      deviceId: String,
      temperature: Number,
      humidity: Number,
      moisture: Number,
      timestamp: { type: Date, default: Date.now },
    },
    { strict: false }
  );

  return mongoose.model(safeName, schema, safeName);
}

// --------------------
// Routes
// --------------------

// Health
app.get("/health", async (req, res) => {
  const mongoState = mongoose.connection.readyState; // 0 disconnected, 1 connected
  res.json({
    status: mongoState === 1 ? "ok" : "degraded",
    mongoState,
    db: mongoose.connection.name || null,
    timestamp: new Date().toISOString(),
  });
});

// POST /data
app.post("/data", async (req, res) => {
  try {
    const { deviceId, temperature, humidity, moisture, timestamp, collection } = req.body;

    if (!collection) {
      return res.status(400).json({ success: false, message: "Missing 'collection' in request body" });
    }

    const DynamicModel = getDynamicModel(collection);

    const doc = new DynamicModel({
      deviceId,
      temperature,
      humidity,
      moisture,
      timestamp: timestamp ? new Date(timestamp) : new Date(),
    });

    await doc.save();

    return res.status(201).json({ success: true, saved_in: collection, data: doc });
  } catch (err) {
    console.error("POST /data error:", err && err.message ? err.message : err);
    return res.status(500).json({ success: false, error: err.message || String(err) });
  }
});

// GET /latest?collection=<name>
app.get("/latest", async (req, res) => {
  try {
    const collection = req.query.collection;
    if (!collection) return res.status(400).json({ success: false, message: "Missing ?collection=" });

    const DynamicModel = getDynamicModel(collection);
    const latest = await DynamicModel.findOne().sort({ timestamp: -1 }).lean().exec();

    return res.json({ success: true, collection, data: latest || null });
  } catch (err) {
    console.error("GET /latest error:", err && err.message ? err.message : err);
    return res.status(500).json({ success: false, error: err.message || String(err) });
  }
});

// GET /collections
app.get("/collections", async (req, res) => {
  try {
    const db = mongoose.connection.db;
    const cols = await db.listCollections().toArray();
    const collectionNames = cols.map((c) => c.name);
    return res.json({ success: true, collections: collectionNames });
  } catch (err) {
    console.error("GET /collections error:", err && err.message ? err.message : err);
    return res.status(500).json({ success: false, error: err.message || String(err) });
  }
});

// --------------------
// Start server
// --------------------
app.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
});
