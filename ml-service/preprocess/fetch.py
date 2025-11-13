#!/usr/bin/env python3

import os
import sys
from typing import List, Dict, Any

from pymongo import MongoClient
import pandas as pd

# -------------------------
# CONFIG — no CLI needed
# -------------------------
SRC_URI = "mongodb://127.0.0.1:27017"
SRC_DB = "esp_input"
SRC_COLLECTION = "test_data"

DST_URI = "mongodb://127.0.0.1:27017"
DST_DB = "irrigation"
DST_COLLECTION = "test_data"

# SAVE CSV TO YOUR EXACT PATH
CSV_OUTPUT = r"C:\Users\agrim\Downloads\smart-irrigation\ml-service\transferred_test_data.csv"

LIMIT = 0   # 0 = fetch all docs

# -------------------------
# Helpers
# -------------------------
def connect(uri: str, db_name: str):
    client = MongoClient(uri)
    db = client[db_name]
    client.admin.command("ping")
    return db

def fetch_docs(db, collection, limit):
    coll = db[collection]
    cursor = coll.find().sort("timestamp", 1)
    if limit > 0:
        cursor = cursor.limit(limit)
    return list(cursor)

def normalize_for_csv(docs):
    rows = []
    for d in docs:
        row = dict(d)
        if "_id" in row:
            row["_id"] = str(row["_id"])
        if "timestamp" in row:
            try:
                row["timestamp"] = pd.to_datetime(row["timestamp"]).isoformat()
            except:
                row["timestamp"] = str(row["timestamp"])
        rows.append(row)
    return rows

def save_csv(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)

def transfer_docs(source_docs, dst_db, dst_collection):
    inserted = skipped = errors = 0
    coll = dst_db[dst_collection]

    for d in source_docs:
        try:
            src_id = str(d["_id"])

            if coll.find_one({"source_id": src_id}):
                skipped += 1
                continue

            new_doc = dict(d)
            new_doc.pop("_id", None)
            new_doc["source_id"] = src_id

            coll.insert_one(new_doc)
            inserted += 1

        except Exception as e:
            print("Insert error:", e)
            errors += 1

    return inserted, skipped, errors

# -------------------------
# Main
# -------------------------
def main():
    print("Connecting to source DB...")
    try:
        src_db = connect(SRC_URI, SRC_DB)
    except Exception as e:
        print("❌ Failed to connect to source:", e)
        sys.exit(1)

    print("Connecting to destination DB...")
    try:
        dst_db = connect(DST_URI, DST_DB)
    except Exception as e:
        print("❌ Failed to connect to destination:", e)
        sys.exit(1)

    print(f"Fetching from {SRC_DB}.{SRC_COLLECTION}...")
    docs = fetch_docs(src_db, SRC_COLLECTION, LIMIT)
    print(f"Fetched {len(docs)} documents.")

    csv_ready = normalize_for_csv(docs)

    print(f"Saving CSV -> {CSV_OUTPUT}")
    save_csv(csv_ready, CSV_OUTPUT)

    print("Transferring to irrigation DB...")
    inserted, skipped, errors = transfer_docs(docs, dst_db, DST_COLLECTION)

    print("\n------ Summary ------")
    print("Inserted:", inserted)
    print("Skipped:", skipped)
    print("Errors:", errors)
    print("\n✅ DONE.")

if __name__ == "__main__":
    main()
