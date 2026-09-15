# 🏪 Smart Store Intelligence

> AI-powered retail analytics using computer vision, customer tracking, and machine learning.

Smart Store Intelligence is a real-time retail analytics system that analyzes customer movement inside a store using a live camera feed.

The system detects and tracks customers, identifies their movement across store zones, calculates dwell time, generates customer events, reconstructs customer journeys, and provides business insights through an interactive Streamlit dashboard.

---

## 🎯 Problem Statement

Traditional retail stores have limited visibility into how customers behave inside the store.

This project aims to answer questions such as:

- How many customers entered the store?
- Which zones receive the most attention?
- How long do customers spend in each zone?
- Which zones do customers visit?
- How do customers move between zones?
- How many customers reach checkout?
- Can customer behavior be used to predict checkout/conversion?

---

## 🚀 Key Features

### 👁️ Computer Vision

- Real-time person detection using YOLO
- Multi-object tracking using ByteTrack
- Persistent customer tracking across video frames
- Bounding-box based customer localization

### 📍 Zone Intelligence

The store is divided into logical zones:

- Entrance
- Shelf A
- Checkout

The customer's **bottom-center / feet point** is used to determine the zone where the customer is physically standing.

```text
feet_point = ((x1 + x2) / 2, y2)