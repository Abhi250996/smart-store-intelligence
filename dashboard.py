"""
Mall Vision — Store Intelligence Dashboard
==========================================

ARCHITECTURE
------------
  LIVE CAMERA (stable, never recreated by analytics):
    WebRTC → SmartStoreVideoProcessor → YOLO → ByteTrack
    → ZoneManager → EventEngine → EventStore

  ANALYTICS (fragment, refreshes every PAGE_REFRESH seconds):
    Date/Camera selector → EventStore (date-filtered) → KPIs

WEBRTC LIFECYCLE FIX
--------------------
  streamlit-webrtc 0.77 checks _last_rendered_run_count on every
  script run.  If webrtc_streamer() is not called on a run the gap
  exceeds 1 and the worker is reset (play/pause/reconnect loop).

  Fix: webrtc_streamer() is called inside the @st.fragment body on
  EVERY fragment execution.  The stable key "smart-store-camera"
  ensures the frontend component is never recreated.

DATE-WISE ANALYTICS
-------------------
  Selecting a past date queries EventStore with date boundaries.
  Live camera is completely unaffected by date changes.
  "Today" shows live data (refreshed from DB + live processor state).
  Historical dates show read-only, date-filtered data.

MULTI-CAMERA ARCHITECTURE
--------------------------
  Events carry camera_id (default "camera_01").
  ZoneManager accepts per-camera zone configs.
  EventEngine stamps camera_id on every event.
  UI has a Camera filter (All Cameras / individual camera).
  Adding a second camera = add its ZoneManager config + new processor.

THREAD SAFETY
-------------
  SmartStoreVideoProcessor.__init__ runs in a background WebRTC
  thread with no Streamlit ScriptRunContext.  Module-level globals
  (_g_es, _g_cja, _g_cp) are set from the Streamlit thread just
  before webrtc_streamer() is called.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from typing import Optional

import av
import cv2
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

# ── Suppress aioice/aiortc cleanup noise ──────────────────────────────────
# When a WebRTC session ends (tab close, Ctrl+C, refresh), aioice fires
# Transaction.__retry() callbacks after the event loop is already torn down.
# This produces benign AttributeError spam:
#   'NoneType' object has no attribute 'sendto'
#   'NoneType' object has no attribute 'call_exception_handler'
# These do not affect functionality. Suppress them at the log level.
for _noisy_logger in ("aioice", "aioice.ice", "aioice.stun",
                      "aiortc", "aiortc.rtcdtlstransport",
                      "aiortc.rtcpeerconnection"):
    logging.getLogger(_noisy_logger).setLevel(logging.CRITICAL)

from src.customer_journey import CustomerJourneyAnalyzer
from src.detection import ObjectDetector
from src.event_engine import EventEngine
from src.event_store import EventStore
from src.ml.anomaly_detection import AnomalyDetector
from src.ml.customer_clustering import CustomerClustering
from src.ml.feature_engineering_v2 import FeatureEngineerV2
from src.ml.predict import ConversionPredictor
from src.store_analytics import StoreAnalytics
from src.zone_manager import ZoneManager

# ── constants ──────────────────────────────────────────────────────────────
STORE_NAME   = "Mall Vision"
PAGE_REFRESH = "3s"
CAMERA_01_ID = "camera_01"

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"{STORE_NAME} — Store Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

  :root{
    --purple:#7C3AED; --purple-d:#5B21B6; --teal:#0EA5A0; --teal-d:#0B7A76;
    --orange:#FB923C; --orange-d:#C2540C; --pink:#EC4899; --pink-d:#BE185D;
    --ink:#161320; --sub:#6b6478; --card:#ffffff; --line:#ece7f5;
    --bg:#f6f1fb;
  }

  #MainMenu,header,footer{visibility:hidden;height:0}
  div[data-testid="stToolbar"],div[data-testid="stDecoration"],
  div[data-testid="stStatusWidget"]{display:none}

  /* ── Fill the viewport, no page-level scrolling ─────────────────────── */
  html,body{height:100vh !important;overflow:hidden !important;
    margin:0;padding:0}
  [data-testid="stAppViewContainer"]{
    height:100vh !important;overflow:hidden !important;
    display:flex;flex-direction:column;
    background:
      radial-gradient(520px 380px at 100% 0%, rgba(124,58,237,.10), transparent 60%),
      radial-gradient(480px 360px at 0% 100%, rgba(14,165,160,.12), transparent 60%),
      var(--bg);
    font-family:'Inter',-apple-system,sans-serif;
  }
  [data-testid="stAppViewContainer"] > .main{
    overflow:hidden !important;flex:1;min-height:0;
    display:flex;flex-direction:column;
  }
  section.main > div.block-container{
    overflow:hidden !important;
  }
  [data-testid="stAppViewContainer"]::before{
    content:"";position:fixed;left:-120px;bottom:-140px;width:340px;height:340px;
    border-radius:50%;
    background:radial-gradient(circle at 30% 30%, rgba(14,165,160,.28), rgba(14,165,160,0) 70%);
    pointer-events:none;z-index:0;
  }
  [data-testid="stAppViewContainer"]::after{
    content:"";position:fixed;right:-140px;top:-140px;width:360px;height:360px;
    border-radius:50%;
    background:radial-gradient(circle at 70% 30%, rgba(124,58,237,.22), rgba(124,58,237,0) 70%);
    pointer-events:none;z-index:0;
  }
  .block-container{padding:.55rem 1rem !important;max-width:100% !important;
    height:100vh !important;overflow:hidden !important;
    display:flex;flex-direction:column;gap:.4rem;
    position:relative;z-index:1}
  .block-container > div{min-height:0}
  div[data-testid="stVerticalBlockBorderWrapper"],
  div[data-testid="stVerticalBlock"]{min-height:0}

  /* Hide every scrollbar on the page — content still scrolls if needed,
     it just never shows a visible/usable scroll track */
  *{scrollbar-width:none !important;-ms-overflow-style:none !important}
  *::-webkit-scrollbar{width:0 !important;height:0 !important;display:none !important}

  h1,h2,h3,.panel-title,.dash-title{font-family:'Poppins',sans-serif}

  /* Metric cards */
  [data-testid="stMetric"]{background:var(--card);border:1px solid var(--line);
    border-radius:16px;padding:.7rem .85rem .6rem;
    border-top:4px solid var(--purple);
    box-shadow:0 4px 14px rgba(76,29,149,.06);
    transition:transform .15s ease, box-shadow .15s ease}
  [data-testid="stMetric"]:hover{transform:translateY(-2px);
    box-shadow:0 8px 20px rgba(76,29,149,.10)}
  [data-testid="stMetricValue"]{font-size:1.28rem;font-weight:800;color:var(--ink);
    font-family:'Poppins',sans-serif}
  [data-testid="stMetricLabel"]{font-size:.62rem;color:var(--sub);
    font-weight:700;text-transform:uppercase;letter-spacing:.5px}

  /* cycle accent colors across each row of 4 metrics */
  div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(4n+1) [data-testid="stMetric"]{border-top-color:var(--purple)}
  div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(4n+2) [data-testid="stMetric"]{border-top-color:var(--teal)}
  div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(4n+3) [data-testid="stMetric"]{border-top-color:var(--orange)}
  div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(4n+4) [data-testid="stMetric"]{border-top-color:var(--pink)}

  /* Titles */
  .dash-title{font-size:1.55rem;font-weight:800;color:var(--ink);margin:0;
    display:flex;align-items:center;gap:.4rem}
  .dash-sub{font-size:.78rem;color:var(--sub);margin:.1rem 0 0;font-weight:500}
  .panel-title{font-size:.92rem;font-weight:700;color:var(--ink);
    margin-bottom:.5rem;padding-bottom:.4rem;
    border-bottom:1px solid var(--line);
    display:flex;align-items:center;gap:.4rem}

  /* Header hero card */
  .hero-card{background:linear-gradient(120deg,#ffffff 0%,#f8f5ff 100%);
    border:1px solid var(--line);border-radius:20px;
    padding:.9rem 1.2rem;box-shadow:0 6px 20px rgba(76,29,149,.07);
    margin-bottom:.6rem}

  /* Containers used via st.container(border=True) — stretch to fill,
     clip instead of scroll */
  div[data-testid="stVerticalBlockBorderWrapper"]{
    border-radius:18px !important;border:1px solid var(--line) !important;
    background:var(--card) !important;
    box-shadow:0 4px 18px rgba(76,29,149,.06) !important;
    flex:1;min-height:0;display:flex;flex-direction:column;overflow:hidden !important;
  }
  div[data-testid="stVerticalBlockBorderWrapper"] > div{min-height:0;overflow:hidden}

  /* Rows that host bordered panels grow to fill all leftover height;
     rows above (control bar) keep their natural compact height */
  div[data-testid="stHorizontalBlock"]:has(div[data-testid="stVerticalBlockBorderWrapper"]){
    flex:1;min-height:0}
  div[data-testid="column"]{display:flex !important;flex-direction:column !important;
    min-height:0}

  /* Badges */
  .badge{display:inline-flex;align-items:center;gap:6px;
    font-weight:800;font-size:.68rem;padding:4px 12px;
    border-radius:20px;text-transform:uppercase;letter-spacing:.3px}
  .badge-live{background:linear-gradient(90deg,#e6faf2,#e6faf2);color:#0f9d58;
    box-shadow:inset 0 0 0 1px rgba(15,157,88,.18)}
  .badge-hist{background:#f1ebfd;color:var(--purple-d);
    box-shadow:inset 0 0 0 1px rgba(124,58,237,.18)}
  .badge-off {background:#fdeceb;color:#c0362c;
    box-shadow:inset 0 0 0 1px rgba(192,54,44,.18)}
  .live-dot{width:7px;height:7px;border-radius:50%;
    background:currentColor;animation:pulse 1.6s infinite}
  .badge-off .live-dot,.badge-hist .live-dot{animation:none}
  @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(26,127,55,.4)}
    70%{box-shadow:0 0 0 6px rgba(26,127,55,0)}
    100%{box-shadow:0 0 0 0 rgba(26,127,55,0)}}

  /* Date banner */
  .date-banner{background:var(--card);border:1px solid var(--line);border-radius:14px;
    padding:.5rem .85rem;margin-bottom:.5rem;
    display:flex;align-items:center;gap:.6rem}
  .date-banner-hist{background:#f6f1fb;border-color:#e3d6fb}
  .date-label{font-size:.7rem;font-weight:700;color:var(--sub);
    text-transform:uppercase;letter-spacing:.4px}
  .date-value{font-size:.95rem;font-weight:800;color:var(--ink)}

  /* Zone cards — color-cycled left accent, like a timeline list */
  .zone-card{border:1px solid var(--line);border-radius:12px;
    padding:.55rem .7rem;margin-bottom:.4rem;background:#fdfcff;
    border-left:4px solid var(--purple);
    transition:box-shadow .15s ease}
  .zone-card:hover{box-shadow:0 4px 12px rgba(76,29,149,.08)}
  .zone-card:nth-of-type(3n+2){border-left-color:var(--teal)}
  .zone-card:nth-of-type(3n+3){border-left-color:var(--orange)}
  .zone-name{font-weight:800;font-size:.82rem;color:var(--ink)}
  .zone-stats{font-size:.7rem;color:var(--sub);margin-top:3px;font-weight:500}

  /* Insight cards */
  .ins{background:#fafafd;border-left:4px solid var(--teal-d);
    padding:.5rem .75rem;border-radius:10px;margin-bottom:.4rem;
    box-shadow:0 2px 8px rgba(0,0,0,.03)}
  .ins.w{border-left-color:var(--orange-d)}
  .ins.a{border-left-color:#D32F2F}
  .ins-t{font-size:.83rem;font-weight:800;color:var(--ink)}
  .ins-d{font-size:.71rem;color:#5b5568;margin-top:.15rem}

  /* Camera grid (multi-cam future) */
  .cam-tile{background:linear-gradient(150deg,#ffffff,#f8f6fd);
    border:1px solid var(--line);border-radius:14px;
    padding:.6rem .7rem;text-align:center;
    box-shadow:0 2px 8px rgba(76,29,149,.04)}
  .cam-label{font-size:.7rem;font-weight:800;color:#453e58;
    text-transform:uppercase;letter-spacing:.5px;margin-bottom:.25rem}

  /* Tabs */
  button[data-baseweb="tab"]{font-weight:700 !important;font-size:.78rem !important;
    color:var(--sub) !important}
  button[data-baseweb="tab"][aria-selected="true"]{color:var(--purple-d) !important}
  div[data-baseweb="tab-highlight"]{background-color:var(--purple) !important}

  /* Inputs (selects, date picker, text) → calendar + dropdown UI */
  div[data-baseweb="select"] > div, .stDateInput input, .stTextInput input{
    border-radius:12px !important;border:1px solid var(--line) !important;
    background:#fff !important;font-weight:600 !important;color:var(--ink) !important;
  }
  div[data-baseweb="select"]:focus-within > div, .stDateInput input:focus{
    border-color:var(--purple) !important;
    box-shadow:0 0 0 2px rgba(124,58,237,.15) !important;
  }
  .stDateInput svg{color:var(--purple) !important}

  /* Divider */
  hr{border-color:var(--line) !important;margin:.5rem 0 !important}

  ::-webkit-scrollbar{width:6px;height:6px}
  ::-webkit-scrollbar-thumb{background:#d6c9f2;border-radius:4px}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# SESSION BOOT  — once per browser session
# ══════════════════════════════════════════════════════════════════════════
def _boot() -> None:
    if st.session_state.get("_ready"):
        return
    es = EventStore()
    st.session_state.es   = es
    st.session_state.sa   = StoreAnalytics(es)
    st.session_state.cja  = CustomerJourneyAnalyzer(es)
    st.session_state.ad   = AnomalyDetector(es)
    st.session_state.cc   = CustomerClustering(es)
    try:
        st.session_state.cp = ConversionPredictor(es)
    except Exception as exc:
        st.session_state.cp = None
        print(f"[boot] ConversionPredictor: {exc}")
    st.session_state.proc_ref = None
    st.session_state._ready   = True

_boot()


# ══════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL GLOBALS — passed to background WebRTC thread
# ══════════════════════════════════════════════════════════════════════════
_g_es  : EventStore              | None = None
_g_cja : CustomerJourneyAnalyzer | None = None
_g_cp  : ConversionPredictor     | None = None

def _arm_globals() -> None:
    """Called from Streamlit thread immediately before webrtc_streamer()."""
    global _g_es, _g_cja, _g_cp
    _g_es  = st.session_state.es
    _g_cja = st.session_state.cja
    _g_cp  = st.session_state.get("cp")


# ══════════════════════════════════════════════════════════════════════════
# VIDEO PROCESSOR
# ══════════════════════════════════════════════════════════════════════════
class SmartStoreVideoProcessor(VideoProcessorBase):
    """
    Single-camera processor (camera_01).

    Multi-camera future: instantiate one processor per camera, each with
    its own camera_id, ZoneManager config, and EventEngine instance.
    """

    def __init__(self) -> None:
        self.predictor  = _g_cp
        self.event_store = _g_es  if _g_es  is not None else EventStore()
        self.analyzer    = _g_cja if _g_cja is not None else CustomerJourneyAnalyzer(self.event_store)

        # YOLO — once per processor.
        self.detector: ObjectDetector | None = None
        try:
            self.detector = ObjectDetector()
        except Exception as exc:
            print(f"[proc] YOLO: {exc}")

        # Camera-specific zone config (camera_01 default).
        self.zone_manager = ZoneManager(camera_id=CAMERA_01_ID)
        # EventEngine stamps CAMERA_01_ID on every event.
        self.event_engine = EventEngine(camera_id=CAMERA_01_ID)

        self.frame_count     : int   = 0
        self.last_frame_time : float = time.monotonic()
        self.current_zone_of_track: dict = {}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.last_frame_time = time.monotonic()
        self.frame_count    += 1

        if self.detector is None:
            out = self.zone_manager.draw_zones(img.copy())
            cv2.putText(out, "Detector loading...", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
            return av.VideoFrame.from_ndarray(
                cv2.resize(out, (640, 360), interpolation=cv2.INTER_AREA),
                format="bgr24")

        try:
            results = self.detector.detect(img)
        except Exception as exc:
            print(f"[recv] detect: {exc!r}")
            out = self.zone_manager.draw_zones(img.copy())
            return av.VideoFrame.from_ndarray(
                cv2.resize(out, (640, 360), interpolation=cv2.INTER_AREA),
                format="bgr24")

        if results is None or len(results) == 0:
            out = self.zone_manager.draw_zones(img.copy())
            return av.VideoFrame.from_ndarray(
                cv2.resize(out, (640, 360), interpolation=cv2.INTER_AREA),
                format="bgr24")

        result    = results[0]
        annotated = result.plot()
        boxes = track_ids = class_ids = []
        if result.boxes is not None:
            boxes     = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.int().cpu().tolist()
            track_ids = (
                result.boxes.id.int().cpu().tolist()
                if result.boxes.id is not None
                else [None] * len(boxes)
            )

        active_ids: list = []
        zone_snap : dict = {}

        for box, tid, cid in zip(boxes, track_ids, class_ids):
            if str(result.names.get(cid, cid)).lower() != "person":
                continue
            if tid is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            fx, fy = int((x1 + x2) / 2), int(y2)
            active_ids.append(tid)

            zone = self.zone_manager.get_zone(fx, fy)
            if zone:
                zone_snap[tid] = zone
                cv2.putText(annotated, f"ZONE:{zone}", (fx + 8, fy + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

            try:
                zone_name, dwell_sec, t_ev = self.event_engine.update_dwell_time(tid, zone)
                if t_ev:
                    self.event_store.save_event(t_ev)
                chk = self.event_engine.check_events(tid, zone_name, dwell_sec)
                if chk:
                    self.event_store.save_event(chk)

                if self.predictor and zone_name != "Checkout":
                    lf = self.event_engine.get_live_customer_features(tid)
                    if lf:
                        try:
                            pr = self.predictor.predict_live_features(lf)
                            if pr:
                                p  = pr["conversion_probability"]
                                lb = "HIGH" if p >= 80 else "MED" if p >= 50 else "LOW"
                                cv2.putText(annotated, f"Intent:{lb}",
                                            (fx + 8, fy + 42),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1)
                        except Exception:
                            pass

                if zone_name:
                    self.zone_manager.draw_person_zone(annotated, fx, fy, zone_name, tid)
                else:
                    cv2.circle(annotated, (fx, fy), 5, (0, 255, 255), -1)
            except Exception:
                pass

        try:
            gone, gone_evs = self.event_engine.cleanup_missing_tracks(active_ids)
            for ev in gone_evs:
                self.event_store.save_event(ev)
            for tid in gone:
                self.analyzer.get_customer_summary(tid)
                zone_snap.pop(tid, None)
        except Exception:
            pass

        self.current_zone_of_track = zone_snap
        annotated = self.zone_manager.draw_zones(annotated)
        return av.VideoFrame.from_ndarray(
            cv2.resize(annotated, (640, 360), interpolation=cv2.INTER_AREA),
            format="bgr24")


# ══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════
def fmt_dur(sec) -> str:
    try:    sec = int(sec)
    except Exception: return "N/A"
    if sec <= 0: return "N/A"
    m, s = divmod(sec, 60)
    return f"{m}m {s}s" if m else f"{s}s"

def _today_str() -> str:
    return date.today().isoformat()

def _proc() -> SmartStoreVideoProcessor | None:
    return st.session_state.get("proc_ref")

def _camera_state() -> str:
    p = _proc()
    if p is None: return "OFFLINE"
    try:
        if p.frame_count == 0: return "CONNECTING"
        elapsed = time.monotonic() - p.last_frame_time
        if elapsed <= 3.0:  return "LIVE"
        if elapsed <= 12.0: return "CONNECTING"
    except Exception: pass
    return "OFFLINE"

def _zone_occ() -> dict:
    p = _proc()
    if not p: return {}
    try:
        occ: dict = {}
        for _, z in (p.current_zone_of_track or {}).items():
            occ[z] = occ.get(z, 0) + 1
        return occ
    except Exception: return {}

def _inside():
    p = _proc()
    if not p: return "N/A"
    try:    return len(p.current_zone_of_track or {})
    except Exception: return "N/A"

# ── date-aware KPI helpers ─────────────────────────────────────────────────

def _avg_dwell_for(es: EventStore, date_str: Optional[str],
                   cam: Optional[str]) -> Optional[int]:
    try:
        events = es.get_events_by_date(date_str, cam, limit=2000)
        if not events: return None
        totals: dict = {}
        for e in events:
            t = e[1]; totals[t] = totals.get(t, 0) + (e[4] or 0)
        return int(sum(totals.values()) / len(totals)) if totals else None
    except Exception: return None

def _total_trans_for(es: EventStore, date_str: Optional[str],
                     cam: Optional[str]) -> Optional[int]:
    try:
        tr = es.get_zone_transitions_by_date(date_str, cam)
        return sum(tr.values()) if tr else None
    except Exception: return None

def _zone_breakdown_for(es: EventStore, sa: StoreAnalytics,
                        date_str: Optional[str], cam: Optional[str],
                        is_today: bool) -> dict:
    eng: dict = {}
    try: eng = sa.get_zone_engagement_rate_for(date_str, cam) or {}
    except Exception: pass
    za  = es.get_zone_analytics_by_date(date_str, cam)
    occ = _zone_occ() if is_today else {}
    out: dict = {}
    for z in sorted(za.keys()):
        s = za[z]
        out[z] = {
            "cur":   occ.get(z, None) if is_today else None,
            "vis":   s["visitors"] or None,
            "dwell": int(s["avg_dwell"]) if s["avg_dwell"] else None,
            "eng":   eng.get(z),
        }
    return out

def _insights_for(sa: StoreAnalytics, es: EventStore,
                  date_str: Optional[str], cam: Optional[str]) -> list:
    try:
        tot  = sa.get_total_customer_for(date_str, cam)
        if not tot: return []
        chk  = sa.get_checkout_customers_for(date_str, cam)
        rate = chk / tot * 100; ins = []
        eng  = sa.get_zone_engagement_rate_for(date_str, cam)
        if eng:
            ae = sum(eng.values()) / len(eng); tz = max(eng, key=eng.get)
            if ae >= 60:
                ins.append({"s":"pos","t":"🟢 Strong engagement",
                            "m":f"{tz} had highest engagement.",
                            "d":f"Avg: {ae:.0f}%"})
            elif ae >= 40:
                ins.append({"s":"pos","t":"🟢 Good activity",
                            "m":"Healthy engagement levels.",
                            "d":f"Avg: {ae:.0f}%"})
        if rate <= 20:
            ins.append({"s":"a","t":"🔴 Low conversion",
                        "m":f"Only {rate:.0f}% reached checkout.",
                        "d":"Attention needed."})
        elif rate < 50:
            ins.append({"s":"w","t":"🟠 Conversion opportunity",
                        "m":f"Checkout conversion: {rate:.0f}%.",
                        "d":"Review layout."})
        else:
            ins.append({"s":"pos","t":"🟢 Excellent conversion",
                        "m":f"Checkout conversion: {rate:.0f}%.",
                        "d":"Store performing well."})
        ad = _avg_dwell_for(es, date_str, cam)
        if ad:
            ins.append({"s":"pos","t":"🕒 Visit length",
                        "m":f"Avg visit: {fmt_dur(ad)}.",
                        "d":"Based on all tracked visits."})
        return ins
    except Exception: return []

def _high_intent_for(es: EventStore, date_str: Optional[str],
                     cam: Optional[str]) -> int:
    cp = st.session_state.get("cp")
    if not cp: return 0
    try:
        ids = es.get_customer_ids_by_date(date_str, cam)
        return sum(
            1 for cid in ids
            if (cp.predict_customer(cid) or {}).get("conversion_probability", 0) >= 70
        )
    except Exception: return 0


# ══════════════════════════════════════════════════════════════════════════
# STATIC HEADER  (rendered once — outside fragment, never recreated)
# ══════════════════════════════════════════════════════════════════════════
st.markdown("<div class='hero-card'>", unsafe_allow_html=True)
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown(
        f"<p class='dash-title'>🛍️ {STORE_NAME}</p>"
        f"<p class='dash-sub'>Store Intelligence &amp; Operations Dashboard</p>",
        unsafe_allow_html=True)
with h2:
    components.html("""
    <div style="font-family:'Poppins',-apple-system,sans-serif;text-align:right;padding-top:4px">
      <div id="ct" style="font-weight:800;color:#161320;font-size:1.15rem;letter-spacing:.3px"></div>
      <div id="cd" style="font-size:.7rem;color:#6b6478;font-weight:600;margin-top:1px"></div>
    </div>
    <script>
      (function tick(){
        var n=new Date();
        document.getElementById('ct').innerText=
          n.toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'});
        document.getElementById('cd').innerText=
          n.toLocaleDateString([],{weekday:'short',year:'numeric',
                                    month:'short',day:'numeric'});
        setTimeout(tick,1000);
      })();
    </script>""", height=52)
st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# MAIN FRAGMENT
#
# THE WEBRTC FIX: webrtc_streamer() is called inside this fragment on
# every run_every tick, keeping _last_rendered_run_count current and
# preventing the stale-context reset that caused play/pause loops.
#
# DATE SELECTOR + CAMERA FILTER live inside the fragment too, so
# changing them triggers a fragment-only rerun — the camera is
# completely unaffected.
# ══════════════════════════════════════════════════════════════════════════

@st.fragment(run_every=PAGE_REFRESH)
def _page() -> None:

    es : EventStore      = st.session_state.es
    sa : StoreAnalytics  = st.session_state.sa

    # ── Top control bar ───────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([1.6, 1.4, 1], gap="small")

    with ctrl1:
        # ── Date selector (calendar UI) ──────────────────────────────────
        available_dates = es.get_available_dates()   # ['2026-09-15', '2026-09-14', ...]
        today_str       = _today_str()
        today_date      = date.today()

        # Earliest selectable day = oldest date with data (fallback: today)
        parsed_dates = []
        for d in available_dates:
            try:
                parsed_dates.append(datetime.strptime(d, "%Y-%m-%d").date())
            except Exception:
                pass
        min_date = min(parsed_dates) if parsed_dates else today_date

        st.markdown("<div style='font-size:.68rem;font-weight:700;"
                    "color:#6b6478;text-transform:uppercase;"
                    "letter-spacing:.4px;margin-bottom:2px'>📅 Business Date</div>",
                    unsafe_allow_html=True)
        picked_date = st.date_input(
            "Business Date",
            value=today_date,
            min_value=min_date,
            max_value=today_date,
            key="sel_date",
            label_visibility="collapsed",
        )
        selected_date = picked_date.isoformat()
        is_today      = (selected_date == today_str)

    with ctrl2:
        # ── Camera filter ──────────────────────────────────────────────────
        cameras = es.get_available_cameras()
        cam_options = ["📹 All Cameras"] + [f"📷 {c}" for c in cameras]
        cam_values  = [None]             + cameras
        st.markdown("<div style='font-size:.68rem;font-weight:600;"
                    "color:#6b7280;text-transform:uppercase;"
                    "letter-spacing:.4px;margin-bottom:2px'>📹 Camera View</div>",
                    unsafe_allow_html=True)
        cam_idx = st.selectbox(
            "Camera View", options=range(len(cam_options)),
            format_func=lambda i: cam_options[i],
            key="sel_cam", label_visibility="collapsed",
        )
        selected_cam: Optional[str] = cam_values[cam_idx]

    with ctrl3:
        # ── Status badge ───────────────────────────────────────────────────
        st.write("")
        st.write("")
        if is_today:
            cam_state = _camera_state()
            if cam_state == "LIVE":
                st.markdown(
                    "<span class='badge badge-live'>"
                    "<span class='live-dot'></span>&nbsp;● LIVE</span>",
                    unsafe_allow_html=True)
            elif cam_state == "CONNECTING":
                st.markdown(
                    "<span class='badge badge-off'>"
                    "<span class='live-dot'></span>&nbsp;Connecting…</span>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    "<span class='badge badge-off'>"
                    "<span class='live-dot'></span>&nbsp;Offline</span>",
                    unsafe_allow_html=True)
        else:
            try:
                disp = datetime.strptime(selected_date, "%Y-%m-%d").strftime("%d %b %Y")
            except Exception:
                disp = selected_date
            st.markdown(
                f"<span class='badge badge-hist'>"
                f"<span class='live-dot'></span>&nbsp;📅 Historical</span>"
                f"<div style='font-size:.72rem;font-weight:700;"
                f"color:#1d4ed8;margin-top:3px'>{disp}</div>",
                unsafe_allow_html=True)

    st.divider()

    # ── Body: camera column + analytics column ────────────────────────────
    col_cam, col_dash = st.columns([1.05, 1.95], gap="medium")

    # ── CAMERA PANEL ──────────────────────────────────────────────────────
    with col_cam:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>📷 Live Store Camera</div>",
                        unsafe_allow_html=True)

            # Multi-camera future: show a tile grid here.
            # For now: single camera_01 tile.
            st.markdown(
                "<div class='cam-tile' style='margin-bottom:.4rem'>"
                "<div class='cam-label'>Camera 01 — Entrance / Shelf / Checkout</div>"
                "</div>",
                unsafe_allow_html=True)

            source = st.selectbox(
                "Source",
                ["Laptop Webcam", "CCTV / IP Camera", "DVR / NVR"],
                key="cam_source", label_visibility="collapsed",
            )

            rtsp: str | None = None
            if source == "CCTV / IP Camera":
                rtsp = st.text_input("RTSP URL", key="rtsp_cctv",
                                     label_visibility="collapsed")
            elif source == "DVR / NVR":
                rtsp = st.text_input("RTSP URL", key="rtsp_dvr",
                                     label_visibility="collapsed")

            if source != "Laptop Webcam" and not rtsp:
                st.info("ℹ️ Enter an RTSP URL to start.")

            if source == "Laptop Webcam" or rtsp:
                mc = (
                    {"video": {"width":  {"ideal": 640},
                               "height": {"ideal": 360},
                               "frameRate": {"ideal": 15}},
                     "audio": False}
                    if source == "Laptop Webcam"
                    else {"video": True, "audio": False}
                )
                # Arm module-level globals before background thread starts.
                _arm_globals()

                # ── stable webrtc_streamer call (inside fragment = key fix) ──
                ctx = webrtc_streamer(
                    key="smart-store-camera",
                    video_processor_factory=SmartStoreVideoProcessor,
                    media_stream_constraints=mc,
                    async_processing=True,
                    sendback_video=True,
                    video_html_attrs={
                        "style": {"width": "100%", "border-radius": "8px",
                                  "display": "block"},
                        "controls": False,
                        "autoPlay": True,
                        "muted": True,
                    },
                )
                new_p = getattr(ctx, "video_processor", None)
                if new_p is not None:
                    st.session_state.proc_ref = new_p
                if st.session_state.proc_ref is None:
                    st.info("👆 Click **START** to begin monitoring.")

            # Historical note when viewing past date
            if not is_today:
                st.info(
                    f"📅 Viewing historical data for "
                    f"**{datetime.strptime(selected_date,'%Y-%m-%d').strftime('%d %b %Y')}**. "
                    f"Live camera continues independently.")

    # ── ANALYTICS COLUMN ──────────────────────────────────────────────────
    with col_dash:

        # Query everything for the selected date + camera
        total  = sa.get_total_customer_for(selected_date, selected_cam)
        chkout = sa.get_checkout_customers_for(selected_date, selected_cam)
        rate   = (chkout / total * 100) if total else None
        mv     = sa.get_most_visited_zone_for(selected_date, selected_cam)
        peak   = es.get_peak_hour_by_date(selected_date, selected_cam)
        avg_d  = _avg_dwell_for(es, selected_date, selected_cam)
        trans  = _total_trans_for(es, selected_date, selected_cam)
        occ    = _zone_occ() if is_today else {}
        busy   = max(occ, key=occ.get) if occ else "N/A"
        inside = _inside() if is_today else "N/A"

        # KPI row 1
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Visitors",         total  or "N/A")
        k2.metric("Inside Now" if is_today else "Unique Visitors",
                  inside if is_today else (total or "N/A"))
        k3.metric("Checkout Customers", chkout or "N/A")
        k4.metric("Conversion Rate",
                  f"{rate:.0f}%" if rate is not None else "N/A")

        # KPI row 2
        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Avg. Dwell Time",   fmt_dur(avg_d))
        k6.metric("Most Visited Zone", mv.get("zone", "N/A") or "N/A")
        k7.metric("Peak / Busy Zone",  busy if is_today else (mv.get("zone","N/A") or "N/A"))
        k8.metric("Peak Period",       peak or "N/A")

        # Zone + tabs
        zc, tc = st.columns([1, 1.2], gap="medium")

        with zc:
            with st.container(border=True):
                st.markdown("<div class='panel-title'>🏬 Zone Analytics</div>",
                            unsafe_allow_html=True)
                bd = _zone_breakdown_for(es, sa, selected_date, selected_cam, is_today)
                if bd:
                    for z, s in bd.items():
                        cur_str = str(s["cur"]) if s["cur"] is not None else ("N/A" if not is_today else "0")
                        eng_str = f"{s['eng']:.0f}%" if s["eng"] is not None else "N/A"
                        st.markdown(f"""
                        <div class='zone-card'>
                          <div class='zone-name'>{z}</div>
                          <div class='zone-stats'>
                            {"Now: " + cur_str + " &nbsp;·&nbsp; " if is_today else ""}
                            Visits: {s['vis'] or 'N/A'}
                            &nbsp;·&nbsp; Dwell: {fmt_dur(s['dwell'])}
                            &nbsp;·&nbsp; Eng: {eng_str}
                          </div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.caption("No zone data for this selection.")

                # Zone transition summary
                tr_dict = es.get_zone_transitions_by_date(selected_date, selected_cam)
                if tr_dict:
                    top_tr = sorted(tr_dict.items(), key=lambda x: x[1], reverse=True)[:3]
                    st.markdown(
                        "<div style='font-size:.67rem;color:#374151;"
                        "margin-top:.3rem;font-weight:600'>Top transitions:</div>",
                        unsafe_allow_html=True)
                    for route, cnt in top_tr:
                        st.markdown(
                            f"<div style='font-size:.65rem;color:#6b7280'>"
                            f"&nbsp;&nbsp;{route} &nbsp;({cnt}×)</div>",
                            unsafe_allow_html=True)

        with tc:
            with st.container(border=True):
                st.markdown("<div class='panel-title'>📊 Insights &amp; Behavior</div>",
                            unsafe_allow_html=True)
                t_ins, t_beh, t_vis, t_cam = st.tabs(
                    ["Insights", "Behavior", "Visitor", "Cameras"])

                with t_ins:
                    ins = _insights_for(sa, es, selected_date, selected_cam)
                    if ins:
                        for i in ins:
                            css = f"ins {'w' if i['s']=='w' else 'a' if i['s']=='a' else ''}"
                            st.markdown(f"""
                            <div class='{css}'>
                              <div class='ins-t'>{i['t']}</div>
                              <div class='ins-d'>{i['m']}</div>
                            </div>""", unsafe_allow_html=True)
                        hi = _high_intent_for(es, selected_date, selected_cam)
                        st.caption(f"High purchase intent: {hi} visitor(s)")
                    else:
                        st.info("Insights appear once tracking data exists.")

                with t_beh:
                    cids = es.get_customer_ids_by_date(selected_date, selected_cam)
                    if cids:
                        b1, b2 = st.columns(2)
                        with b1:
                            st.metric("Visits Tracked", len(cids))
                            try:
                                fe = FeatureEngineerV2(es)
                                zc2 = []
                                for c in cids[:50]:   # cap at 50 for performance
                                    try:
                                        zc2.append(
                                            fe.get_customer_features(c).get("zones_visited", 0))
                                    except Exception:
                                        pass
                                az = sum(zc2) / len(zc2) if zc2 else None
                                st.caption(f"Avg zones: {az:.1f}" if az else "Avg zones: N/A")
                            except Exception:
                                st.caption("Avg zones: N/A")
                        with b2:
                            st.metric("Checkout Rate",
                                      f"{rate:.1f}%" if rate is not None else "N/A")
                            if rate is not None:
                                st.caption(f"{100-rate:.1f}% did not reach checkout")
                    else:
                        st.info("No behavior data for this selection.")

                with t_vis:
                    cids = es.get_customer_ids_by_date(selected_date, selected_cam)
                    if not cids:
                        st.info("No visitor data for this selection.")
                    else:
                        sel = st.selectbox("Visitor ID", cids,
                                           key="vis_sel",
                                           label_visibility="collapsed")
                        cp = st.session_state.get("cp")
                        try:
                            fe   = FeatureEngineerV2(es)
                            feat = fe.get_customer_features(sel)
                            v1, v2 = st.columns(2)
                            if cp:
                                try:
                                    p2   = cp.predict_customer(sel)
                                    prob = p2["conversion_probability"]
                                    lbl  = ("🟢 High"   if prob >= 80
                                            else "🟡 Med" if prob >= 50
                                            else "🔵 Low")
                                    v1.metric("Buy Probability", f"{prob}%")
                                    v2.metric("Intent", lbl)
                                except Exception:
                                    v1.metric("Buy Probability", "N/A")
                                    v2.metric("Intent", "N/A")
                            else:
                                v1.metric("Buy Probability", "N/A")
                                v2.metric("Intent", "N/A")

                            st.metric("Duration", fmt_dur(feat.get("total_dwell", 0)))

                            try:
                                res = st.session_state.ad.detect()
                                m   = next((r for r in res
                                            if r["track_id"] == sel), None)
                                st.metric("Behaviour",
                                          "✅ Normal"
                                          if not m or m["status"] == "NORMAL"
                                          else "⚠️ Unusual")
                            except Exception:
                                pass

                            try:
                                if len(cids) >= 3:
                                    cl = st.session_state.cc.cluster()
                                    m  = next((r for r in cl
                                               if r["track_id"] == sel), None)
                                    if m:
                                        st.metric("Segment",
                                                  f"Group {m['cluster']}")
                            except Exception:
                                pass

                            st.divider()
                            c1, c2 = st.columns(2)
                            c1.write(f"**Zones:** {feat.get('zones_visited',0)}")
                            c2.write(f"**Transitions:** {feat.get('transitions',0)}")
                        except Exception as exc:
                            st.warning(f"Cannot load visitor: {exc}")

                with t_cam:
                    # ── Multi-camera architecture preview ─────────────────
                    st.markdown(
                        "<div class='panel-title' style='border:none'>"
                        "📷 Camera Infrastructure</div>",
                        unsafe_allow_html=True)

                    db_cameras = es.get_available_cameras()

                    # Show a tile for each known camera
                    cam_defs = [
                        ("camera_01", "Entrance / Shelf / Checkout", "Active"),
                        ("camera_02", "Shelf Area (future)",         "Planned"),
                        ("camera_03", "Checkout (future)",           "Planned"),
                        ("camera_04", "Exit (future)",               "Planned"),
                    ]
                    rows = [cam_defs[i:i+2] for i in range(0, len(cam_defs), 2)]
                    for row in rows:
                        cols = st.columns(len(row), gap="small")
                        for col, (cid, zones_desc, status) in zip(cols, row):
                            has_data   = cid in db_cameras
                            is_active  = status == "Active"
                            state_str  = (
                                "● LIVE" if is_active and has_data
                                else "● Active" if is_active
                                else "○ Planned"
                            )
                            badge_style = (
                                "color:#1a7f37;font-weight:700"
                                if is_active else "color:#9ca3af"
                            )
                            with col:
                                st.markdown(f"""
                                <div class='cam-tile'>
                                  <div class='cam-label'>{cid.upper()}</div>
                                  <div style='{badge_style};font-size:.7rem'>
                                    {state_str}</div>
                                  <div style='font-size:.65rem;color:#6b7280;
                                    margin-top:2px'>{zones_desc}</div>
                                </div>""", unsafe_allow_html=True)

                    st.markdown("""
                    <div style='font-size:.67rem;color:#9ca3af;margin-top:.5rem;
                      line-height:1.5'>
                      Each camera has its own ZoneManager config and EventEngine.<br>
                      Events carry <code>camera_id</code> for cross-camera filtering.<br>
                      Future: <code>global_customer_id</code> for cross-camera journeys.
                    </div>""", unsafe_allow_html=True)


_page()
