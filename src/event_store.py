"""
EventStore — SQLite persistence for Smart Store Intelligence
============================================================

Schema (current):
  id, track_id, event_type, from_zone, to_zone,
  zone_name, dwell_sec, timestamp

Schema additions (migration applied in create_table()):
  store_id   TEXT DEFAULT 'store_01'
  camera_id  TEXT DEFAULT 'camera_01'

Migration is safe: ALTER TABLE ADD COLUMN is ignored if the
column already exists (caught and swallowed).

All original methods are preserved unchanged.
New *_by_date / *_by_camera methods accept:
  date_str  : 'YYYY-MM-DD'  (None → no date filter)
  camera_id : e.g. 'camera_01' (None → all cameras)
"""

from __future__ import annotations

import sqlite3
from typing import Optional


class EventStore:
    def __init__(self, db_path: str = "store_events.db"):
        self.db_path = db_path
        self.create_table()

    # ── connection ────────────────────────────────────────────────────────

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = None          # keep tuple rows (existing code expects tuples)
        return conn

    # ── schema + migration ────────────────────────────────────────────────

    def create_table(self) -> None:
        base_ddl = """
        CREATE TABLE IF NOT EXISTS events(
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id   INTEGER NOT NULL,
            event_type TEXT    NOT NULL,
            from_zone  TEXT,
            to_zone    TEXT,
            zone_name  TEXT,
            dwell_sec  REAL    DEFAULT 0.0,
            timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP,
            store_id   TEXT    DEFAULT 'store_01',
            camera_id  TEXT    DEFAULT 'camera_01'
        );
        """
        with self._get_connection() as conn:
            conn.execute(base_ddl)
            # Safe migration: add new columns to existing databases.
            for col, defn in [
                ("store_id",  "TEXT DEFAULT 'store_01'"),
                ("camera_id", "TEXT DEFAULT 'camera_01'"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE events ADD COLUMN {col} {defn}")
                except sqlite3.OperationalError:
                    pass   # column already exists — fine
            conn.commit()

    # ── write ─────────────────────────────────────────────────────────────

    def save_event(self, event: dict) -> None:
        """Persist an event dict.  Accepts optional store_id / camera_id."""
        if not event:
            return
        query = """
        INSERT INTO events(
            track_id, event_type, from_zone, to_zone,
            zone_name, dwell_sec, store_id, camera_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        values = (
            event.get("track_id"),
            event.get("event_type"),
            event.get("from_zone"),
            event.get("to_zone"),
            event.get("zone_name"),
            float(event.get("dwell_sec", 0.0) or 0.0),
            event.get("store_id",  "store_01"),
            event.get("camera_id", "camera_01"),
        )
        with self._get_connection() as conn:
            conn.execute(query, values)
            conn.commit()

    # ── helpers ───────────────────────────────────────────────────────────

    def _date_bounds(self, date_str: Optional[str]):
        """Return (start, end) ISO strings for a YYYY-MM-DD date, or (None, None)."""
        if not date_str:
            return None, None
        return f"{date_str} 00:00:00", f"{date_str} 23:59:59"

    def _where(
        self,
        date_str: Optional[str] = None,
        camera_id: Optional[str] = None,
        extra: str = "",
    ) -> tuple[str, list]:
        """Build a WHERE clause fragment + params list."""
        clauses: list[str] = []
        params:  list      = []
        if date_str:
            clauses.append("timestamp BETWEEN ? AND ?")
            start, end = self._date_bounds(date_str)
            params.extend([start, end])
        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if extra:
            clauses.append(extra)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        return where, params

    # ── original read methods (unchanged) ────────────────────────────────

    def get_events(self, limit: int = 100) -> list:
        query = """
        SELECT id, track_id, event_type, zone_name, dwell_sec, timestamp
        FROM events
        ORDER BY id DESC
        LIMIT ?;
        """
        with self._get_connection() as conn:
            return conn.execute(query, (limit,)).fetchall()

    def get_customer_events(self, track_id: int) -> list:
        query = """
        SELECT id, track_id, event_type, from_zone, to_zone,
               zone_name, dwell_sec, timestamp
        FROM events
        WHERE track_id = ?
        ORDER BY id ASC;
        """
        with self._get_connection() as conn:
            return conn.execute(query, (track_id,)).fetchall()

    def get_total_customer(self) -> int:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT track_id) FROM events"
            ).fetchone()
        return int(row[0] or 0)

    def get_customers_most_visit_zone(self) -> dict:
        query = """
        SELECT zone_name, COUNT(DISTINCT track_id)
        FROM events
        WHERE zone_name IS NOT NULL
        GROUP BY zone_name;
        """
        with self._get_connection() as conn:
            rows = conn.execute(query).fetchall()
        return {r[0]: int(r[1]) for r in rows}

    def get_average_dwell_by_zone(self) -> dict:
        query = """
        SELECT zone_name, AVG(dwell_sec)
        FROM events
        WHERE zone_name IS NOT NULL
          AND dwell_sec IS NOT NULL
          AND event_type != 'LONG_DWELL'
        GROUP BY zone_name;
        """
        with self._get_connection() as conn:
            rows = conn.execute(query).fetchall()
        return {r[0]: round(float(r[1]), 2) for r in rows}

    def get_checkout_customers(self) -> int:
        query = """
        SELECT COUNT(DISTINCT track_id) FROM events
        WHERE (event_type = 'ZONE_ENTRY'      AND zone_name = 'Checkout')
           OR (event_type = 'ZONE_TRANSITION' AND to_zone   = 'Checkout');
        """
        with self._get_connection() as conn:
            row = conn.execute(query).fetchone()
        return int(row[0] or 0)

    def get_zone_transitions(self) -> dict:
        query = """
        SELECT from_zone, to_zone, COUNT(*)
        FROM events
        WHERE event_type = 'ZONE_TRANSITION'
        GROUP BY from_zone, to_zone
        ORDER BY COUNT(*) DESC;
        """
        with self._get_connection() as conn:
            rows = conn.execute(query).fetchall()
        return {f"{r[0]} -> {r[1]}": int(r[2]) for r in rows}

    def get_customer_ids(self) -> list:
        with self._get_connection() as conn:
            return [
                r[0] for r in conn.execute(
                    "SELECT DISTINCT track_id FROM events ORDER BY track_id"
                ).fetchall()
            ]

    # ── new date / camera -filtered methods ──────────────────────────────

    def get_available_dates(self) -> list[str]:
        """Return sorted list of 'YYYY-MM-DD' strings that have events."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT DATE(timestamp) FROM events "
                "ORDER BY DATE(timestamp) DESC"
            ).fetchall()
        return [r[0] for r in rows if r[0]]

    def get_available_cameras(self) -> list[str]:
        """Return sorted list of camera_id values present in the DB."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT camera_id FROM events ORDER BY camera_id"
            ).fetchall()
        return [r[0] for r in rows if r[0]]

    def get_events_by_date(
        self,
        date_str: Optional[str],
        camera_id: Optional[str] = None,
        limit: int = 500,
    ) -> list:
        where, params = self._where(date_str, camera_id)
        params.append(limit)
        query = f"""
        SELECT id, track_id, event_type, zone_name, dwell_sec, timestamp
        FROM events
        {where}
        ORDER BY id DESC
        LIMIT ?;
        """
        with self._get_connection() as conn:
            return conn.execute(query, params).fetchall()

    def get_customer_events_by_date(
        self,
        track_id: int,
        date_str: Optional[str],
        camera_id: Optional[str] = None,
    ) -> list:
        where, params = self._where(date_str, camera_id,
                                    extra="track_id = ?")
        params.append(track_id)
        query = f"""
        SELECT id, track_id, event_type, from_zone, to_zone,
               zone_name, dwell_sec, timestamp
        FROM events
        {where}
        ORDER BY id ASC;
        """
        with self._get_connection() as conn:
            return conn.execute(query, params).fetchall()

    def get_total_customer_by_date(
        self,
        date_str: Optional[str],
        camera_id: Optional[str] = None,
    ) -> int:
        where, params = self._where(date_str, camera_id)
        query = f"SELECT COUNT(DISTINCT track_id) FROM events {where};"
        with self._get_connection() as conn:
            row = conn.execute(query, params).fetchone()
        return int(row[0] or 0)

    def get_checkout_customers_by_date(
        self,
        date_str: Optional[str],
        camera_id: Optional[str] = None,
    ) -> int:
        # Build separate WHERE for the two checkout conditions
        clauses: list[str] = []
        params:  list      = []
        if date_str:
            start, end = self._date_bounds(date_str)
            clauses.append("timestamp BETWEEN ? AND ?")
            params.extend([start, end])
        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        base_where = (" AND ".join(clauses) + " AND ") if clauses else ""
        query = f"""
        SELECT COUNT(DISTINCT track_id) FROM events
        WHERE {base_where}(
            (event_type = 'ZONE_ENTRY'      AND zone_name = 'Checkout')
         OR (event_type = 'ZONE_TRANSITION' AND to_zone   = 'Checkout')
        );
        """
        with self._get_connection() as conn:
            row = conn.execute(query, params).fetchone()
        return int(row[0] or 0)

    def get_customers_most_visit_zone_by_date(
        self,
        date_str: Optional[str],
        camera_id: Optional[str] = None,
    ) -> dict:
        where, params = self._where(date_str, camera_id,
                                    extra="zone_name IS NOT NULL")
        query = f"""
        SELECT zone_name, COUNT(DISTINCT track_id)
        FROM events {where}
        GROUP BY zone_name;
        """
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return {r[0]: int(r[1]) for r in rows}

    def get_average_dwell_by_zone_by_date(
        self,
        date_str: Optional[str],
        camera_id: Optional[str] = None,
    ) -> dict:
        where, params = self._where(
            date_str, camera_id,
            extra="zone_name IS NOT NULL AND dwell_sec IS NOT NULL AND event_type != 'LONG_DWELL'",
        )
        query = f"""
        SELECT zone_name, AVG(dwell_sec)
        FROM events {where}
        GROUP BY zone_name;
        """
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return {r[0]: round(float(r[1]), 2) for r in rows}

    def get_zone_transitions_by_date(
        self,
        date_str: Optional[str],
        camera_id: Optional[str] = None,
    ) -> dict:
        where, params = self._where(date_str, camera_id,
                                    extra="event_type = 'ZONE_TRANSITION'")
        query = f"""
        SELECT from_zone, to_zone, COUNT(*)
        FROM events {where}
        GROUP BY from_zone, to_zone
        ORDER BY COUNT(*) DESC;
        """
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return {f"{r[0]} -> {r[1]}": int(r[2]) for r in rows}

    def get_customer_ids_by_date(
        self,
        date_str: Optional[str],
        camera_id: Optional[str] = None,
    ) -> list:
        where, params = self._where(date_str, camera_id)
        query = f"""
        SELECT DISTINCT track_id FROM events {where}
        ORDER BY track_id;
        """
        with self._get_connection() as conn:
            return [r[0] for r in conn.execute(query, params).fetchall()]

    def get_peak_hour_by_date(
        self,
        date_str: Optional[str],
        camera_id: Optional[str] = None,
    ) -> Optional[str]:
        """Return the peak hour label like '5 PM - 6 PM', or None."""
        where, params = self._where(date_str, camera_id)
        query = f"""
        SELECT strftime('%H', timestamp) AS hr, COUNT(DISTINCT track_id)
        FROM events {where}
        GROUP BY hr
        ORDER BY COUNT(DISTINCT track_id) DESC
        LIMIT 1;
        """
        with self._get_connection() as conn:
            row = conn.execute(query, params).fetchone()
        if not row or row[0] is None:
            return None
        try:
            h = int(row[0])
            def _fmt(hh: int) -> str:
                if hh == 0:   return "12 AM"
                if hh < 12:   return f"{hh} AM"
                if hh == 12:  return "12 PM"
                return f"{hh - 12} PM"
            return f"{_fmt(h)} - {_fmt(h + 1)}"
        except Exception:
            return None

    def get_zone_analytics_by_date(
        self,
        date_str: Optional[str],
        camera_id: Optional[str] = None,
    ) -> dict:
        """Combined zone analytics (visitors + avg dwell) for a date."""
        visitors   = self.get_customers_most_visit_zone_by_date(date_str, camera_id)
        avg_dwells = self.get_average_dwell_by_zone_by_date(date_str, camera_id)
        zones = sorted(set(list(visitors.keys()) + list(avg_dwells.keys())))
        return {
            z: {
                "visitors":  visitors.get(z, 0),
                "avg_dwell": avg_dwells.get(z),
            }
            for z in zones
        }
