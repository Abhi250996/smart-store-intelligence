"""Compatibility entry point. Use `streamlit run dashboard.py` for the main app."""

import runpy

if __name__ == "__main__":
    runpy.run_path("dashboard.py", run_name="__main__")
