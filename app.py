"""GliaAnalysis — Dash entry point.

Run with:  python app.py
"""

from glia_dash.main import app


if __name__ == "__main__":
    app.run(debug=True, port=8050)
