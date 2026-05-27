#!/bin/bash
# macOS launcher — double-click in Finder to open GliaAnalysis as a native window.
# First time only: make it executable with  chmod +x "Start GliaAnalysis (Native).command"
cd "$(dirname "$0")" || exit 1
echo "============================================================"
echo "  GliaAnalysis  |  native desktop app (WKWebView)"
echo "  The app opens in its own window in a few seconds."
echo "  Close the app window (or press Ctrl+C here) to stop."
echo "============================================================"
echo
exec .venv/bin/python -m glia_dash.desktop
