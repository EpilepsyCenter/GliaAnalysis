#!/bin/bash
# Linux launcher — open GliaAnalysis as a native window (GTK/Qt WebKit).
# First time only: make it executable with  chmod +x start-gliaanalysis-native.sh
# Requires the desktop add-on with a Linux webview backend:
#     pip install -r requirements-desktop.txt   # pulls pywebview[qt]
# (Qt bundles its own web engine; no system WebKit packages needed.)
cd "$(dirname "$0")" || exit 1
echo "============================================================"
echo "  GliaAnalysis  |  native desktop app (Linux WebKit)"
echo "  The app opens in its own window in a few seconds."
echo "  Close the app window (or press Ctrl+C here) to stop."
echo "============================================================"
echo
exec .venv/bin/python -m glia_dash.desktop
