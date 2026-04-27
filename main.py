"""
main.py  –  Entry point for the drawing machine controller.
Run with:  python3 main.py
"""

import sys
import os
import importlib

# Disable bytecode caching so Python always reads the raw .py source.
# This prevents stale .pyc files from loading old versions of edited modules.
sys.dont_write_bytecode = True
importlib.invalidate_caches()

# Also nuke any existing __pycache__ folders on startup
for root, dirs, files in os.walk(os.path.dirname(__file__) or "."):
    for d in dirs:
        if d == "__pycache__":
            import shutil
            shutil.rmtree(os.path.join(root, d))

from gui import App

if __name__ == "__main__":
    app = App()
    app.mainloop()
