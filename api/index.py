import os
import sys
import traceback

from flask import Flask

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    from cropcast import app
except Exception:
    error_text = traceback.format_exc()
    app = Flask(__name__)

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def show_error(path):
        return "<h2>Startup error</h2><pre>" + error_text + "</pre>", 500
