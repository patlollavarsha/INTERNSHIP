import os
import sys
import traceback

from flask import Flask

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

app = Flask(__name__)
error_text = None

try:
    from cropcast import app as real_app
    app = real_app
except Exception:
    error_text = traceback.format_exc()


@app.before_request
def show_startup_error():
    if error_text:
        return "<h2>Startup error</h2><pre>" + error_text + "</pre>", 500
