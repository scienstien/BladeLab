"""Main Flask/FastAPI application for TurboDesigner 2.0"""

import os

from dotenv import load_dotenv
from flask import Flask

load_dotenv()

app = Flask(__name__)

# Import routes after app creation to avoid circular imports
from api import routes

# Register blueprints
app.register_blueprint(routes.bp)

if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode, host="0.0.0.0", port=5000)
