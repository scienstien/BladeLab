"""Main Flask/FastAPI application for TurboDesigner 2.0"""

from flask import Flask

app = Flask(__name__)

# Import routes after app creation to avoid circular imports
from api import routes

# Register blueprints
app.register_blueprint(routes.bp)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
