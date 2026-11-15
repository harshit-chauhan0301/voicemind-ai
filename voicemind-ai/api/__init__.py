
"""Flask application factory."""
from flask import Flask, jsonify

from utils.config import MAX_AUDIO_MB


def create_app() -> Flask:
    app = Flask(__name__)

    app.config["MAX_CONTENT_LENGTH"] = (
        MAX_AUDIO_MB + 5
    ) * 1024 * 1024

    from api.routes import api_bp

    app.register_blueprint(api_bp, url_prefix="/api")

    @app.route("/", methods=["GET"])
    def home():
        return jsonify({
            "status": "running",
            "message": "VoiceMind AI API is live"
        }), 200

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "service": "VoiceMind AI Flask API"
        }), 200

    return app

