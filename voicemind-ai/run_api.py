"""Start the Flask REST API:  python run_api.py"""
from api import create_app
from utils.config import API_HOST, API_PORT

app = create_app()

if __name__ == "__main__":
    app.run(host=API_HOST, port=API_PORT, debug=False, threaded=True)
