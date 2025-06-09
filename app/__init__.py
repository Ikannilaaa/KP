from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS

socketio = SocketIO(cors_allowed_origins="*", async_mode="eventlet")

def create_app():
    app = Flask(__name__)

    CORS(app)

    from .routes import pose_bp
    app.register_blueprint(pose_bp)

    socketio.init_app(app)

    return app
