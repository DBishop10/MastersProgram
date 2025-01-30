# 10/15/2023
# This init function follows the tutorial: https://www.youtube.com/watch?v=AMp6hlA8xKA

# it imports flask and socket io, then loads in the main from routes.py
# it first initializes our blueprint then initializes the app via socketio.init_app
#

from flask import Flask 
from .events import socketio
from .routes import main 


def create_app():
    app = Flask(__name__)
    app.config["DEBUG"] = True
    app.config["SECRET_KEY"] = "secret"

    app.register_blueprint(main)

    socketio.init_app(app)

    return app
