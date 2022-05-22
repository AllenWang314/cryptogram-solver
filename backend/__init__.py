from flask import Flask, session, request

app = None
datasets = None

def create_app():
    global app
    app = Flask(__name__, static_folder='static')

    app.config["STATIC_DIR"] = "/Users/allen/Desktop/cryptogram-solver/backend/static"

    # register module blueprints
    from backend.index_bp import index_bp
    app.register_blueprint(index_bp)

    return app