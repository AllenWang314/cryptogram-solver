from flask import Flask, session, request

app = None

def create_app():
    global app
    app = Flask(__name__, static_folder='static')

    # register module blueprints
    from backend.index_bp import index_bp
    app.register_blueprint(index_bp)

    return app
