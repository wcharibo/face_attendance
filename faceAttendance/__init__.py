from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy


import config

db = SQLAlchemy()
migrate = Migrate()
from . import models

def create_app():
    app = Flask(__name__)
    app.config.from_object(config)

    #ORM
    db.init_app(app)
    migrate.init_app(app, db)

    from .views import main_views
    app.register_blueprint(main_views.bp)


    if __name__ == '__main__':
        app.run(port=3000, debug=True)

    return app

