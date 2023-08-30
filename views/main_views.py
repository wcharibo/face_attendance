from flask import Blueprint, render_template , request

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')


@bp.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    return render_template('index.html')