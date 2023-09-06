from flask import Blueprint, render_template , request

from faceAttendance.models import Student

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/', methods=['GET'])
def hello_world():
    student_list = Student.query.order_by(Student.id.desc())
    return render_template('index.html', student_list = student_list)


@bp.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    return render_template('index.html')

@bp.route('/detail/<int:student_id>/', methods=['GET'])
def hello_student(student_id):
    student = Student.query.get(student_id)
    return render_template('student.html', student = student)