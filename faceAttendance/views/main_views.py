from flask import Blueprint, render_template , request
from faceAttendance.models import Student, Course, CourseStudent

#for AI
import numpy as np
import cv2
import os
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from skimage.transform import resize
from keras.models import load_model
from retinaface import RetinaFace


bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/', methods=['GET'])
def hello_world():
    course_list = Course.query.order_by(Course.id)
    return render_template('index.html', course_list = course_list)


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

@bp.route('/', methods=['POST'])
def train():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    return render_template('index.html')