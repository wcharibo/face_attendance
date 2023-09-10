from flask import Blueprint, render_template , request, url_for
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

@bp.route('/', methods=['GET'])
def my_view():
    global course_id_to_query, image_size, model_path, model, image_dir_basepath, names
    course_id_to_query = 0
    image_size=160
    model_path = url_for('static', filename='model/keras/model/facenet_keras.h5')
    model = load_model(model_path)
    image_dir_basepath = url_for('static', filename='lectures')
    students_in_course = Student.query.join(CourseStudent, (CourseStudent.student_id == Student.id)).filter(CourseStudent.course_id == course_id_to_query).all()
    names = [student.name for student in students_in_course]




@bp.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    return render_template('index.html')

@bp.route('/detail/<int:course_id>/', methods=['GET'])
def detail(course_id):
    course = Course.query.get(course_id)
    global course_id_to_query
    course_id_to_query= course_id
    return render_template('course.html', course = course, id=course_id_to_query)

@bp.route('/detail/<int:course_id>/', methods=['POST'])
def attendance_check():
    return render_template('course.html')

@bp.route('/', methods=['POST'])
def train():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    return render_template('index.html')

def check_trained(course_name):
    if os.path.isfile("../static/lectures/" + course_name +"/embedding/" + course_name):
        embs=np.load(course_name + ".npy")
        return None
    else:
        image_dir_basepath= image_dir_basepath + course_name
        embs, labels = train(image_dir_basepath, names)
        le, clf = test(embs, labels )
        return None

def prewhiten(x):
    #print('prewhiten - x shape: ', x.shape)
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def train_load_and_align_images(filepaths, margin):

    aligned_images = []
    for filepath in filepaths:
        img = cv2.imread(filepath)
        faces = RetinaFace.extract_faces(img, align = True)
        for face in faces:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            aligned = resize(face, (image_size, image_size), anti_aliasing=True)
            aligned_images.append(aligned)
    global var_faces
    var_faces = np.array(aligned_images)

    return np.array(aligned_images)

def test_load_and_align_images(filepaths, margin):

    aligned_images = []
    for filepath in filepaths:
        img = cv2.imread(filepath)
        faces = RetinaFace.extract_faces(img, align = True)
        for face in faces:
            if(face.size<20000):
                continue
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            aligned = resize(face, (image_size, image_size), anti_aliasing=True)
            aligned_images.append(aligned)
    global var_faces
    var_faces = np.array(aligned_images)

    return np.array(aligned_images)

def train_calc_embs(filepaths, margin=10, batch_size=1):
    aligned_images = prewhiten(train_load_and_align_images(filepaths, margin))
    pd = []
    for start in range(0, len(aligned_images), batch_size):
        pd.append(model.predict_on_batch(aligned_images[start:start+batch_size]))
    embs = l2_normalize(np.concatenate(pd))
    #print(embs.shape)

    return embs

def test_calc_embs(filepaths, margin=10, batch_size=1):
    aligned_images = prewhiten(test_load_and_align_images(filepaths, margin))
    pd = []
    for start in range(0, len(aligned_images), batch_size):
        pd.append(model.predict_on_batch(aligned_images[start:start+batch_size]))
    embs = l2_normalize(np.concatenate(pd))
    #print(embs.shape)

    return embs

def train(dir_basepath, names, max_num_img=10):
    labels = []
    embs = []
    for name in names:
        dirpath = os.path.abspath(dir_basepath + name)
        filepaths = [os.path.join(dirpath, f) for f in os.listdir(dirpath)][:max_num_img]
        embs_ = train_calc_embs(filepaths)
        labels.extend([name] * len(embs_))
        embs.append(embs_)
        #print(name)

    embs = np.concatenate(embs)
    return embs, labels

def test(embs, labels):
    le = LabelEncoder().fit(labels)
    y= le.transform(labels)
    clf = clf = SVC(kernel='linear', probability=True).fit(embs, y)
    return le, clf

def infer(le, clf, filepaths):
    embs = test_calc_embs(filepaths)
    pred = le.inverse_transform(clf.predict(embs))
    pred_proba=clf.predict_proba(embs)
    return pred, pred_proba