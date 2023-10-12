from flask import Blueprint, render_template , request, url_for, session, redirect, g
from faceAttendance.models import Student, Course, CourseStudent, User

#for AI
import os
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from skimage.transform import resize
from keras.models import load_model
from retinaface import RetinaFace


bp = Blueprint('main', __name__, url_prefix='/')
os.chdir(bp.root_path)

@bp.before_app_request
def load_logged_in_user():
    user_id = session.get('user_id')
    if user_id is None:
        g.user = None
    else:
        g.user = User.query.get(user_id)

@bp.route('/', methods=['GET'])
def hello_world():
    if g.user:
        professor_id = g.user.id  # 현재 로그인한 교수님의 아이디를 가져옴
        course_list = Course.query.filter_by(professor_id=professor_id).order_by(Course.id).all()
        return render_template('index.html', course_list=course_list)
    else:
        print("hello")
    return render_template('index.html')

@bp.route('/detail/<int:course_id>/', methods=['GET'])
def detail(course_id):
    course = Course.query.get(course_id)
    global course_id_to_query
    course_id_to_query= course_id
    # model_path = '../static/model/keras/model/facenet_keras.h5'
    # global model
    # model = load_model(model_path)
    return render_template('course.html', course = course, id=course_id_to_query)

@bp.route('/detail/<int:course_id>/', methods=['POST'])
def predict(course_id):
    global course
    course = Course.query.get(course_id)

    imagefiles = request.files.getlist('imagefile')

    for imagefile in imagefiles:
        if imagefile.filename !='':
            test_image_path = '../static/images/lectures/' + course.course_name + '/' + imagefile.filename
            imagefile.save(test_image_path)


    global image_size
    image_size=160

    embs, labels, names = check_trained(course)
    le, clf = test(embs, labels )

    test_dirpath = '../static/images/lectures/' + course.course_name + '/'
    test_filepaths = [os.path.join(test_dirpath, f) for f in os.listdir(test_dirpath)]

    pred, pred_proba = infer(le, clf, test_filepaths)
    result = list(np.unique(pred))

    for test_image_path in test_filepaths:
        os.remove(test_image_path)

    return render_template('course.html', course = course, id=course_id_to_query, result = result, len_result=len(result), total_students=len(names))

# def check_trained(course):
    students_in_course = Student.query.join(CourseStudent, (CourseStudent.student_id == Student.id)).filter(CourseStudent.course_id == course.id).all()
    names = [student.id for student in students_in_course]

    if os.path.isfile("../static/lectures/" + course.course_name +"/embedding/np_embs.npy"):
        embs=np.load('../static/lectures/'+course.course_name+'/embedding/np_embs.npy')
        labels = []
        with open('../static/lectures/'+course.course_name+'/embedding/labels.txt' , 'r') as file:
            for line in file:
                labels.append(line.strip())

        return embs, labels, names
    else:
        image_dir_basepath= '../static/lectures/' + course.course_name + '/images/'
        embs, labels = train(image_dir_basepath, names)
        return embs, labels, names

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
        dirpath = os.path.abspath(dir_basepath + str(name))
        filepaths = [os.path.join(dirpath, f) for f in os.listdir(dirpath)][:max_num_img]
        embs_ = train_calc_embs(filepaths)
        labels.extend([name] * len(embs_))
        embs.append(embs_)
        #print(name)

    embs = np.concatenate(embs)
    np.save('../static/lectures/'+course.course_name+'/embedding/np_embs.npy', embs)
    with open('../static/lectures/'+course.course_name+'/embedding/labels.txt', 'w') as file:
        for item in labels:
            file.write(f'{item}\n')
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