from flask import Blueprint, render_template , request, url_for, session, redirect, g, flash, jsonify
from faceAttendance.models import Student, Course, CourseStudent, User, AttendanceCheck
from faceAttendance.forms import CourseCreateForm
from faceAttendance import db

#for AI
import os
import shutil
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
def index():#로그인하면 강의모록 보여주는 페이지
    if g.user:  #로그인되어있으면 강의 몰록 출력
        professor_id = g.user.id  # 현재 로그인한 교수님의 아이디를 가져옴
        course_list = Course.query.filter_by(professor_id=professor_id).order_by(Course.id).all()
        return render_template('index.html', course_list=course_list)
    else:
        print("hello")
    return redirect(url_for('auth.login'))  #로그인 안되어있으면 로그인 페이지로 이동

@bp.route('/course/add/', methods=['GET', 'POST'])#강의를 추가하는 페이지
def courseAdd():
    form = CourseCreateForm()
    file_path = '../static/lectures/'
    students = request.form.getlist('students')
    if request.method == 'POST':
        for student_form in students:
            stu_check=Student.query.filter_by(id=student_form).first()
            if not stu_check:#틀린 학생번호가 적힌 경우 넘어감
                flash(student_form+'잘못된 학번입니다.')
                return render_template('courseAdd.html', form=form)
        course=Course.query.filter_by(course_name=form.coursename.data).first()
        if not course:
            os.makedirs(file_path + str(form.coursename.data)+'/embedding') #디렉토리 생성
            os.mkdir('../static/images/lectures'+ str(form.coursename.data))
            course = Course(course_name=form.coursename.data, image_path='1',professor_id=g.user.id)
            db.session.add(course)
            db.session.commit()
            course=Course.query.filter_by(course_name=form.coursename.data).first()
            for student_form in students:
                stu_check=Student.query.filter_by(id=student_form).first()
                #DB.courseStudent에 추가
                courseStudent=CourseStudent(course_id=course.id, student_id=student_form)
                db.session.add(courseStudent)
                #DB.AttendanceCheck에 추가
                for i in range(15):
                    attendance=AttendanceCheck(course_id=course.id, student_id=student_form, check_week=i+1, result=False)
                    db.session.add(attendance)
                #학생 디렉토리 추가
                os.makedirs(file_path + course.course_name + '/images/' + str(stu_check.id))
                #학생 이미지 저장
                image_source_path = '../static/images/students/' + str(stu_check.id) + '.jpg'
                image_destination_path = file_path + course.course_name + '/images/' + str(stu_check.id) + '/'
                shutil.copy(image_source_path, image_destination_path)
            db.session.commit()
            return redirect(url_for('main.index'))
        else:
            flash(form.coursename.data+'강의는 이미 존재하는 강의입니다.')
    return render_template('courseAdd.html', form=form)

@bp.route('/detail/<int:course_id>/', methods=['GET'])#
def detail(course_id):
    if g.user:
        course = Course.query.get(course_id)
    else:
        flash('로그인 후 사용해주세요')
        return redirect(url_for('auth.login'))
    model_path = '../static/model/keras/model/facenet_keras.h5'
    global model
    model = load_model(model_path)

    attendance_data = []

    students = Student.query.join(CourseStudent, (CourseStudent.student_id == Student.id)).filter(CourseStudent.course_id == course.id).all()

    for student in students:
        attendance_row = {
            'id': student.id,
            'name': student.name,
            'attendance': []
        }
        for week in range(1, 16):
            attendance_check = AttendanceCheck.query.filter_by(student_id=student.id, course_id=course.id,check_week=week).first()
            if attendance_check:
                attendance_row['attendance'].append(True if attendance_check.result else False)
        attendance_data.append(attendance_row)

    return render_template('course.html', course = course, attendance_data = attendance_data)

@bp.route('/update_attendance', methods=['POST'])
def update_attendance():
    course_id = request.form.get('courseId')
    student_id = request.form.get('studentId')
    week = request.form.get('week')
    selected_status = request.form.get('selectedStatus')

    if selected_status.lower() =='true':
        status=True
    elif selected_status.lower() =='false':
        status=False
    else:
        print('no status')

    attendance_check = AttendanceCheck.query.filter_by(
             course_id=course_id,
             student_id=student_id,
             check_week=week  # 어떤 주차의 출석을 업데이트할지 지정
         ).first()
    if attendance_check:
            attendance_check.result = status
    db.session.commit()

    # 새로운 버튼의 HTML을 생성하여 응답으로 보냅니다.
    attendance_check = AttendanceCheck.query.filter_by(
             course_id=course_id,
             student_id=student_id,
             check_week=week  # 어떤 주차의 출석을 업데이트할지 지정
         ).first()
    if attendance_check.result==True:
        new_select_html = f'<select class="form-control" ' \
                  f'onchange="updateAttendance({course_id}, {student_id}, {week}, this.value)">' + \
                  '<option value=True selected>출석</option>' + \
                  '<option value=False>결석</option>' + \
                  '</select>'
    elif attendance_check.result==False:
        new_select_html = f'<select class="form-control" ' \
                  f'onchange="updateAttendance({course_id}, {student_id}, {week}, this.value)">' + \
                  '<option value=True>출석</option>' + \
                  '<option value=False selected >결석</option>' + \
                  '</select>'
    return jsonify({'newSelectHtml': new_select_html})

@bp.route('/detail/<int:course_id>/', methods=['POST'])
def predict(course_id):
    global course
    course = Course.query.get(course_id)

    imagefiles = request.files.getlist('imagefile')
    week_number = int(request.form['week'])

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

    for student_id in result:
        # 해당 주차의 출석 정보가 이미 DB에 있는지 확인
        attendance_check = AttendanceCheck.query.filter_by(
            course_id=course.id,
            student_id=student_id,
            check_week=week_number  # 어떤 주차의 출석을 업데이트할지 지정
        ).first()
        if attendance_check:
            attendance_check.result = True
        # 만약 해당 주차의 출석 정보가 없으면 새로운 레코드를 생성
        # if attendance_check is None:
        #     attendance_check = AttendanceCheck(
        #         course_id=course.id,
        #         student_id=student_id,
        #         check_week=week_number,
        #         result=True  # 얼굴 인식 결과가 True 또는 False인지에 따라 저장할 값 지정
        #     )
        #     db.session.add(attendance_check)
        # else:
        #     이미 해당 주차의 출석 정보가 있는 경우, 결과를 업데이트
        #   얼굴 인식 결과가 True 또는 False인지에 따라 업데이트할 값 지정
    db.session.commit()  # 변경사항을 DB에 저장

    attendance_data = []
    students = Student.query.join(CourseStudent, (CourseStudent.student_id == Student.id)).filter(CourseStudent.course_id == course.id).all()

    for student in students:
        attendance_row = {
            'id': student.id,
            'name': student.name,
            'attendance': []
        }
        for week in range(1, 16):
            attendance_check = AttendanceCheck.query.filter_by(student_id=student.id, course_id=course.id,check_week=week).first()
            if attendance_check:
                attendance_row['attendance'].append(True if attendance_check.result else False)
        attendance_data.append(attendance_row)

    for test_image_path in test_filepaths:
        os.remove(test_image_path)

    return render_template('course.html', course = course, result = result, len_result=len(result), total_students=len(names),attendance_data=attendance_data)

def check_trained(course):
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