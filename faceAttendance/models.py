from faceAttendance import db

class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    subject = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text(), nullable=False)
    create_date = db.Column(db.DateTime(), nullable=False)

class Answer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(db.Integer, db.ForeignKey('question.id', ondelete='CASCADE'))
    content = db.Column(db.Text(), nullable=False)
    create_date = db.Column(db.DateTime(), nullable=False)

    question = db.relationship('Question', backref=db.backref('answer_set', ))

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(200), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)  # 비밀번호를 저장할 열

# 학생 테이블
class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)  # 이미지 경로를 저장할 수 있는 열

# 수업 테이블
class Course(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    course_name = db.Column(db.String(200), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)  # 이미지 경로를 저장할 수 있는 열
    professor_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)  # 교수님과의 관계를 나타내는 외래 키

    students = db.relationship('Student', secondary='course_student', backref=db.backref('courses', lazy=True))

# 수업-학생 매핑 테이블
class CourseStudent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id', ondelete='CASCADE'), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id', ondelete='CASCADE'), nullable=False)

class AttendanceCheck(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id', ondelete='CASCADE'), nullable=False)
    student_id = db.Column('student_id', db.Integer, db.ForeignKey('student.id', ondelete='CASCADE'), nullable=False)
    check_week = db.Column(db.Integer, nullable=False)
    result = db.Column(db.Boolean, nullable=False)

    course = db.relationship('Course', backref=db.backref('courses', lazy=True))
    student = db.relationship('Student',  backref=db.backref('students', lazy=True))