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

# 수업-학생 매핑 테이블
class CourseStudent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)

    course = db.relationship('Course', backref=db.backref('students', lazy=True))
    student = db.relationship('Student', backref=db.backref('courses', lazy=True))

# 학생과 출결 체크 간의 관계를 다루는 연결 테이블
student_attendance = db.Table(
    'student_attendance',
    db.Column('student_id', db.Integer, db.ForeignKey('student.id'), primary_key=True),
    db.Column('attendance_check_id', db.Integer, db.ForeignKey('attendance_check.id'), primary_key=True),
)

class AttendanceCheck(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    check_date = db.Column(db.Date, nullable=False)
    result = db.Column(db.Boolean, nullable=False)

    course = db.relationship('Course', backref=db.backref('attendance_checks', lazy=True))
    students = db.relationship('Student', secondary=student_attendance, lazy='subquery', backref=db.backref('attendance_checks', lazy=True))