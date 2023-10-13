from faceAttendance import db
from faceAttendance.models import Student, Course, CourseStudent

# 학생 데이터 추가
students_data = [
    (2018001, '이중현', '1'),
    (2018002, '김을중', '1'),
    (2018003, '고종현', '1'),
    (2018004, '서범석', '1'),
    (2018005, '이제훈', '1'),
    (2018006, '빌게이츠', '1'),
    (2018007, '래리페이지', '1'),
    (2018008, '마크저커버그', '1')
]

for data in students_data:
    student = Student(id=data[0], name=data[1], image_path=data[2])
    db.session.add(student)
    db.session.commit()

# 강좌 데이터 추가
courses_data = [
    ('서양문화예술기행', '1', 123),
    ('동양문화예술기행', '1', 123)
]

for data in courses_data:
    course = Course(course_name=data[0], image_path=data[1], professor_id=data[2])
    db.session.add(course)
    db.session.commit()

# 수강생 데이터 추가
course_students_data = [
    (1, 2018001),
    (1, 2018002),
    (1, 2018003),
    (1, 2018004),
    (1, 2018005),
    (1, 2018006),
    (2, 2018001),
    (2, 2018002),
    (2, 2018003),
    (2, 2018004),
    (2, 2018007),
    (2, 2018008)
]

for data in course_students_data:
    course_student = CourseStudent(course_id=data[0], student_id=data[1])
    db.session.add(course_student)
    db.session.commit()