from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    Name: str = 'GGR'  # Setting default value
    Roll_no:int
    Age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0 , lt =10, default=6.5, description='A decimal value represnting the cgpa of the stuudent')  # Here constraint is applied using Field function


new_student = {'Roll_no':19, 'Age':21,'email':'abc@gmail.com','cgpa':9.8}

student = Student(**new_student)
print(student)
print(type(student))

student_dict = dict(student)
print(student_dict['email'])

student_json = student.model_dump_json()
print(student_json)
