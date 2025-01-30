# This file has no relationship with our project at all.
# This file aims to help you learn relative knowledge
# This file LearningClassInteritence is aimed to help you learn about the class inheritance of python

from LearningClassMethod import Unit

class subUnit1(Unit):
    pass

class subUnit2(Unit):
    pass


a=subUnit1(400,"Timo")
b=subUnit2(500,"John")

print(subUnit1.instances_count)
print(subUnit2.instances_count)