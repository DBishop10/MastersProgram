# This file has no relationship with our project at all.
# This file aims to help you learn relative knowledge
# This file LearningClassMethod is aimed to help you learn about the class method of python

class MyClass:
    my_class_variable = 10

    @classmethod
    def myClassMethod(cls):
        print("Hello! I am MyClass!")

    @classmethod
    def myClassVariable(cls):
        print("Hello! My class variable is {}".format(cls.my_class_variable))


# MyClass.myClassMethod()
# MyClass.myClassVariable()


class Unit:
    instances_count = 0
    instances_database = []

    def __init__(self, id, name):
        if not Unit.checkIdUniqueness(id):
            raise ValueError("The id already exists!")
        self.id = id
        self.name = name
        Unit.instances_database.append(self)
        Unit.instances_count += 1

    @classmethod
    def getInstanceById(cls, instance_id):
        res = None
        for instance_each in cls.instances_database:
            print(instance_each.id)
            if instance_each.id == instance_id:
                res = instance_each
        return res

    @classmethod
    def getInstancesDatabase(cls):
        return cls.instances_database

    @classmethod
    def getInstancesCount(cls):
        return cls.instances_count

    @classmethod
    def checkIdUniqueness(cls, id_for_check):
        res = True
        for instance_each in cls.instances_database:
            if instance_each.id == id_for_check:
                res = False
        return res
#
# a = Unit(100, "Mike")
# b = Unit(200, "Jack")
# print(Unit.getInstancesDatabase())
# instance=Unit.getInstanceById(100)
# print(instance.name)
# print(Unit.checkIdUniqueness(100))
# print(Unit.checkIdUniqueness(300))
# c = Unit(200,"Biden")
