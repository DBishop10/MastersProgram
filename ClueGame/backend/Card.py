# Connor Finn
# 11/11/2023
# This class defines a card object according to the design specifications doc

class Card:
    instances_count = 0
    instances_database = []
    def __init__(self, id, type, name):
        Card.checkIdUniqueness(id)
        self.name = name  # e.g. Mr Green, candlestick,
        self.type = type  # 'room', 'person', 'weapon'
        self.id = id  # unique ID. This should not be changed
        Card.instances_database.append(self)
        Card.instances_count += 1

    # set a couple of getters. This is not explicetely necessary for python
    # but I wanted to set the interface.
    def getType(self):
        return self.type

    def getName(self):
        return self.name

    def getId(self):
        return self.id

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
                break
        if not res:
            raise ValueError("The id already exists!")