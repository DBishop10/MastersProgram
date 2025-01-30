class Room:
    instances_count = 0
    instances_database = []
    room_names = ["Study", "Hall", "Lounge", "Library", "Billiard Room", "Dining Room", "Conservatory", "Ballroom", "Kitchen"]
    hallway_names = ["StudyHall", "HallLounge", "StudyLibrary", "HallBilliard", "LoungeDining", "LibraryBilliard",
                     "BilliardDining", "LibraryConservatory", "BilliardBall", "DiningKitchen", "ConservatoryBall",
                     "BallKitchen"]
    mapping_from_name_to_boolean = {name: False for name in room_names + hallway_names}

    def __init__(self, id: int, isHallway: bool, htmlLocation, name: str):
        Room.checkName(name)
        Room.checkIdUniqueness(id)
        self.id = id
        self.isHallway = isHallway
        self.name = name
        self.htmlLocation = htmlLocation
        # Hi David, I believe that "name" is a better name for this attribute than "location", because it is a string
        # What is more, consider that we have function get_room_by_name
        Room.instances_database.append(self)
        Room.instances_count += 1

    @classmethod
    def checkName(cls, name):
        res = False
        if name in cls.room_names:
            res = True
        if name in cls.hallway_names:
            res = True
        if not res:
            raise ValueError("The room name doesn't belong the game ClueLess!")
        if cls.mapping_from_name_to_boolean[name]:
            raise ValueError("The room name is already taken up by other rooms")

    @classmethod
    def getInstanceById(cls, instance_id):
        res = None
        for instance_each in cls.instances_database:
            if instance_each.id == instance_id:
                res = instance_each
        return res

    @classmethod
    def getInstanceByName(cls,instance_name):
        res = None
        for instance_each in cls.instances_database:
            if instance_each.name == instance_name:
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



class Gameboard:
    def __init__(self):
        # Initialize all rooms and hallways
        room_names = ["Study", "Hall", "Lounge", "Library", "Billiard Room", "Dining Room", "Conservatory", "Ballroom", "Kitchen"]
        hallway_names = ["StudyHall", "HallLounge", "StudyLibrary", "HallBilliard", "LoungeDining", "LibraryBilliard",
                         "BilliardDining", "LibraryConservatory", "BilliardBall", "DiningKitchen", "ConservatoryBall",
                         "BallKitchen"]

        self.htmlLayout = {"Study": [(423, 564), (170, 297)], "Ballroom": [(639, 805), (587, 715)],
                            "Hall": [(642,805), (169,295)],
                           "Lounge": [(880,1022), (169,298)], "Library": [(423,564), (369,517)], "Billiard Room": [(638,806), (367,518)], "Dining Room": [(880,1024), (368,520)],
                           "Conservatory": [(423,563), (587,715)],
                           "Kitchen": [(880,1024), (586,716)]}
        self.htmlSecretPassageLayout = {"Kitchen": [(427, 479), (177, 229)], "Conservatory": [(955, 1013), (180, 242)],
                                        "Lounge": [(430, 480), (656, 709)], "Study": [(967, 1018), (653, 704)]}
        self.htmlHallLayout = {"StudyHall": [(566,636), (189,256)], "HallLounge": [(808,879), (191,256)],
                               "StudyLibrary": [(466,517), (302,365)], "HallBilliard": [(688,759), (302,365)],
                               "LoungeDining": [(927,1000), (302,365)], "LibraryBilliard": [(566,636), (410,477)],
                               "BilliardDining": [(808,879), (410,477)],
                               "LibraryConservatory": [(446,517), (522,583)],
                               "BilliardBall": [(686,759), (522,583)], "DiningKitchen": [(927,1000), (522,583)],
                               "ConservatoryBall": [(566,636), (630,697)], "BallKitchen": [(807,877), (630,696)]}

        # We follow the rule of "from left to right, from up to below" to give hallway their names
        # For example, the hallway between Hall and Billiary should be "HallBilliard", not "BilliardHall"
        self.rooms = {name: Room(i * 2 + 1, False, self.htmlLayout[name], name) for i, name in enumerate(room_names)}
        self.hallways = {name: Room(i * 2 + 2, True, self.htmlHallLayout[name], name) for i, name in enumerate(hallway_names)}

        self.layout = {
            self.rooms["Study"]: [self.rooms["Library"], self.rooms["Hall"], self.rooms["Kitchen"], self.hallways["StudyHall"],
                                  self.hallways["StudyLibrary"]],
            self.rooms["Hall"]: [self.rooms["Study"], self.rooms["Billiard Room"], self.rooms["Lounge"], self.hallways["StudyHall"],
                                 self.hallways["HallLounge"], self.hallways["HallBilliard"]],
            self.rooms["Lounge"]: [self.rooms["Hall"], self.rooms["Dining Room"], self.rooms["Conservatory"],
                                   self.hallways["HallBilliard"], self.hallways["LoungeDining"]],
            self.rooms["Library"]: [self.rooms["Study"], self.rooms["Billiard Room"], self.rooms["Conservatory"],
                                    self.hallways["StudyLibrary"], self.hallways["LibraryBilliard"], self.hallways["LibraryConservatory"]],
            self.rooms["Billiard Room"]: [self.rooms["Hall"], self.rooms["Library"], self.rooms["Dining Room"],
                                     self.rooms["Ballroom"], self.hallways["HallBilliard"], self.hallways["LibraryBilliard"], self.hallways["BilliardDining"],
                                     self.hallways["BilliardBall"]],
            self.rooms["Dining Room"]: [self.rooms["Billiard Room"], self.rooms["Lounge"], self.rooms["Kitchen"],
                                   self.hallways["LoungeDining"], self.hallways["BilliardDining"], self.hallways["DiningKitchen"]],
            self.rooms["Conservatory"]: [self.rooms["Library"], self.rooms["Ballroom"], self.rooms["Lounge"],
                                         self.hallways["LibraryConservatory"], self.hallways["ConservatoryBall"]],
            self.rooms["Ballroom"]: [self.rooms["Conservatory"], self.rooms["Billiard Room"], self.rooms["Kitchen"],
                                     self.hallways["BilliardBall"], self.hallways["ConservatoryBall"], self.hallways["BallKitchen"]],
            self.rooms["Kitchen"]: [self.rooms["Ballroom"], self.rooms["Dining Room"], self.rooms["Study"], self.hallways["DiningKitchen"],
                                    self.hallways["BallKitchen"]]}
        self.hallLayout = {
            self.hallways["StudyHall"]: [self.rooms["Study"], self.rooms["Hall"]], 
            self.hallways["HallLounge"]: [self.rooms["Lounge"], self.rooms["Hall"]],
            self.hallways["StudyLibrary"]: [self.rooms["Study"], self.rooms["Library"]],
            self.hallways["HallBilliard"]: [self.rooms["Billiard Room"], self.rooms["Hall"]],
            self.hallways["LoungeDining"]: [self.rooms["Lounge"], self.rooms["Dining Room"]],
            self.hallways["LibraryBilliard"]: [self.rooms["Library"], self.rooms["Billiard Room"]],
            self.hallways["BilliardDining"]: [self.rooms["Billiard Room"], self.rooms["Dining Room"]],
            self.hallways["LibraryConservatory"]: [self.rooms["Library"], self.rooms["Conservatory"]],
            self.hallways["BilliardBall"]: [self.rooms["Billiard Room"], self.rooms["Ballroom"]], 
            self.hallways["DiningKitchen"]: [self.rooms["Kitchen"], self.rooms["Dining Room"]],
            self.hallways["ConservatoryBall"]: [self.rooms["Conservatory"], self.rooms["Ballroom"]], 
            self.hallways["BallKitchen"]: [self.rooms["Ballroom"], self.rooms["Kitchen"]]
        }

    def get_room_by_id(self, room_id):
        # Check in rooms
        for room in self.rooms.values():
            if room.id == room_id:
                return room

        # Check in hallways
        for room in self.hallways.values():
            if room.id == room_id:
                return room

        # If no room is found
        return None

    def get_room_by_name(self, room_name):
        # Check in rooms
        for room in self.rooms.values():
            if room.name == room_name:
                return room

        # Check in hallways
        for room in self.hallways.values():
            if room.name == room_name:
                return room

        # If no room is found
        return None

    # Determines where user clicked on board and returns the room, Hall, or secret passage
    def determine_html_location(self, x, y):
        for room in self.htmlLayout.items():
            location = room[1]
            if (location[0][0] <= x <= location[0][1] and location[1][0] <= y <= location[1][1]):
                if (room[0] == "Study" or room[0] == "Lounge" or room[0] == "Conservatory" or room[0] == "Kitchen"):
                    for secret in self.htmlSecretPassageLayout.items():
                        secretpassage = secret[1]
                        if (secretpassage[0][0] <= x <= secretpassage[0][1] and secretpassage[1][0] <= y <=
                                secretpassage[1][1]):
                            return secret[0]
                    else:
                        return room[0]
                else:
                    return room[0]
        for halls in self.htmlHallLayout.items():
            hallLocation = halls[1]
            if (hallLocation[0][0] <= x <= hallLocation[0][1] and hallLocation[1][0] <= y <= hallLocation[1][1]):
                return halls[0]
        return

#       Layout Graph
# Study         Hall       Lounge
# Library       Billiard Room   Dining Room
# Conservatory  Ballroom   Kitchen
