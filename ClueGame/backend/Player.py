# Shawn Mai
# 11/11/2023
# This class defines a Player object for clue less

from backend.Board import Gameboard
from backend.Card import Card
import random 

class Player:
    instances_count = 0
    instances_database = []
    def __init__(self, id, characterId, characterName, name, roomId, room, hand):
        Player.checkIdUniqueness(id)
        self.name=name
        self.id = id  #unique player id
        self.characterId = characterId  #unique character id
        self.characterName = characterName
        self.roomId = roomId  #unqiue room id 
        self.room = room
        self.hand = hand  # List of Card objects
        self.isActive = True
        self.wasMovedInSuggestion = False
        Player.instances_database.append(self)
        Player.instances_count += 1

    def move(self, new_room):
        # Before calling this method, need to check if the room id is valid 
        # Update the roomId if the move is valid
        self.roomId = new_room.id
        self.room = new_room

    def makeSuggestion(self, suggested_cards):
        # This method return a list of Card objects representing the suggestion

        # Check if all card types are different
        card_types = [card.getType() for card in suggested_cards]
        if len(set(card_types)) != len(suggested_cards):
            raise ValueError("All card types must be different for a suggestion.")
        
        return suggested_cards

    def answerSuggestion(self, suggested_cards):
        # ask the user which card to return 
        # get the intersection of the players hand and the suggested cards
        possible_cards =  [card for card in suggested_cards if card in self.hand]

        # if there are none,  return None
        if len(possible_cards) == 0:
            return None
        
        # otherwise ask the user which card to return
        else:
            card_names = [card.name for card in possible_cards]
            output_string = "Which card would you like to return? \n "

            for i1 in range(len(card_names)):
                output_string = output_string + str(i1) + ": " + card_names[i1] + " \n "
            
            # get player selection: to do update to UI
            card_num = input(output_string)      
            return possible_cards[int(card_num)]



    def makeAccusation(self, accusation_cards):
        # This method return a list of Card objects representing the accusation

        # Check if all card types are different
        card_types = [card.getType() for card in accusation_cards]
        if len(set(card_types)) != len(accusation_cards):
            raise ValueError("All card types must be different for a suggestion.")

        return accusation_cards

    @classmethod
    def getInstanceById(cls, instance_id):
        res = None
        for instance_each in Player.instances_database:
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

    @classmethod
    def gethtmlLocal(cls, room, suggested):
        x = room.htmlLocation[0]
        y = room.htmlLocation[1]
        middlex = (x[0] + x[1])/2
        if(suggested):
            middlex = middlex - 50
        middley = (y[0] + y[1])/2
        return (str(middlex -313) + "px, " + str(middley - 15) + "px")
