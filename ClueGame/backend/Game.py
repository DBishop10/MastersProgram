# Connor Finn
# 11/11/2023
# This class defines a game object for clue less
from backend.Card import Card
from backend.Deck import Deck
from backend.Board import Gameboard, Room
from backend.Player import *
from client.sql import *

class Game:

    def __init__(self):
        self.players = []  # this is a list of player object
        self.board = []  # this will be the GameBoard object
        self.cards = Deck()
        self.solution = []
        self.numPlayers = 0
        self.turnPhase  = "initialize"    # move, suggestion, accusation
        self.playerTurn = 0
        self.currentPlayerId = None
        self.finished = False
        self.numActivePlayers = 0
        self.started = False
    # this method will start the game, after this, no players will be added
    def initializeGame(self):
        # we need to deal the cards to the players
        # build a list of all the player ids
        self.numPlayers = len(self.players)
        playerIds_temp = list(range(0, self.numPlayers))

        self.numActivePlayers = self.numPlayers
        # deal out the cards
        dealt_cards = self.cards.Deal(playerIds_temp)

        # set the solution
        self.solution = dealt_cards["solution"]

        # set the player hands
        for i in range(self.numPlayers):
            self.players[i].hand = dealt_cards[i]

        self.currentPlayerId = self.players[self.playerTurn].id

        self.started = True

    def move_phase(self):
        curr_player = self.players[self.playerTurn]
        if curr_player.wasMovedInSuggestion == True:

            # TODO make this a UI
            print("\n \n \n \n \n \n \n \n \n \n \n \n \n \n")
            stay = input("Do you want to Stay where you are? 1 for Yes, 0 for No")
            
            # if the player chooses to stay put
            if stay == 1:
                self.turnPhase = "suggestion"
                curr_player.wasMovedInSuggestion == False
                return
            
        # if they weren't moved or they dont want to stay, set the phase to move
        curr_player.wasMovedInSuggestion == False
        self.turnPhase = "move"
        return 

    def suggestion_phase(self, db):
        curr_player = self.players[self.playerTurn]

        # Can't do anything until the move phase is over.
        while self.turnPhase != "suggestion":
            pass

        if curr_player.room.isHallway:
            # the player is in the hallway, move onto the accusation phase
            print("\n \n \n \n \n \n \n \n \n \n \n \n \n \n")
            print("you cant make a suggestion in the hallway")  # (swap this to the UI)
            return
            
        # next we go to the suggestion phase
        suggested_cards         = self.get_suggestion_cards(db, curr_player.room.name)
        
        # see if any players are the selected character
        for i1 in range(self.numPlayers):
            if self.players[i1].characterName == suggested_cards[1].name:
                self.players[i1].move(Room.getInstanceByName(suggested_cards[0].name))
                self.players[i1].wasMovedInSuggestion = True
        
        # verify the cards
        suggested_cards         = curr_player.makeSuggestion(suggested_cards)
        
        answered_card           = self.handle_suggestion(suggested_cards)
        print("\n \n \n \n \n \n \n \n \n \n \n \n \n \n")
        print(answered_card.name)  # (swap this to the UI)
        print("\n \n \n \n \n \n \n \n \n \n \n \n \n \n")
        self.set_next_player_turn()

    def accusation_phase(self, db):
        curr_player = self.players[self.playerTurn]
        print("\n \n \n \n \n \n \n \n \n \n \n \n \n \n")
        answer = input("do you want to make an Accusaiton? 1 for Yes, or 0 for No")
        if answer == "1":
            accusation_cards         = self.get_accusation_cards(db)
            accusation_cards         = curr_player.makeAccusation(accusation_cards)

            is_correct              = self.handle_accusation(accusation_cards)

            # if they got it wrong, game goes on.
            if not is_correct:
                curr_player.isActive = False
                self.set_next_player_turn()
            else:
                self.finished = True
        else:
            self.set_next_player_turn()


    # this method will execute a turn for a player object
    def turn(self, player):
        # do they want to move

        # can they make a suggestion or accusation

        # does someone need to respsond to the suggestion


        # 
        self.playerTurn = (self.playerTurn + 1 ) % self.numPlayers
        

    # this method will add a player to the Game object
    def addPlayer(self, player):
        self.players.append(player)
        self.numPlayers += 1
    # compare the guess with the stored solution
    def checkSolution(guess):
        pass

    def get_player_object_based_on_player_name(self, name):
        for player in self.players:
            if player.name==name:
                return player
            

    def handle_suggestion(self, suggested_cards):
        self.players[self.playerTurn + 1].wasMovedInSuggestion = True
        self.players[self.playerTurn + 1].move(self.players[self.playerTurn].room)
        if(suggested_cards['Murder'] == self.solution[0].name and suggested_cards['Weapon'] == self.solution[2].name):
            print("MADE IT HERE")
        
        self.set_next_player_turn()
        # answered = False
        # # we answer the suggestion clockwise
        # player_answering_id = (self.playerTurn + 1) % self.numPlayers

        # # make sure we haven't already answered this, and also ensure that all other players 
        # # haven't already checked
        # while answered == False & player_answering_id != self.currentPlayerId:

        #     player_answering = self.players[player_answering_id]
            
        #     returned_card = player_answering.answerSuggestion(suggested_cards)
        #     if returned_card != None:
        #         # move on to the next phase
        #         self.turnPhase = "accusation"                
        #         return returned_card
        #     else:
        #         print(' You do not have a card to answer') # add something to the user interface for this!
        #         player_answering_id = (player_answering_id + 1) % self.numPlayers
        #         if player_answering_id == self.playerTurn:
        #             self.turnPhase = "accusation"
        #             return None
        # # move on to the next phase
        # self.turnPhase = "accusation"
        # we didn't get a solution
        return None


    def handle_accusation(self, suggested_cards):

        correct_cards =  [card for card in suggested_cards if card in self.solution]
        if len(correct_cards) == 3:
            print("you got it right!")
            return True
        else:
            print("you got it wrong!")
            self.players[self.playerTurn].isActive = False
            return False
            

    def set_next_player_turn(self):
        # get the next player up!
        current_player = self.playerTurn
        valid_player   = False

        # can only pass the turn to players who are active
        while valid_player == False:
            self.playerTurn  = (self.playerTurn + 1) % self.numPlayers

            # if the next player is active set it!
            if self.players[self.playerTurn].isActive:
                self.currentPlayerId = self.players[self.playerTurn].id
                self.turnPhase = "move"
                valid_player  = True

            # are all other players out (back to start)
            if self.playerTurn == current_player:
                print("You win, all other players are out")
                valid_player == False
                self.finished = True
                break

    def get_suggestion_cards(self, db,suggestion_room_name):
            # for now, get player input (this is annoying with the logging output, just type and hit enter)
            print("\n \n \n \n \n \n \n \n \n \n \n \n \n \n")
            suggestion_person_name  = input("Suggestion: What person  ?")
            print("\n \n \n \n \n \n \n \n \n \n \n \n \n \n")

            suggestion_weapon_name  = input("Suggestion: What weapon  ?")

            # note, after the user has selected the cards, we can use the following logic to get the card id.

                # use the database to get the card IDS from their names
            suggested_room_id       = SQL_get_card_id(db, suggestion_room_name)
            suggested_person_id     = SQL_get_card_id(db, suggestion_person_name)
            suggested_weapon_id     = SQL_get_card_id(db, suggestion_weapon_name)

                # get the card objects  - this bit 
            suggested_room_object   = Card.getInstanceById(suggested_room_id)
            suggested_person_object = Card.getInstanceById(suggested_person_id)
            suggested_weapon_object = Card.getInstanceById(suggested_weapon_id)

            return [suggested_room_object, suggested_person_object,suggested_weapon_object]

    def get_accusation_cards(self, db):
            # for now, get player input (this is annoying with the logging output, just type and hit enter)
            print("\n \n \n \n \n \n \n \n \n \n \n \n \n \n")
            suggestion_room_name    = input("Suggestion: What Room  ?")
            print("\n \n \n \n \n \n \n \n \n \n \n \n \n \n")
            suggestion_person_name  = input("Suggestion: What person  ?")
            print("\n \n \n \n \n \n \n \n \n \n \n \n \n \n")

            suggestion_weapon_name  = input("Suggestion: What weapon  ?")

            # note, after the user has selected the cards, we can use the following logic to get the card id.

                # use the database to get the card IDS from their names
            suggested_room_id       = SQL_get_card_id(db, suggestion_room_name)
            suggested_person_id     = SQL_get_card_id(db, suggestion_person_name)
            suggested_weapon_id     = SQL_get_card_id(db, suggestion_weapon_name)

                # get the card objects  - this bit 
            suggested_room_object   = Card.getInstanceById(suggested_room_id)
            suggested_person_object = Card.getInstanceById(suggested_person_id)
            suggested_weapon_object = Card.getInstanceById(suggested_weapon_id)

            return [suggested_room_object, suggested_person_object,suggested_weapon_object]