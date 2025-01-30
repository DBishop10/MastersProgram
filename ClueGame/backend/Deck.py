# Connor Finn
# 11/11/2023
# This class defines a deck object for the game clue. 
#   It is initialized with all the cards in clue
#   It has a shuffle method that randomizes the order of the cards
#   It has a deal method that returns the three solution cards, and the hands 
#   designated to the players
from backend.Card import Card
import random

class Deck:
  
  def __init__(self):
    self.cards = []

    self.buildDeck()

    self.cardIds  = list(range(0, 21))    # build a list from 0 to 20

  # the buildDeck function will only be called during init, and it sets
  # the cards attribute to have one of each of the Card objects for the game clue
  def buildDeck(self):
    self.cards = [Card( 0, "person", "MrGreen"),       # people
                  Card( 1, "person", "ProfPlum"),
                  Card( 2, "person", "MrsPeacock"),
                  Card( 3, "person", "MissScarlet"),
                  Card( 4, "person", "MrsWhite"),
                  Card( 5, "person", "ColMustard"),
                  Card( 6, "room", "Study"),             # rooms
                  Card( 7, "room", "Hall"),
                  Card( 8, "room", "Lounge"),
                  Card( 9, "room", "Library"),
                  Card( 10, "room", "Billiard Room"),
                  Card( 11, "room", "Dining Room"),
                  Card( 12, "room", "Conservatory"),
                  Card( 13, "room", "Ballroom"),
                  Card( 14, "room", "Kitchen"),  
                  Card( 15, "weapon", "candlestick"),               # weapons       
                  Card( 16, "weapon", "revolver"),       
                  Card( 17, "weapon", "knife"),       
                  Card( 18, "weapon", "lead pipe"),       
                  Card( 19, "weapon", "rope"),       
                  Card( 20, "weapon", "wrench")]    

  # the Deal method will deal out the cards, it will build a dictionary 
  # where the keys are 
  #    'solution'
  #     and an id for each id in playerIds
  def Deal(self, playerIds):
    
    # initialize the dictionary to be returned
    dealtCards = {}

    # first get the solution cards
    solution_ids = [random.randint(0, 5), random.randint(6, 14), random.randint(15, 20)]
    solution_cards = list( [self.cards[i] for i in solution_ids] )
    
    # add the solution to the dealtCards dictionary
    dealtCards["solution"] = solution_cards

    # next deal out the cards to all the players we will do this the dirty way
    numPlayers = len(playerIds)
    for pid in playerIds:
        dealtCards[pid]  = []

    # what cards remain, and shuffle the order
    remaining_cards = [x for x in self.cardIds if x not in solution_ids]
    random.shuffle(remaining_cards)

    counter = 0
    for cardId in remaining_cards: 
        dealtCards[playerIds[counter]].append(self.cards[cardId])
        counter = (counter +1 ) % numPlayers
    
    # return the solution
    return dealtCards
  


