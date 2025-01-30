# 10/14/2023
# This script holds all the flask events that we will use
# to update the webpage

# Chuan Lin: I am sorry that I don't know how to get the game_global instance created in run.py
# from run import game_global
# from backend.Character import mapping_from_character_to_locations

from flask import request, session
from flask_socketio import emit
from backend.Board import *
from backend.Player import *
from backend.Character import *
from backend.Game import Game

# we will add all the events onto this object, then import
# this in init
from .extensions import socketio

# This will base the events on database
from .sql import *
import pymysql
from client import dbAccount

db = pymysql.connect(host='localhost', port=3306, user="root", password="password", db='NotAClue', charset='utf8')
board = Gameboard()
game = Game()
characterChoices = {}
characterList = ["Green", "Peacock", "Scarlet", "Plum", "White", "Mustard"]

selected = {
    'selectedChar': "",
    'userName': ""
}


# This connection function is from the tutorial: https://www.youtube.com/watch?v=AMp6hlA8xKA
@socketio.on("connect")
def handle_connect():
     print("client connected in sql eventss")

# This adds a player to the player selection view
# note that the game.html enforces that the user name is nonempty
@socketio.on("user_join")
def handle_user_join(username):
    selected["userName"] = username
    SQL_handle_user_join(db, session["id"], username)
    emit("player_joined", broadcast=True)

    # want to add the player here!
    player_id=session["id"]
    player_object=Player.getInstanceById(player_id)
    
    if(player_object == None):
        player_object = Player(player_id, None, None, username, None, None, None)

    game.addPlayer(player_object)

    print(len(game.players))
    # if a new player joins, we need to refresh their screen with all the users that have
    # previously made selections
    for character, username in SQL_get_characters_and_usernames(db):
        emit("playerChoice", {"player": character, "username": username}, broadcast=True)

# Notification for suggestion and accusation
@socketio.on("notification")
def toast_notify(data):
    suggestion, cards = data
    cards[0] = game.solution[0].name
    cards[1] = game.solution[2].name
    cards[2] = game.solution[1].name
    if(suggestion['Type'] == "Next Turn"):
        game.set_next_player_turn()
    else:
        game.handle_suggestion(suggestion)
        socketio.emit("notifying", [suggestion, cards])


@socketio.on("addCharacterChoices")
def add(character):
    username = SQL_get_username_based_on_sid(db, session["id"])
    if(username in characterChoices.keys() or character in characterChoices.values()):
        raise ValueError("You have already selected a character!!")
    else:
        selected['selectedChar'] = character
        characterChoices[username] = character
        emit("characterAdd", {"usr": username, "chr": character})

@socketio.on("removeCharacterChoices")
def remove(character):
    username = SQL_get_username_based_on_sid(db, session["id"])
    if (username in characterChoices.keys()):
        if(character != characterChoices[username]):
            raise ValueError("This character is already taken up by other users!!")
        else:
            selected['selectedChar'] = ""
            emit("characterRemove", {"usr": username, "chr": character})
            del characterChoices[username]
    else:
        raise ValueError("You cannot unselect other players characters!!")
        
@socketio.on("updateCharacterChoices")
def update():
    emit("characterUpdate", {"choices": characterChoices, "characters": characterList})
    
# This handles the event where a player selectss a character to use
@socketio.on("player_select")
def handle_player_select(character):
    # "playerName" is synonymous with "character"
    # find the username, this is a bad way to go about it
    print("PLAYERSELECT: ", session["id"])
    
    username = SQL_get_username_based_on_sid(db, session["id"])

    characters_and_usernames = SQL_get_characters_and_usernames(db)
    characters = [i[0] for i in characters_and_usernames]
    usernames = [i[1] for i in characters_and_usernames]

    # check to make sure another player hasnt chosen this character
    if character in characters:
        return

    # if a player is switching their character
    if username in usernames:
        character_old = SQL_get_character_based_on_username(db, username)
        SQL_delete_character_based_on_charactername(db, character_old)
        emit("playerChoice", {"player": character_old, "username": ""}, broadcast=True)

    SQL_handle_player_select(db, character, session["id"])
    emit("playerChoice", {"player": character, "username": username}, broadcast=True)

    # set the characterId field and position field for the player object
    # note that we assume that <<<<<<<request.id==player.id>>>>>>>>
    player_id=session["id"]
    player_object=Player.getInstanceById(player_id)

    character_object=Character(None,character,"icon")
    
    roomName=character_object.location
    room_object=Room.getInstanceByName(roomName)

    if(player_object == None):
        player_object = Player(player_id, character_object.id, character, username, room_object.id, room_object, None)

    else:
        player_object.characterId=character_object.id
        player_object.characterName= character
        player_object.roomId=room_object.id
        player_object.room = room_object

@socketio.on("has_game_started")
def did_game_start():
    if game.started:
       emit("click_button")

# This function starts the game!
@socketio.on("game_start")
def handle_game_start():   
    emit("start_game", broadcast=True)
   
   # if a game has started already, dont rerun everything,  THIS IS VERY BAD IF IT HAPPENS
    if game.started:
        return
    # initialize the game object
    game.initializeGame()
    print( "\n \n \n \n \n \n \n \n \n \n")
    print(game.solution[0].name, game.solution[2].name)
    print( "\n \n \n \n \n \n \n \n \n \n")
    # add the deck to the database. 
    # TODO: connect game position to gameboard position to the room card
    for i1 in range(len(game.cards.cards)):
        temp_card = game.cards.cards[i1]
        SQL_add_card(db, temp_card.id, temp_card.name, temp_card.type)


# =======================================================================================================================
       
    while game.finished == False:
       # Here is the game logic. This piece needs to be integrated with the front end, rather than just occuring here!!!

        # first, we get the move from the player whose turn it is
        # TODO, if the player was moved last time by a suggestion 
        game.move_phase() 
        
        # next we go to the suggestion phase
        game.suggestion_phase(db)

        # accustaion phase
        game.accusation_phase(db)


# =======================================================================================================================



# Determines where the user clicked and returns the room
@socketio.on("room_select")
def room_selected(x, y):    
    print(x,y)
    location = board.determine_html_location(x, y)
    print(location)
    if location == None:
        ValueError("Choose a room, hallway, or secret passage please")
    else:
        handle_player_room_choose(location)


@socketio.on("character")
def handle_character(character):
    emit("character", character, broadcast=True)


# Handles when a player chooses a room
# TODO: Get current player and make function def handle_player_room_choose(player: Player, newRoomId)
def handle_player_room_choose(room: str):
    player = Player.getInstanceById(session["id"])
    currRoom = board.get_room_by_id(player.room.id)
    newRoom = board.get_room_by_name(room)
    if(currRoom.name in board.rooms):
        newRoomChoices = board.layout[currRoom]
    elif (currRoom.name in board.hallways):
        newRoomChoices = board.hallLayout[currRoom]
    
    # first check if it is the current player's turn
    if(player.id != game.currentPlayerId):        
        print("It's not your turn")

    # next check if it is the move phase of the turn
    elif(game.turnPhase != "move"):
        print("The move phase is over")

    elif(newRoom not in newRoomChoices):
        raise ValueError("You must choose a location that is adjacent to you (unless you can take a secret passage)")
    else:
        player.move(newRoom)
        game.turnPhase = "suggestion"   # this player's move is over



@socketio.on("connect")
def return_check():
    socketio.emit("start_with_player", selected)




    