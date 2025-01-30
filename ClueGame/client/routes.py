# This routes function follows the tutorial: https://www.youtube.com/watch?v=AMp6hlA8xKA
# it initializes the website according to the template, game.html

from flask import Blueprint, render_template, jsonify, session
from backend.Player import Player
import uuid

main = Blueprint("main", __name__)

@main.route("/")
def entry():
    session["id"] = uuid.uuid4()
    return render_template("entry.html")

@main.route("/home", methods=["GET"])
def home():
    return render_template("home.html")

@main.route("/board", methods=["GET", "POST"])
def board():
    return render_template("board.html", instances = Player.instances_database)

@main.route("/move", methods=["GET"])
def move():
    data = {}
    playerIds = []
    for player in Player.instances_database:
        data[str(player.id)] = {"character": player.characterName, "htmlLocal": player.gethtmlLocal(player.room, player.wasMovedInSuggestion)}
        playerIds.append(player.id)
    return jsonify(data, playerIds)