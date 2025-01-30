# This run function follows the tutorial: https://www.youtube.com/watch?v=AMp6hlA8xKA

# first import our game from the clueless package

# from clueLess import create_app, socketio

from client import create_app, socketio
from backend.Deck import Deck
from backend.Card import Card
from backend.Game import Game

# from client.dbAccount import usr, pwd

# This will base the events on database
from client.sql import *
import pymysql
# first connect to the database
db = pymysql.connect(host='localhost', port=3306, user="root", password="password", db='NotAClue', charset='utf8')

SQL_refresh_database(db)

# then create the app
app = create_app()



# run the game!
try:
    socketio.run(app, host=('0.0.0.0'), log_output=False, debug=False)
except:
    socketio.run(app, host=('0.0.0.0'), allow_unsafe_werkzeug=True, log_output=False, debug=False)
