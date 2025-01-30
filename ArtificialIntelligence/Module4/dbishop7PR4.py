import streamlit as st
import pandas as pd
import random
import streamlit.components.v1 as components
from typing import List, Tuple, Dict, Callable
from copy import deepcopy

# SETUP FUNCTIONS FOR A STAR TO WORK

# Possible emojis and their corresponding weights
COSTS = { '🌾': 1, '🌲': 3, '⛰': 5, '🐊': 7,  '🌋': 10}
MOVES = [(0,-1), (1,0), (0,1), (-1,0)]

# Extract emojis and their corresponding weights
emojis = list(COSTS.keys())
weights = list(COSTS.values())

# Invert the weights: lower weights become higher and higher weights become lower, done so I can ensure mountains arent spawned too often and grain is most common
inverse_weights = [1 / weight for weight in weights]

# Function to generate an emoji grid
def generate_world(rows, cols):
    """
    Generates the list of lists of emojis based on the amount of rows and columns provided. Selects emojis based on the weights to ensure we don't get a grid of impassable mountains.

    Parameters:
    rows (int): Number of rows requested
    cols (int): Number of columns requested
    """
    return [
        random.choices(emojis, inverse_weights, k=cols) 
        for _ in range(rows)
    ]

def display_emoji_grid(emoji_grid):
    """
    Display a List of Lists of emojis in a perfect grid (table) via html.
    
    Parameters:
    emoji_grid (list of list of str): A 2D list containing emojis to display in a grid.
    """
    # Create HTML table
    html = '<table style="border-collapse: collapse;">'
    
    for row in emoji_grid:
        html += '<tr>'
        for emoji in row:
            html += f'<td style="border: none; padding: 0px; text-align: center; font-size: 2em;">{emoji}</td>'
        html += '</tr>'
    
    html += '</table>'
    
    st.markdown(html, unsafe_allow_html=True)


def get_possible_moves(current_position: Tuple[int, int], world: List[List[str]]):
    """
    Generates all possible moves that can be taken in current state.

    Parameters:
    current_position (Tuple[int, int]): The current position on the grid as (x, y) coordinates.
    world (List[List[str]]): The grid represented as a list of lists of strings, where each string is an emoji representing the terrain.
    """
    possible_moves: List[Tuple[int,int]] = []
    for move in MOVES:
        new_x, new_y = current_position[0] + move[0], current_position[1] + move[1]
        if new_x >= 0 and new_y >= 0 and new_y < len(world) and new_x < len(world[0]):
            possible_moves.append(move)
    return possible_moves

def manhattan_distance(current: Tuple[int, int], goal: Tuple[int, int]):
    """
    Calculates the Manhatten Distance between given location and the goal

    Parameters:
    current (Tuple[int, int]): The current position as (x, y) coordinates.
    goal (Tuple[int, int]): The goal position as (x, y) coordinates.
    """
    return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

def heuristic(current_position: Tuple[int, int], move: Tuple[int, int], goal_position: Tuple[int, int], world: List[List[str]]):
    """
    This function estimates the cost to reach the goal with a selected move on the grid. 
    It considers the cost associated with moving onto different terrains (emojis), as well as the Manhattan distance to the goal. 

    Parameters:
    current_position (Tuple[int, int]): The current position on the grid as (x, y) coordinates.
    goal_position (Tuple[int, int]): The goal position on the grid as (x, y) coordinates.
    world (List[List[str]]): The grid represented as a list of lists of strings, where each string is an emoji representing the terrain.
    """
    new_position = (current_position[0] + move[0], current_position[1] + move[1])

    # Calculate the Manhattan distance from the new position to the goal as a simple heuristic
    manhattan_distance = abs(new_position[0] - goal_position[0]) + abs(new_position[1] - goal_position[1])

    # Add any additional cost considerations based on the terrain at the new position
    terrain_cost = COSTS.get(world[new_position[1]][new_position[0]], float('inf'))

    move_cost: int = manhattan_distance + terrain_cost

    # Return the combined heuristic cost
    return move_cost

def a_star_search( world: List[List[str]], start: Tuple[int, int], goal: Tuple[int, int], costs: Dict[str, int], moves: List[Tuple[int, int]], heuristic: Callable) -> List[Tuple[int, int]]:
    """
    The `a_star_search` function finds the optimal path from a starting point to a goal within a grid. 
    This implementation of A* search evaluates possible paths by balancing the actual cost to reach a node and an estimated cost to reach the goal from that node.
    
    Parameters:
    world [List[str]]: the actual context for the navigation problem.
    start Tuple[int, int]: the starting location of the bot, `(x, y)`.
    goal Tuple[int, int]: the desired goal position for the bot, `(x, y)`.
    costs Dict[str, int]: is a `Dict` of costs for each type of terrain in **world**.
    moves List[Tuple[int, int]]: the legal movement model expressed in offsets in **world**.
    heuristic Callable: is a heuristic function, $h(n)$.
    """
    open_set: List[Tuple[int, Tuple[int, int]]] = [(0, start)]  # List of (f_score, position)
    came_from: Dict[Tuple[int, int], Tuple[int, int] | None] = {start: None}  # Track the path
    came_by_move: Dict[Tuple[int, int], Tuple[int, int] | None] = {}  # Track the move used to get to each position
    g_score: Dict[Tuple[int, int], float] = {start: 0}  # Cost from start to each position
    f_score: Dict[Tuple[int, int], float] = {start: heuristic(start, (0, 0), goal, world)}  # Initial heuristic cost for the start

    while open_set:
        # Sort the open set to find the position with the lowest f_score
        open_set.sort(key=lambda x: x[0])
        current_f, current_position = open_set.pop(0)  # Pop the element with the lowest f_score

        if current_position == goal:
            # Reconstruct path using moves
            move_path: List[Tuple[int, int]] = []
            while current_position != start:
                move = came_by_move[current_position]
                move_path.append(move)
                current_position = (current_position[0] - move[0], current_position[1] - move[1])
            return move_path[::-1]  # Return the move sequence in correct order

        possible_moves = get_possible_moves(current_position, world)

        for move in possible_moves:
            # Calculate the new position from this move
            new_position = (current_position[0] + move[0], current_position[1] + move[1])
            x, y = new_position
            tentative_g_score = g_score[current_position] + costs.get(world[y][x], float('inf'))

            if new_position not in g_score or tentative_g_score < g_score[new_position]:
                came_from[new_position] = current_position
                came_by_move[new_position] = move  # Track the move that led to this position
                g_score[new_position] = tentative_g_score
                f_score[new_position] = tentative_g_score + heuristic(current_position, move, goal, world)
                open_set.append((f_score[new_position], new_position))

    return None  # Return None if no path is found

def pretty_print_path( world: List[List[str]], path: List[Tuple[int, int]], start: Tuple[int, int], goal: Tuple[int, int], costs: Dict[str, int]) -> int:
    """
    The `pretty_print` function takes a 2D grid of terrain types, each represented by an emoji, and visually displays it in a structured format. 
    It  overlays a path, generated by the `a_star_search` function, to show the route from start to goal on the grid. 
    
    Parameters:
    world List[List[str]]: the world (terrain map) for the path to be printed upon.
    path List[Tuple[int, int]]: the path from start to goal, in offsets.
    start Tuple[int, int]: the starting location for the path.
    goal Tuple[int, int]: the goal location for the path.
    costs Dict[str, int]: the costs for each action.
    """
    moves_to_emoji: Dict[Tuple[int, int], str] = {
        (1, 0): '⏩',   # Right
        (-1, 0): '⏪',  # Left
        (0, -1): '⏫',  # Up
        (0, 1): '⏬'    # Down
    }

    # Copy the world grid to modify it as to not mess up grid for other runs
    world_copy: List[List[str]] = [row[:] for row in world]

    total_cost: int = 0

    # Place the emojis along the path and calculate the total cost
    current_position: Tuple[int, int] = start
    for move in path:
        emoji: str = moves_to_emoji.get(move, '❓')  # Get the correct emoji for the move, use ? if one isnt found somehow instead of crashing
        world_copy[current_position[1]][current_position[0]] = emoji  # Place emoji at the current position

        # Calculate the next position and update the total cost
        next_position = (current_position[0] + move[0], current_position[1] + move[1])
        total_cost += costs[world[next_position[1]][next_position[0]]]

        # Update the current position
        current_position = next_position

    world_copy[goal[1]][goal[0]] = '🎁'  # Mark the goal position with a gift emoji

    # Display the modified grid
    display_emoji_grid(world_copy)

    # Return the total cost of the path
    return total_cost

# Display the main title
st.markdown("# A Star Search")

# Display the subtitle
st.markdown("## Edit the World Grid")

# Check if the DataFrame exists in session state, if not, initialize it
if 'df' not in st.session_state:
    rows, cols = 4, 4  # Default values for rows and columns
    world = generate_world(rows, cols)
    st.session_state.df = pd.DataFrame(world)

# Side panel for selecting DataFrame dimensions with a submit button
with st.sidebar.form(key='grid_form'):
    st.markdown("## World Size")
    cols = st.number_input("Select a width", 2, 10, 4) # Left at min of 2 and max of 10, could be up to 30 before things start acting funky but the "site" doesnt look as good
    rows = st.number_input("Select a height", 2, 10, 4)# Left at min of 2 and max of 10, could be up to 30 before things start acting funky but the "site" doesnt look as good
    submit_button = st.form_submit_button(label='Submit')

# Sidebar container for emoji cost key, so whoever is looking at this outside the project knows what the costs are
with st.sidebar.container(border=True):
    st.markdown("## Emoji Cost Key")
    for emoji, cost in COSTS.items():
        st.markdown(f"{emoji} : {cost}")

# Generate the DataFrame when the submit button is clicked
if submit_button:
    world = generate_world(rows, cols)
    st.session_state.df = pd.DataFrame(world)

# Convert the DataFrame columns to categorical with only emoji options, ensures we cant enter non emoji items as well as gives emoji options instead of having to be copied from somewhere
for col in st.session_state.df.columns:
    st.session_state.df[col] = pd.Categorical(st.session_state.df[col], categories=emojis)

# Allow the user to edit the DataFrame with emojis using st.data_editor
edited_df = st.data_editor(st.session_state.df, key="emoji_editor")

if st.button('Solve') and not st.session_state.df.empty:
    path = a_star_search(edited_df.to_numpy(), (0, 0), (cols-1, rows-1), COSTS, MOVES, heuristic)
    pretty_print_path(edited_df.to_numpy(), path, (0, 0), (cols-1, rows-1), COSTS)
else:
    display_emoji_grid(edited_df.to_numpy())

