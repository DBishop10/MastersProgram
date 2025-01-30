from qlearningSARSA import *
from valueiteration import *
from trackandcar import *
import random

def demo_value_iteration_sweep(track_file):
    print("Demonstrating Value Iteration Sweep")
    vi = ValueIteration(track_file, theta=1, debug=True)
    vi.run_value_iteration() 

def demo_qlearning_updates(track_file):
    print("Demonstrating Q-learning Updates")
    agent = RLAgent(track_file, debug=True)
    agent.run_qlearning(episodes=1)  # Only run for one episode to see updates on one sequence

def demo_sarsa_updates(track_file):
    print("Demonstrating SARSA Updates")
    agent = RLAgent(track_file, debug=True)
    agent.run_sarsa(episodes=1)  # Only run for one episode to see updates on one sequence

def demo_exploration(track_file):
    print("Demonstrating Exploration Function")
    agent = RLAgent(track_file, debug=True, epsilon=.5) #increased epsilon for greedy and random
    for _ in range(10):  # Show several choices to observe both random and greedy selections
        state = (random.randint(0, agent.racetrack.size[0] - 1), 
                random.randint(0, agent.racetrack.size[1] - 1), 
                random.randint(-5, 5), 
                random.randint(-5, 5))
        agent.choose_action(*state)

def demo_path_generation(track_file, agent_type="Q-learning"):
    print(f"Demonstrating Path Generation for {agent_type}")
    if agent_type == "Q-learning":
        agent = RLAgent(track_file)
        agent.run_qlearning(episodes=10000)
    elif agent_type == "SARSA":
        agent = RLAgent(track_file)
        agent.run_sarsa(episodes=10000)
    else:
        agent = ValueIteration(track_file)
        agent.run_value_iteration()

    agent.test_policy()

def demo_restart_behavior(track_file):
    print("Demonstrating Restart Behavior: Restart")
    track = Racetrack(track_file, crash="restart")
    actions = [(0, 1)] * 12 # Yeet itself into a wall
    for ax, ay in actions:
        print(f"Car Local: {track.car.x}, {track.car.y}")
        result = track.step(ax, ay)
        if result == "crash":
            print(f"Collision detected, new position: ({track.car.x}, {track.car.y})")
            break
    input()

    print("Demonstrating Restart Behavior: Nearest")
    track = Racetrack(track_file, crash="nearest")
    actions = [(0, 1)] * 12 # Yeet itself into a wall
    for ax, ay in actions:
        print(f"Car Local: {track.car.x}, {track.car.y}")
        result = track.step(ax, ay)
        if result == "crash":
            print(f"Collision detected, new position: ({track.car.x}, {track.car.y})") 
            break
    input()

    print("Demonstrating Path Crash Detection")
    track = Racetrack(track_file, crash="nearest", debug=True)
    actions = [(0, 1)] * 12 # Yeet itself into a wall
    for ax, ay in actions:
        print(f"Car Local: {track.car.x}, {track.car.y}")
        result = track.step(ax, ay)
        if result == "crash":
            print(f"Collision detected, new position: ({track.car.x}, {track.car.y})") 
            break

if __name__ == "__main__":
    track_file = '../Data/track/L-track.txt'

    demo_value_iteration_sweep(track_file)
    input() #Inputs added so I could seperate the functions easier
    
    demo_qlearning_updates(track_file)
    input()

    demo_sarsa_updates(track_file)
    input()

    demo_exploration(track_file)
    input()

    demo_path_generation(track_file, agent_type="Q-learning")
    input()

    demo_path_generation(track_file, agent_type="SARSA")
    input()

    demo_path_generation(track_file, agent_type="Value Iteration")
    input()

    demo_restart_behavior(track_file)