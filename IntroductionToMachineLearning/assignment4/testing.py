from qlearningSARSA import *
from valueiteration import *
from trackandcar import *

def log_to_file(log_message, filename="tests.txt"):
    """
    Appends a log message to a specified log file.

    Parameters:
        log_message (str): The message to log.
        filename (str): The name of the log file. Default is "tests.txt".
    """
    with open(filename, "a") as log_file:
        log_file.write(log_message + "\n")

def test_on_track(agent_type="Q-learning", track_file=None, episodes=10000, runs=10, log_file="tests.txt", crash="nearest"):
    """
    Tests a specified agent type on a given racetrack over multiple runs.

    Parameters:
        agent_type (str): The type of agent to test ("Q-learning", "SARSA", "Value Iteration"). Default is "Q-learning".
        track_file (str): The path to the racetrack file.
        episodes (int): The number of episodes to run during training. Default is 10,000.
        runs (int): The number of test runs to perform. Default is 10.
        log_file (str): The name of the log file to record test results. Default is "tests.txt".
        crash (str): The crash handling strategy ('nearest' or 'restart'). Default is "nearest".

    Returns:
        float: The average number of steps taken to complete the track across all runs.
    """
    total_steps = 0
        
    for run in range(runs):
        print(f"Run {run + 1}/{runs} of {agent_type} on {track_file}...")
        
        if agent_type == "Q-learning":
            agent = RLAgent(track_file, crash=crash, model=agent_type)
            agent.run_qlearning(episodes=episodes)
        elif agent_type == "SARSA":
            agent = RLAgent(track_file, crash=crash, model=agent_type)
            agent.run_sarsa(episodes=episodes)
        elif agent_type == "Value Iteration":
            agent = ValueIteration(track_file, crash=crash)
            agent.run_value_iteration()

        print(f"Testing {agent_type} policy on {track_file}...")
        steps = agent.test_policy(greedy=(agent_type != "Value Iteration"))
        print(f"Run {run + 1} steps: {steps}")
        total_steps += steps
        
        # Log each run's steps
        log_to_file(f"Run {run + 1}: {steps} steps", log_file)

    average_steps = total_steps / runs
    print(f"Average steps for {agent_type} on {track_file}: {average_steps}\n")
    
    # Log average steps
    log_to_file(f"Average steps: {average_steps}\n", log_file)
    
    return average_steps


def test_r_track(agent_type, track_file, episodes=10000, runs=10, log_file="tests.txt"):
    """
    Tests an agent on the R-track using two different crash conditions: 
    1) Nearest point to the crash 
    2) Return to the start line closest to the crash

    Parameters:
        agent_type (str): The type of agent to test ("Q-learning", "SARSA", "Value Iteration").
        track_file (str): The path to the racetrack file.
        episodes (int): The number of episodes to run during training. Default is 10,000.
        runs (int): The number of test runs to perform. Default is 10.
        log_file (str): The name of the log file to record test results. Default is "tests.txt".

    Returns:
        tuple: A tuple containing the average steps for the nearest crash condition and the restart crash condition.
    """
    # First crash condition: Nearest point to the crash
    print(f"\nTesting {agent_type} on {track_file} with crash condition: Nearest point to the crash...")
    log_to_file(f"\nTesting {agent_type} on {track_file} with crash condition: Nearest point to the crash", log_file)
    average_steps_nearest = test_on_track(agent_type=agent_type, track_file=track_file, episodes=episodes, runs=runs, log_file=log_file, crash="nearest")

    # Second crash condition: Return to the start line closest to the crash
    print(f"\nTesting {agent_type} on {track_file} with crash condition: Return to the start point...")
    log_to_file(f"\nTesting {agent_type} on {track_file} with crash condition: Return to the start point", log_file)
    average_steps_restart = test_on_track(agent_type=agent_type, track_file=track_file, episodes=episodes, runs=runs, log_file=log_file, crash="restart")

    return average_steps_nearest, average_steps_restart


if __name__ == "__main__":
    tracks = ['L-track.txt', 'W-track.txt', 'O-track.txt', 'R-track.txt']
    agent_types = ["Q-learning", "SARSA", "Value Iteration"]
    
    for track in tracks:
        track_file = f'../Data/track/{track}'
        for agent_type in agent_types:
            if agent_type == "Value Iteration":
                if track == "R-track.txt":
                    test_r_track(agent_type=agent_type, track_file=track_file)
                else:
                    log_to_file(f"Testing {agent_type} on {track_file}", "tests.txt")
                    test_on_track(agent_type=agent_type, track_file=track_file)
            else:
                if track == "R-track.txt":
                    test_r_track(agent_type=agent_type, track_file=track_file, episodes=10000)
                else:
                    log_to_file(f"Testing {agent_type} on {track_file}", "tests.txt")
                    test_on_track(agent_type=agent_type, track_file=track_file, episodes=10000)
