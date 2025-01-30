from trackandcar import *
import numpy as np
import random

class RLAgent:
    """
    RLAgent implements reinforcement learning algorithms (Q-learning and SARSA) 
    for a race car to navigate a racetrack.

    Attributes:
        racetrack (Racetrack): The racetrack object that the agent will interact with.
        crash (str): The crash handling strategy ('nearest' or 'restart').
        alpha (float): The learning rate for Q-learning and SARSA.
        gamma (float): The discount factor for future rewards.
        epsilon (float): The exploration rate for the epsilon-greedy policy.
        Q (ndarray): The Q-table storing state-action values.
        actions (list): A list of possible actions (ax, ay) for the car.
    """
    def __init__(self, track_file, alpha=0.1, gamma=0.95, epsilon=0.1, crash='nearest', debug = False, model="Q-learning"):
        """
        Initializes the RLAgent class with the specified racetrack and parameters.

        Parameters:
            track_file (str): The path to the racetrack file.
            alpha (float): The learning rate for Q-learning and SARSA. Default is 0.1.
            gamma (float): The discount factor for future rewards. Default is 0.95.
            epsilon (float): The exploration rate for the epsilon-greedy policy. Default is 0.1.
            crash (str): The crash handling strategy ('nearest' or 'restart'). Default is 'nearest'.
            debug (bool): Determines whether or not to print debug items, mostly utilized for demo video
            model (str): States whether it is a Q-learning model or a SARSA
        """
        self.racetrack = Racetrack(track_file, crash=crash)
        self.crash = crash
        self.alpha = alpha 
        self.gamma = gamma 
        self.debug = debug
        self.epsilon = epsilon 
        self.model = model
        self.Q = np.zeros((self.racetrack.size[0], self.racetrack.size[1], 11, 11, 9))
        self.actions = [(ax, ay) for ax in [-1, 0, 1] for ay in [-1, 0, 1]]

    def choose_action(self, x, y, vx, vy, greedy=False):
        """
        Chooses an action based on the current state using an epsilon-greedy policy.

        Parameters:
            x (int): The current x-position of the car.
            y (int): The current y-position of the car.
            vx (int): The current velocity in the x-direction.
            vy (int): The current velocity in the y-direction.
            greedy (bool): If True, chooses the action with the highest Q-value (exploitation). 
                           If False, chooses an action randomly with a probability of epsilon (exploration). Default is False.

        Returns:
            tuple: The chosen action (ax, ay).
        """
        if not greedy and random.random() < self.epsilon: # If greedy is false it goes in (this kept confusing me when I would look at it and scare me, hence this comment)
            action = random.choice(self.actions) 
            if self.debug:
                print(f"Random action chosen: {action}")
            return action  
        else:
            q_values = self.Q[x][y][vx + 5][vy + 5]
            action = self.actions[np.argmax(q_values)]
            if self.debug:
                print(f"Greedy action chosen: {action}")
            return action

    def update_qlearning(self, state, action, reward, next_state):
        """
        Updates the Q-table using the Q-learning algorithm.

        Parameters:
            state (tuple): The current state of the car (x, y, vx, vy).
            action (tuple): The action taken (ax, ay).
            reward (float): The reward received after taking the action.
            next_state (tuple): The next state of the car (x, y, vx, vy).
        """
        x, y, vx, vy = state
        next_x, next_y, next_vx, next_vy = next_state
        action_index = self.actions.index(action)

        best_next_q = np.max(self.Q[next_x][next_y][next_vx + 5][next_vy + 5])
        if self.debug:
            print(f"Q before update for state {state}, action {action}: {self.Q[x][y][vx + 5][vy + 5][action_index]}")
        self.Q[x][y][vx + 5][vy + 5][action_index] += self.alpha * (
            reward + self.gamma * best_next_q - self.Q[x][y][vx + 5][vy + 5][action_index]
        )
        if self.debug:
            print(f"Q after update for state {state}, action {action}: {self.Q[x][y][vx + 5][vy + 5][action_index]}")

    def update_sarsa(self, state, action, reward, next_state, next_action):
        """
        Updates the Q-table using the SARSA algorithm.

        Parameters:
            state (tuple): The current state of the car (x, y, vx, vy).
            action (tuple): The action taken (ax, ay).
            reward (float): The reward received after taking the action.
            next_state (tuple): The next state of the car (x, y, vx, vy).
            next_action (tuple): The next action taken (ax, ay).
        """
        x, y, vx, vy = state
        next_x, next_y, next_vx, next_vy = next_state
        action_index = self.actions.index(action)
        next_action_index = self.actions.index(next_action)

        if self.debug:
            print(f"Q before update for state {state}, action {action}: {self.Q[x][y][vx + 5][vy + 5][action_index]}")
        self.Q[x][y][vx + 5][vy + 5][action_index] += self.alpha * (
            reward + self.gamma * self.Q[next_x][next_y][next_vx + 5][next_vy + 5][next_action_index] - self.Q[x][y][vx + 5][vy + 5][action_index]
        )
        if self.debug:
            print(f"Q after update for state {state}, action {action}: {self.Q[x][y][vx + 5][vy + 5][action_index]}")
            

    def reward(self, crashed=False, state=None, visited_states=None, ax=0, ay=0):
        """
        Calculates the reward for the current state-action pair.

        Parameters:
            crashed (bool): Indicates if the car has crashed. Default is False.
            state (tuple): The current state of the car (x, y, vx, vy).
            visited_states (dict): A dictionary tracking how many times each state has been visited.
            ax (int): The acceleration in the x-direction.
            ay (int): The acceleration in the y-direction.

        Returns:
            int: The calculated reward value.
        """
        values = 0
        if crashed:
            if(self.crash == "restart" and self.model != "SARSA"): # Remove for SARSA, couldnt train with this due to immense speed decline
                values += -1000 
            values += -100  # Strong penalty for crashing
        if ax == 0 and ay == 0:
            values += -100
        if state[:2] in self.racetrack.start_positions:
            values += -10  # Much stronger penalty for staying on the starting line
        if visited_states and visited_states[state] > 1:
            return -10  # Penalty for revisiting the same state
        if state[:2] in self.racetrack.finish_positions:
            values += 100  # Strong reward for reaching the finish
        values += -10  # Small negative reward for any valid move
        return values

    def run_qlearning(self, episodes=1000):
        """
        Trains the agent using the Q-learning algorithm over a specified number of episodes.

        Parameters:
            episodes (int): The number of episodes to run for training. Default is 1000.
        """
        for episode in range(episodes):
            self.racetrack.car.x, self.racetrack.car.y = random.choice(self.racetrack.start_positions)
            self.racetrack.car.vx, self.racetrack.car.vy = 0, 0
            visited_states = {}

            while (self.racetrack.car.x, self.racetrack.car.y) not in self.racetrack.finish_positions:
                state = (self.racetrack.car.x, self.racetrack.car.y, self.racetrack.car.vx, self.racetrack.car.vy)
                visited_states[state] = visited_states.get(state, 0) + 1
                action = self.choose_action(*state)
                result = self.racetrack.step(*action)
                crashed = result == "crash"
                next_state = (self.racetrack.car.x, self.racetrack.car.y, self.racetrack.car.vx, self.racetrack.car.vy)
                reward = self.reward(crashed, state, visited_states, action[0], action[1])
                self.update_qlearning(state, action, reward, next_state)

                if result == "finish":
                    break 
                
        print("Q-learning training complete.")

    def run_sarsa(self, episodes=1000):
        """
        Trains the agent using the SARSA algorithm over a specified number of episodes.

        Parameters:
            episodes (int): The number of episodes to run for training. Default is 1000.
        """
        for episode in range(episodes):
            self.racetrack.car.x, self.racetrack.car.y = random.choice(self.racetrack.start_positions)
            self.racetrack.car.vx, self.racetrack.car.vy = 0, 0
            visited_states = {}  

            state = (self.racetrack.car.x, self.racetrack.car.y, self.racetrack.car.vx, self.racetrack.car.vy)
            action = self.choose_action(*state)
            steps = 0
            while (self.racetrack.car.x, self.racetrack.car.y) not in self.racetrack.finish_positions:
                visited_states[state] = visited_states.get(state, 0) + 1  
                result = self.racetrack.step(*action)
                crashed = result == "crash"
                next_state = (self.racetrack.car.x, self.racetrack.car.y, self.racetrack.car.vx, self.racetrack.car.vy)
                reward = self.reward(crashed, state, visited_states, action[0], action[1])
                next_action = self.choose_action(*next_state)

                self.update_sarsa(state, action, reward, next_state, next_action)

                state = next_state
                action = next_action
                steps += 1

                if result == "finish":
                    break 

        print("SARSA training complete.")

    def test_policy(self, greedy=True):
        """
        Tests the policy learned by the agent on the racetrack.

        Parameters:
            greedy (bool): If True, uses the best learned policy without exploration (greedy). 
                           Default is True.
        
        Returns:
            int: The number of steps taken to reach the finish line.
        """
        self.racetrack.car.x, self.racetrack.car.y = random.choice(self.racetrack.start_positions)
        self.racetrack.car.vx, self.racetrack.car.vy = 0, 0

        print(f"Testing policy: Start Position: ({self.racetrack.car.x}, {self.racetrack.car.y})")
        steps = 0
        while (self.racetrack.car.x, self.racetrack.car.y) not in self.racetrack.finish_positions:
            steps += 1
            state = (self.racetrack.car.x, self.racetrack.car.y, self.racetrack.car.vx, self.racetrack.car.vy)
            action = self.choose_action(*state, greedy=greedy)
            if self.debug:
                print(f"Step {steps}: State: {state}, Action: {action}")
            self.racetrack.step(*action)
            print(f"Step {steps}: Position: ({self.racetrack.car.x}, {self.racetrack.car.y}), Velocity: ({self.racetrack.car.vx}, {self.racetrack.car.vy}), Action: ({action[0]}, {action[1]})")

        print(f"Reached Finish Line at: ({self.racetrack.car.x}, {self.racetrack.car.y}) after {steps} steps.\n")
        return steps


if __name__ == "__main__":

    track_file = '../Data/track/R-track.txt'
    agent = RLAgent(track_file, crash='restart')

    print("Running Q-learning...")
    agent.run_qlearning(episodes=10000)

    print("Testing Q-learning policy...")
    agent.test_policy()

    print("Running SARSA...")
    agent.run_sarsa(episodes=10000)

    print("Testing SARSA policy...")
    agent.test_policy()
