from trackandcar import *
import numpy as np

class ValueIteration:
    """
    ValueIteration implements the value iteration algorithm for a car on a racetrack.

    Parameters:
        track_file (str): The path to the racetrack file.
        gamma (float): The discount factor for future rewards.
        theta (float): The threshold for the change in value function to determine convergence.
        crash (str): The crash handling strategy ('nearest' or 'restart').
    """
    def __init__(self, track_file, gamma=0.8, theta=0.001, crash='nearest', debug = False):
        """
        Initializes the ValueIteration class with the specified racetrack and parameters.

        Parameters:
            track_file (str): The path to the racetrack file.
            gamma (float): The discount factor for future rewards. Default is 0.8.
            theta (float): The threshold for the change in value function to determine convergence. Default is 0.001.
            crash (str): The crash handling strategy ('nearest' or 'restart'). Default is 'nearest'.
            debug (bool): Determines whether or not to print debug items, mostly utilized for demo video
        """
        self.racetrack = Racetrack(track_file, crash=crash)
        self.crash = crash
        self.gamma = gamma
        self.theta = theta
        self.debug = debug
        self.start_position = (0,0)
        self.V = np.zeros((self.racetrack.size[0], self.racetrack.size[1], 11, 11))
        self.policy = np.zeros((self.racetrack.size[0], self.racetrack.size[1], 11, 11, 2), dtype=int)
        self.visit_count = np.zeros((self.racetrack.size[0], self.racetrack.size[1], 11, 11))

    def reward(self, crashed=False, state=None, previous_state=None, ax=0, ay=0):
        """
        Computes the reward for a given state and action.

        Parameters:
            crashed (bool): Whether the car has crashed.
            state (tuple): The current state of the car as (x, y, vx, vy).
            previous_state (tuple): The previous state of the car as (x, y, vx, vy).
            ax (int): The acceleration in the x direction.
            ay (int): The acceleration in the y direction.

        Returns:
            int: The computed reward.
        """
        values = 0
        if crashed:
            if self.crash == "restart":
                values += -1000 #Major penalty for crashing during a restart crash mode
            else:
                values += -100  # Strong penalty for crashing
        if ax == 0 and ay == 0:
            values += -100
        if state[:2] in self.racetrack.start_positions:
            values += -10  # Much stronger penalty for staying on the starting line, VI didnt want to leave ever, just got stuck at start
        if self.visit_count[state] > 1:
            values += -10  # Heavier penalty for revisiting the same position
        if state[:2] in self.racetrack.finish_positions:
            values += 100  # Strong reward for reaching the finish
        values += -10  # Small negative reward for any valid move, ensures it tries to find end instead of going up and down like on R track
        return values

    def simulate_transition(self, x, y, vx, vy, ax, ay):
        """
        Simulates the car's transition from one state to another.

        Parameters:
            x (int): The current x-coordinate of the car.
            y (int): The current y-coordinate of the car.
            vx (int): The current velocity in the x direction.
            vy (int): The current velocity in the y direction.
            ax (int): The acceleration in the x direction.
            ay (int): The acceleration in the y direction.

        Returns:
            tuple: A tuple containing the new state (new_x, new_y, new_vx, new_vy) and a boolean indicating if the car crashed.
        """
        new_vx = min(max(vx + ax, -5), 5)
        new_vy = min(max(vy + ay, -5), 5)

        new_x = x + new_vx
        new_y = y + new_vy

        path = bresenham(x, y, new_x, new_y)
        crashed = False
        for px, py in path:
            if is_off_track(px, py, self.racetrack.grid):
                crashed = True
                new_x, new_y = x, y  # Reset to original position
                new_vx, new_vy = 0, 0  # Reset velocity
                break

        # Penalize if velocity is zero after a crash or stopping, would sometimes keep attempting to jump over O-tracks wall, this stopped it
        if (new_vx == 0 and new_vy == 0) and crashed:
            return new_x, new_y, new_vx, new_vy, crashed

        return new_x, new_y, new_vx, new_vy, crashed
        
    def run_value_iteration(self):
        """
        Runs the value iteration algorithm to compute the optimal policy.
        """
        delta = float('inf')
        actions = [(ax, ay) for ax in [-1, 0, 1] for ay in [-1, 0, 1]]
        iterations = 0

        while delta > self.theta:
            delta = 0
            if self.debug:
                print(f"Iteration {iterations}: Starting V values")
                print(self.V[5], '\n')
                input()
            for x in range(self.racetrack.size[0]):
                for y in range(self.racetrack.size[1]):
                    for vx in range(-5, 6):
                        for vy in range(-5, 6):
                            if is_off_track(x, y, self.racetrack.grid):
                                continue

                            old_value = self.V[x][y][vx + 5][vy + 5]
                            best_value = float('-inf')

                            for ax, ay in actions:
                                previous_state = (x, y, vx, vy)
                                new_x, new_y, new_vx, new_vy, crashed = self.simulate_transition(x, y, vx, vy, ax, ay)
                                state = (new_x, new_y, new_vx, new_vy)
                                
                                # Increment the visit count for the current state
                                self.visit_count[x][y][vx + 5][vy + 5] += 1

                                value = self.reward(crashed, state, previous_state, ax, ay) + self.gamma * self.V[new_x][new_y][new_vx + 5][new_vy + 5]

                                if value > best_value:
                                    best_value = value
                                    self.policy[x][y][vx + 5][vy + 5] = [ax, ay]

                            self.V[x][y][vx + 5][vy + 5] = best_value
                            delta = max(delta, abs(old_value - best_value))

            iterations += 1
            if self.debug:
                print(f"Iteration {iterations}: Updated V values")
                print(self.V[5])
                break

    def get_optimal_policy(self, x, y, vx, vy):
        """
        Retrieves the optimal policy (action) for a given state.

        Parameters:
            x (int): The x-coordinate of the car.
            y (int): The y-coordinate of the car.
            vx (int): The velocity in the x direction.
            vy (int): The velocity in the y direction.

        Returns:
            tuple: The optimal action (ax, ay) for the given state.
        """
        return self.policy[x][y][vx + 5][vy + 5]
    
    def test_policy(self, greedy=True):
        """
        Tests the policy by simulating the car's movement on the track.

        Parameters:
            greedy (bool): If True, the car always follows the optimal policy. This actually isn't utilized but I wanted testing to be easier so it was left in

        Returns:
            int: The number of steps taken to reach the finish line.
        """
        self.start_position = self.racetrack.start_positions[0]
        self.racetrack.car.x, self.racetrack.car.y = self.start_position
        self.racetrack.car.vx, self.racetrack.car.vy = 0, 0

        print(f"Start Position: {self.start_position}")
        steps = 0
        while (self.racetrack.car.x, self.racetrack.car.y) not in self.racetrack.finish_positions:
            ax, ay = self.get_optimal_policy(self.racetrack.car.x, self.racetrack.car.y, self.racetrack.car.vx, self.racetrack.car.vy)
            print(f"Position: ({self.racetrack.car.x}, {self.racetrack.car.y}), Velocity: ({self.racetrack.car.vx}, {self.racetrack.car.vy}), Action: ({ax}, {ay})")
            self.racetrack.step(ax, ay)
            steps += 1
        
        print(f"Reached Finish Line at: ({self.racetrack.car.x}, {self.racetrack.car.y}) after {steps} steps.\n")
        return steps


if __name__ == "__main__":
    track_file = '../Data/track/R-track.txt'
    vi = ValueIteration(track_file, crash='restart')
    vi.run_value_iteration()

    start_position = vi.racetrack.start_positions[0]
    vi.racetrack.car.x, vi.racetrack.car.y = start_position
    vi.racetrack.car.vx, vi.racetrack.car.vy = 0, 0

    vi.test_policy()
