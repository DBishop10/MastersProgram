import random

def read_track(file_path):
    """
    Reads a racetrack from a file and returns its size and grid representation.

    Parameters:
        file_path (str): The path to the track file.

    Returns:
        tuple: A tuple containing the size of the track as (width, height) and 
               the grid representation as a list of lists of characters.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        size = tuple(map(int, lines[0].strip().split(',')))
        grid = [list(line.strip()) for line in lines[1:]]
    return size, grid

class RaceCar:
    """
    Represents a race car on a racetrack.

    Attributes:
        x (int): The current x-coordinate of the car.
        y (int): The current y-coordinate of the car.
        start (tuple): The starting position of the car as (x, y).
        vx (int): The current velocity in the x direction.
        vy (int): The current velocity in the y direction.
        debug (bool): A flag for debugging mode.
        grid (list): The grid representation of the racetrack.
    """
    def __init__(self, start_position, grid, debug=False):
        """
        Initializes the RaceCar with a start position and racetrack grid.

        Parameters:
            start_position (tuple): The starting position of the car as (x, y).
            grid (list): The grid representation of the racetrack.
            debug (bool): A flag for enabling or disabling debugging mode.
        """
        self.x, self.y = start_position
        self.start = start_position
        self.vx, self.vy = 0, 0
        self.debug = debug
        self.grid = grid
    
    def move(self, ax, ay, crash='nearest'):
        """
        Moves the car based on the given acceleration values.

        Parameters:
            ax (int): Acceleration in the x direction, must be -1, 0, or 1.
            ay (int): Acceleration in the y direction, must be -1, 0, or 1.
            crash (str): The crash handling strategy ('nearest' or 'restart').

        Returns:
            bool: True if the move is successful, False if the car crashes.
        """
        if ax not in [-1, 0, 1] or ay not in [-1, 0, 1]:
            raise ValueError("Acceleration must be -1, 0, or 1")
        
        new_vx = min(max(self.vx + ax, -5), 5)
        new_vy = min(max(self.vy + ay, -5), 5)
        new_x = self.x + new_vx
        new_y = self.y + new_vy

        path = bresenham(self.x, self.y, new_x, new_y)

        if self.debug:
            print(f"Checking path from ({self.x},{self.y}) to ({new_x},{new_y})")
            print(f"Path: {path}")
        for x, y in path:
            if self.debug:
                print(f"Checking Position: {x}, {y}")
            if is_off_track(x, y, self.grid):
                if self.debug:
                    print(f"Collision detected at ({x},{y})")
                handle_crash(self, self.grid, path, crash)
                return False  

        if not self.debug:
            if random.random() > 0.2:
                self.vx = new_vx
                self.vy = new_vy
            
            self.x = new_x
            self.y = new_y
        else: # This was to check that the movement was correct, not really needed and should probably be removed for true debugging
            self.x += ax
            self.y += ay
        
        return True 
    
    def get_state(self):
        """
        Gets the current state of the car.

        Returns:
            tuple: The current state as (x, y, vx, vy).
        """
        return (self.x, self.y, self.vx, self.vy)
    
    def reset(self, position):
        """
        Resets the car to a given position.

        Parameters:
            position (tuple): The position to reset the car to as (x, y).
        """
        self.x, self.y = position
        self.vx, self.vy = 0, 0

def bresenham(x0, y0, x1, y1):
    """
    Implements the Bresenham's line algorithm to calculate the points on a line.

    Parameters:
        x0 (int): The starting x-coordinate.
        y0 (int): The starting y-coordinate.
        x1 (int): The ending x-coordinate.
        y1 (int): The ending y-coordinate.

    Returns:
        list: A list of tuples representing the points on the line.
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    
    return points

def is_off_track(x, y, grid):
    """
    Checks if a given position is off the racetrack.

    Parameters:
        x (int): The x-coordinate to check.
        y (int): The y-coordinate to check.
        grid (list): The grid representation of the racetrack.

    Returns:
        bool: True if the position is off the track, False otherwise.
    """
    if x < 0 or x >= len(grid) or y < 0 or y >= len(grid[0]):
        return True
    return grid[x][y] == '#'

def handle_crash(car, grid, path, variant='nearest'):
    """
    Handles the car's crash event based on the given strategy.

    Parameters:
        car (RaceCar): The race car that crashed.
        grid (list): The grid representation of the racetrack.
        path (list): The path the car took before crashing.
        variant (str): The crash handling strategy ('nearest' or 'restart').
    """
    if variant == 'restart':
        car.reset(car.start)  
    else:
        car.vx, car.vy = 0, 0
        for (x, y) in reversed(path):
            if not is_off_track(x, y, grid):
                car.x, car.y = x, y
                break

class Racetrack:
    """
    Represents a racetrack with a car and finish line.

    Attributes:
        size (tuple): The size of the racetrack as (width, height).
        grid (list): The grid representation of the racetrack.
        start_positions (list): List of starting positions on the track.
        finish_positions (list): List of finish line positions on the track.
        car (RaceCar): The race car on the track.
        crash (str): The crash handling strategy.
        debug (bool): A flag for debugging mode.
    """
    def __init__(self, track_file, debug=False, crash='nearest'):
        """
        Initializes the racetrack from a file and places the car at the start.

        Parameters:
            track_file (str): The file path of the racetrack.
            debug (bool): A flag for enabling or disabling debugging mode.
            crash (str): The crash handling strategy ('nearest' or 'restart').
        """
        self.size, self.grid = read_track(track_file)
        self.start_positions = [(x, y) for x in range(self.size[0]) for y in range(self.size[1]) if self.grid[x][y] == 'S']
        self.finish_positions = [(x, y) for x in range(self.size[0]) for y in range(self.size[1]) if self.grid[x][y] == 'F']
        self.car = RaceCar(self.start_positions[0], self.grid, debug)
        self.crash = crash
        self.debug = debug
    
    def step(self, ax, ay):
        """
        Moves the car by one step based on the given acceleration.

        Parameters:
            ax (int): Acceleration in the x direction.
            ay (int): Acceleration in the y direction.

        Returns:
            str: "crash" if the car crashes, "finish" if the car finishes, "valid" otherwise.
        """
        if not self.car.move(ax, ay, self.crash): 
            handle_crash(self.car, self.grid, [], self.crash)
            return "crash" 
        
        if (self.car.x, self.car.y) in self.finish_positions:
            return "finish"  
        
        return "valid" 
    
    def reset(self):
        """
        Resets the car to the start position.
        """
        self.car.reset(self.start_positions[0])
    
    def get_state(self):
        """
        Gets the current state of the car.

        Returns:
            tuple: The current state as (x, y, vx, vy).
        """
        return self.car.get_state()



if __name__ == "__main__":
    track_size, track_grid = read_track("../Data/track/L-track.txt")
    print(track_grid, track_size)

    #Check bresenham works
    path = bresenham(0, 0, 5, 3)
    print(path)

    #Determininstic Example to ensure functionality of start and finish, debug is true so it has no velocity, only movement
    track = Racetrack('../Data/track/L-track.txt', True)

    actions = [(0, 1)] * 34  
    actions += [(-1, 0)] * 5  

    for ax, ay in actions:
        done = track.step(ax, ay)
        car_state = track.get_state()
        
        print(f"Car position: (x: {car_state[0]}, y: {car_state[1]}), velocity: (vx: {car_state[2]}, vy: {car_state[3]})")
        
        if done:
            print("Car reached the finish line!")
            break

    # Non deterministic movement to ensure that functions
    track = Racetrack('../Data/track/L-track.txt', debug=False)

    actions = [(0, 1)] * 34
    actions += [(-1, 0)] * 6
    # This sometimes will cause it to crash into a wall as 34 is too high, but good enough for my testing
    for ax, ay in actions:
        done = track.step(ax, ay)
        car_state = track.get_state()
        print(f"Car position: (x: {car_state[0]}, y: {car_state[1]}), velocity: (vx: {car_state[2]}, vy: {car_state[3]})")
        
        if done == "finish":
            print("Car reached the finish line!")
            break