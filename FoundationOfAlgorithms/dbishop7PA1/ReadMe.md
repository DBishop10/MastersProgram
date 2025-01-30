# Closest Point Pairs Finder

This Python script finds the m closest pairs of points in a two-dimensional plane using the Manhattan distance metric. It supports inputting a custom list of points or generating a random set of points, with the ability to specify the number of closest pairs to find.

## File List
    - Closest_Pairs.py: Python file containing the code written for this project
    - PA1.pdf: PDF containing necessary information regarding the project, ie analysis, pseudocode, and more
    - ReadMe.md: This readme file explaining how everything works
    - Trace_Run_(1-4).txt: These 4 files contain all trace runs performed on the program
    - Black_Box_Test_Runs.txt: Contains the input and output of the Black Box Testing

## Compiler/Interpreter
    - I utilied Python 3.10.11
    - python3 was the command used for running the scripts

## IDE: Visual Studio Code
    - Version: 1.86.1

## Features

- **Custom Point Input**: Users can input their list of points directly through the command line.
- **Random Point Generation**: Alternatively, users can specify the number of points to be randomly generated.
- **Flexible m Value**: Users must specify the value of m, which determines the number of closest point pairs to return.

## Requirements

- Python 3.6 or later
- No external libraries are required for the main functionality. However, `argparse` and `ast` modules are used for parsing command-line arguments.

## Usage

To use this script, you can specify command-line arguments as follows:

- `--numPoints={number}`: (Optional) Specify the number of points to generate randomly. If not provided, no random points are generated.
- `--points={list of points}`: (Optional) Provide a list of points directly. Points should be formatted as a list of tuples, e.g., `"[(1,2), (3,4)]"`. If not provided, this option is ignored.
- `--m={number}`: (Required) Specify the number of closest point pairs to find.

### Examples

- Generating 100 random points and finding the 10 closest pairs:

  ```sh
  python closest_pairs.py --numPoints=100 --m=10