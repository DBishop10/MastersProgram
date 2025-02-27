# Closest Point Pairs Finder

This Python script sorts an Array given to it utilizing Median of Three partitioning. This can be toggled on or off depending on how one want's quicksort to partition.

## File List
    - QuickSort.py: Python file containing the code written for this project
    - PA3.pdf: PDF containing necessary information regarding the project, ie analysis, pseudocode, and more
    - ReadMe.md: This readme file explaining how everything works
    - BB_Already_Sorted_Tests.txt: This File Contains the Runs where the Arrays were already sorted
    - BB_Random_Sorted_Tests.txt: This File Contains the Runs where the Arrays were not already sorted
    - Trace_Runs.txt: More output to check program functionality

## Compiler/Interpreter
    - I utilied Python 3.10.11
    - python3 was the command used for running the scripts

## IDE: Visual Studio Code
    - Version: 1.86.1

## Features

- **Custom Array Input**: Users can input their Array of numbers directly through the command line.
- **Ability to Use or Not use Median Of Three**: Users can specify if they want to utilize MOT or not through the command line

## Requirements

- Python 3.6 or later
- No external libraries are required for the main functionality. However, `argparse` module is used for parsing command-line arguments.

## Usage

To use this script, you can specify command-line arguments as follows:

- `--A={list of numbers}`: Give's program the Array of numbers to sort
- `--MOT={Boolean}`: (Optional) Value to determine if the user wants the list sorted with Median of Three or not. Default is to use it

### Examples

- Sort [4,9,10,9,1,7,3,7,10,8] using MOT

  ```sh
  python Quicksort.py --A=[4,9,10,9,1,7,3,7,10,8] --MOT=True

  or 

  python Quicksort.py --A=[4,9,10,9,1,7,3,7,10,8]

- Sort [4,9,10,9,1,7,3,7,10,8] not using MOT
    
  ```sh
  python Quicksort.py --A=[4,9,10,9,1,7,3,7,10,8] --MOT=True
