# Interwoven Strings

This Python script Determines if a streamed string is an interweave of two substrings including noise or not. 

## File List
    - Interweave.py: Python file containing the code written for this project
    - PA4.pdf: PDF containing necessary information regarding the project, ie analysis, pseudocode, and more
    - ReadMe.md: This readme file explaining how everything works
    - Required_Cases.txt: This File Contains the required test cases requested in part 3 of the project
    - Asymptotic_Analysis.txt: Contains the length/string utilized in asymptotic_analysis

## Compiler/Interpreter
    - I utilied Python 3.10.11
    - python3 was the command used for running the scripts

## IDE: Visual Studio Code
    - Version: 1.86.1

## Features

- **Interweave Checker**: Users can input s, x, and y and check if they are interweaved or not.

## Requirements

- Python 3.6 or later
- No external libraries are required for the main functionality.

## Usage

To use this script, simply run it with python3 Interweave.py. I left only the required Test Cases for this project, but to add more you can simply utilize this code format:
```python
s = ""
x = ""
y = ""
indices_x, indices_y, indices_noise, is_interweave = interweaved(s, x, y)
print("Is Interweaving: " + str(is_interweave) + "")
print("Indices for x: {" + str(indices_x)[1:-1] + "}")
print("Indices for y: {" + str(indices_y)[1:-1] + "}")
print("Indices for noise: {" + str(indices_noise)[1:-1] + "}")
```
### Examples

- Running Required Test Cases

  ```sh
  python3 Interweave.py