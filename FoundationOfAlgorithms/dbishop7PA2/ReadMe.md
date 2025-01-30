# Simulated DTM

This Python script Simulates a Deterministic Turing Machine. It also has the capability to do Binary Addition, Multiplication, and Subtraction.

## File List
    - DTM.py: Python file containing the code written for this project
    - PA2.pdf: PDF containing necessary information regarding the project, ie analysis, pseudocode, and more
    - ReadMe.md: This readme file explaining how everything works
    - Trace_Run_Base.txt: Trace Run for Base DTM
    - Trace_Run_Addition.txt: Trace run for DTM That can do Binary Addition
    - Trace_Run_Subtraction.txt: Trace run for DTM That can do Binary Addition
    - Trace_Run_Multiplication.txt: Trace run for DTM That can do Binary Addition
    - Black_Box_Test_Runs.txt: Contains the input and output of the Black Box Testing

## Compiler/Interpreter
    - I utilied Python 3.10.11
    - python3 was the command used for running the scripts

## IDE: Visual Studio Code
    - Version: 1.86.1

## Features

- **DTM**: A base simulated DTM based on the State Diagram provided
- **Addition**: Users can specify they want to add binary numbers that are input
- **Subtraction**: Users can specify they want to subtract binary numbers that are input
- **Multiplicaton**: Users can specify they want to multiply binary numbers that are input (Incomplete: Only works if the second number is a multiple of 10 (ie, 1, 10, 100...))

## Requirements

- Python 3.6 or later
- No external libraries are required for the main functionality

## Usage

To use this script, you can specify command-line arguments as follows:

- `--input={string}`: Sets the input of the string you want the DTM to use. (Each operating mode uses different seperators, use + for addition, - for subtraction, * for multiplication)
- `--operation={string}`: Sets the operating mode of the DTM, default is None. (Options are: None, Addition, Subtraction, Multiplication)

### Examples

- Run DTM in addition mode on string 100b10

  ```sh
  python DTM.py --input=100+10 --operation=Addition

- Run DTM in addition mode on string 100b10

  ```sh
  python DTM.py --input=100-10 --operation=Subtraction

- Run DTM in addition mode on string 100b10

  ```sh
  python DTM.py --input=100*10 --operation=Multiplication  