import argparse


def simulate_dtm(input_string, operation="None"):
    """Simulate a DTM utilizing the specified operation

	Arguments:
	input_string: String, whatever input you want the DTM to parse
    operation: String, must be None, Addition, Subtraction, or Multiplication. Default is None
	"""
    WhileLoopSteps = 0 #Used for Black Box Testing Analysis
    # Γ (tape alphabet), Q (set of states), s (directions), δ (transition function), Σ (input symbols), b (blank symbol) (these are reiterated below as well)
    # Prepare the tape from input_string and append blanks on either side
    q0 = 'q0'  # Starting state
    qY = 'qY'  # Accept state
    qN = 'qN'  # Reject state
    tape = list(input_string) + ['b'] 
    head_position = 0
    current_state = q0
    steps = 0
    stepArray = []
    if(operation=="None"):
        Gamma = {'0', '1', 'b'} #Set of tape symbols 
        Sigma = {'0', '1'} #Set of input symbols
        b = 'b' #Blank symbol
        Q = {'q0', 'q1', 'q2', 'q3', 'qY', 'qN'} #Finit set of states in which the machine can be in
        s = {'LEFT': -1, 'RIGHT': 1, 'STAY': 0} #Directions the machine can move
        delta = {
            ('q0', '0'): ('q0', '0', 'RIGHT'),
            ('q0', '1'): ('q0', '1', 'RIGHT'),
            ('q0', 'b'): ('q1', 'b', 'LEFT'),
            ('q1', '0'): ('q2', 'b', 'LEFT'),
            ('q1', '1'): ('q3', 'b', 'LEFT'),
            ('q1', 'b'): ('qN', 'b', 'LEFT'),
            ('q2', '0'): ('qY', 'b', 'LEFT'),
            ('q2', '1'): ('qN', 'b', 'LEFT'),
            ('q2', 'b'): ('qN', 'b', 'LEFT'),
            ('q3', '0'): ('qN', 'b', 'LEFT'),
            ('q3', '1'): ('qN', 'b', 'LEFT'),
            ('q3', 'b'): ('qN', 'b', 'LEFT'),
        } #Transition functions that dictate machine movement
    elif(operation=="Addition"):
        Gamma = {'0', '1', 'b', '+', 'x', 'y'} #Set of tape symbols, x==1 y==0
        Sigma = {'0', '1', '+'}  #Set of input symbols
        Q = {'q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6' 'qY', 'qN'}  # States

        s = {'LEFT': -1, 'RIGHT': 1, 'STAY': 0} #Directions the machine can move
        # Transition function for addition
        delta = {
            ('q0', '0'): ('q0', '0', 'RIGHT'),
            ('q0', '1'): ('q0', '1', 'RIGHT'),
            ('q0', '+'): ('q0', '+', 'RIGHT'),
            ('q0', 'x'): ('q0', 'x', 'RIGHT'),
            ('q0', 'y'): ('q0', 'y', 'RIGHT'),
            ('q0', 'b'): ('q1', 'b', 'LEFT'),
            
            ('q1', '0'): ('q5', 'b', 'LEFT'),
            ('q1', '1'): ('q2', 'b', 'LEFT'),
            ('q1', '+'): ('q7', 'b', 'LEFT'),
            
            ('q2', '0'): ('q2', '0', 'LEFT'),
            ('q2', '1'): ('q2', '1', 'LEFT'),
            ('q2', '+'): ('q3', '+', 'LEFT'),
            
            ('q3', '0'): ('q0', 'x', 'RIGHT'),
            ('q3', 'b'): ('q0', 'x', 'RIGHT'),
            ('q3', '1'): ('q4', 'y', 'LEFT'),
            ('q3', 'y'): ('q3', 'y', 'LEFT'),
            ('q3', 'x'): ('q3', 'x', 'LEFT'),
            
            ('q4', '0'): ('q0', '1', 'RIGHT'),
            ('q4', '1'): ('q4', '0', 'LEFT'),
            ('q4', 'b'): ('q0', '1', 'RIGHT'),
            
            ('q5', '0'): ('q5', '0', 'LEFT'),
            ('q5', '1'): ('q5', '1', 'LEFT'),
            ('q5', '+'): ('q6', '+', 'LEFT'),

            ('q6', '0'): ('q0', 'y', 'RIGHT'),
            ('q6', '1'): ('q0', 'x', 'RIGHT'),
            ('q6', 'y'): ('q6', 'y', 'LEFT'),
            ('q6', 'x'): ('q6', 'x', 'LEFT'),

            ('q7', 'x'): ('q7', '1', 'LEFT'),
            ('q7', 'y'): ('q7', '0', 'LEFT'),
            ('q7', 'b'): (qY, 'b', 'STAY'),
            ('q7', '0'): (qY, '0', 'STAY'),
            ('q7', '1'): (qY, '1', 'STAY'),
        }
        tape = (['b']  * 100) + list(input_string) + ['b']  * 100 
        head_position = 100 #Set starting head position at the actual start of input not appended b's
    elif(operation=="Subtraction"):
        Gamma = {'0', '1', 'b', '-', 'x', 'y'}  #Set of tape symbols x==1 y==0
        Sigma = {'0', '1', '-'}  # Input symbols
        Q = {'q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6' 'qY', 'qN'}  # States

        s = {'LEFT': -1, 'RIGHT': 1, 'STAY': 0} #Directions the machine can move
        # Transition function for addition
        delta = {
            ('q0', '0'): ('q0', '0', 'RIGHT'),
            ('q0', '1'): ('q0', '1', 'RIGHT'),
            ('q0', '-'): ('q0', '-', 'RIGHT'),
            ('q0', 'x'): ('q0', 'x', 'RIGHT'),
            ('q0', 'y'): ('q0', 'y', 'RIGHT'),
            ('q0', 'b'): ('q1', 'b', 'LEFT'),
            
            ('q1', '0'): ('q2', 'b', 'LEFT'),
            ('q1', '1'): ('q4', 'b', 'LEFT'),
            ('q1', '-'): ('q8', 'b', 'LEFT'),
            
            ('q2', '0'): ('q2', '0', 'LEFT'),
            ('q2', '1'): ('q2', '1', 'LEFT'),
            ('q2', '-'): ('q3', '-', 'LEFT'),
            
            ('q3', '0'): ('q0', 'y', 'RIGHT'),
            ('q3', '1'): ('q0', 'x', 'RIGHT'),
            ('q3', 'y'): ('q3', 'y', 'LEFT'),
            ('q3', 'x'): ('q3', 'x', 'LEFT'),
            
            ('q4', '0'): ('q4', '0', 'LEFT'),
            ('q4', '1'): ('q4', '1', 'LEFT'),
            ('q4', '-'): ('q5', '-', 'LEFT'),
            
            ('q5', '0'): ('q6', 'x', 'LEFT'),
            ('q5', '1'): ('q0', 'y', 'RIGHT'),
            ('q5', 'y'): ('q5', 'y', 'LEFT'),
            ('q5', 'x'): ('q5', 'x', 'LEFT'),

            ('q6', '0'): ('q6', '0', 'LEFT'),
            ('q6', '1'): ('q7', '0', 'RIGHT'),

            ('q7', '0'): ('q0', '1', 'RIGHT'),
            ('q7', 'x'): ('q7', 'x', 'RIGHT'),
            ('q7', 'y'): ('q7', 'y', 'RIGHT'),
            ('q7', '-'): ('q8', 'b', 'LEFT'),

            ('q8', 'x'): ('q8', '1', 'LEFT'),
            ('q8', 'y'): ('q8', '0', 'LEFT'),
            ('q8', 'b'): (qY, 'b', 'STAY'),
            ('q8', '0'): (qY, '0', 'STAY'),
            ('q8', '1'): (qY, '1', 'STAY'),
        }
        tape = (['b']  * 100) + list(input_string) + ['b']  * 100
        head_position = 100 #Set starting head position at the actual start of input not appended b's
    elif(operation=="Multiplication"):
        Gamma = {'0', '1', 'b', '*', '+', 'x', 'y', "w", 'z'}   #Set of tape symbols x==1 y==0, w==not used, z==already used
        Sigma = {'0', '1', '*'}  # Input symbols
        Q = {'q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q12', 'q13', 'q14', 'q15', 'q16', 'q17' 'qY', 'qN'}  # States

        s = {'LEFT': -1, 'RIGHT': 1, 'STAY': 0} #Directions the machine can move
        # Transition function for addition
        delta = {
            ('q0', '0'): ('q0', '0', 'RIGHT'),
            ('q0', '1'): ('q0', '1', 'RIGHT'),
            ('q0', '*'): ('q0', '*', 'RIGHT'),
            ('q0', 'x'): ('q0', 'x', 'RIGHT'),
            ('q0', 'y'): ('q0', 'y', 'RIGHT'),
            ('q0', 'b'): ('q1', 'b', 'LEFT'),
            
            ('q1', '1'): ('q2', 'b', 'LEFT'),
            ('q1', '0'): ('q17', 'b', 'RIGHT'),
            ('q1', '*'): ('q19', 'b', 'RIGHT'),
            
            ('q2', '0'): ('q2', '0', 'LEFT'),
            ('q2', '1'): ('q2', '1', 'LEFT'),
            ('q2', '*'): ('q3', '*', 'LEFT'),
            
            ('q3', '0'): ('q3', '0', 'LEFT'),
            ('q3', '1'): ('q3', '1', 'LEFT'),
            ('q3', 'y'): ('q4', 'y', 'RIGHT'),
            ('q3', 'x'): ('q4', 'x', 'RIGHT'),
            ('q3', 'b'): ('q4', 'b', 'RIGHT'),
            
            ('q4', '0'): ('q8', 'y', 'RIGHT'),
            ('q4', '1'): ('q5', 'x', 'RIGHT'),
            ('q4', '*'): ('q10', '*', 'RIGHT'),
            
            ('q5', '0'): ('q5', '0', 'RIGHT'),
            ('q5', '1'): ('q5', '1', 'RIGHT'),
            ('q5', 's'): ('q5', 's', 'RIGHT'),
            ('q5', 'b'): ('q5', 'b', 'RIGHT'),
            ('q5', '*'): ('q5', '*', 'RIGHT'),
            ('q5', 'r'): ('q6', 'r', 'RIGHT'),
            ('q5', 'w'): ('q5', 'w', 'RIGHT'),

            ('q6', '0'): ('q6', '0', 'RIGHT'),
            ('q6', '1'): ('q6', '1', 'RIGHT'),
            ('q6', '+'): ('q6', '+', 'RIGHT'),
            ('q6', 'b'): ('q7', '1', 'LEFT'),

            ('q7', '0'): ('q7', '0', 'LEFT'),
            ('q7', '1'): ('q7', '1', 'LEFT'),
            ('q7', 'b'): ('q7', 'b', 'LEFT'),
            ('q7', 's'): ('q7', 's', 'LEFT'),
            ('q7', 'r'): ('q7', 'r', 'LEFT'),
            ('q7', 'w'): ('q7', 'w', 'LEFT'),
            ('q7', '*'): ('q3', '*', 'LEFT'),
            ('q7', '+'): ('q7', '+', 'LEFT'),
        
            ('q8', '0'): ('q8', '0', 'RIGHT'),
            ('q8', '1'): ('q8', '1', 'RIGHT'),
            ('q8', 's'): ('q8', 's', 'RIGHT'),
            ('q8', 'b'): ('q8', 'b', 'RIGHT'),
            ('q8', '*'): ('q8', '*', 'RIGHT'),
            ('q8', 'w'): ('q8', 'w', 'RIGHT'),
            ('q8', 'r'): ('q9', 'r', 'RIGHT'),

            ('q9', '0'): ('q9', '0', 'RIGHT'),
            ('q9', '1'): ('q9', '1', 'RIGHT'),
            ('q9', '+'): ('q9', '+', 'RIGHT'),
            ('q9', 'b'): ('q7', '0', 'LEFT'),

            ('q10', '0'): ('q10', '0', 'RIGHT'),
            ('q10', '1'): ('q10', '1', 'RIGHT'),
            ('q10', 's'): ('q10', 's', 'RIGHT'),
            ('q10', 'b'): ('q10', 'b', 'RIGHT'),
            ('q10', 'w'): ('q10', 'w', 'RIGHT'),
            ('q10', 'r'): ('q11', 'r', 'RIGHT'),

            ('q11', '0'): ('q11', '0', 'RIGHT'),
            ('q11', '1'): ('q11', '1', 'RIGHT'),
            ('q11', '+'): ('q11', '+', 'RIGHT'),
            ('q11', 'b'): ('q12', '+', 'LEFT'),

            ('q12', '0'): ('q12', '0', 'LEFT'),
            ('q12', '1'): ('q12', '1', 'LEFT'),
            ('q12', '+'): ('q12', '+', 'LEFT'),
            ('q12', 'r'): ('q12', 'r', 'LEFT'),
            ('q12', 'b'): ('q12', 'b', 'LEFT'),
            ('q12', 'w'): ('q12', 'w', 'LEFT'),
            ('q12', 'z'): ('q12', 'z', 'LEFT'),
            ('q12', 's'): ('q13', 's', 'RIGHT'),

            ('q13', 'z'): ('q13', 'z', 'RIGHT'),
            ('q13', 'b'): ('q14', 'w', 'LEFT'),
            ('q13', 'w'): ('q15', 'z', 'RIGHT'),

            ('q14', 's'): ('q14', 's', 'LEFT'),
            ('q14', 'w'): ('q14', 'w', 'LEFT'),
            ('q14', 'b'): ('q14', 'b', 'LEFT'),
            ('q14', 'z'): ('q14', 'w', 'LEFT'),
            ('q14', '1'): ('q22', '1', 'LEFT'),
            ('q14', '0'): ('q22', '0', 'LEFT'),
            ('q14', '*'): ('q1', '*', 'STAY'),

            ('q15', '0'): ('q15', '0', 'RIGHT'),
            ('q15', '1'): ('q15', '1', 'RIGHT'),
            ('q15', 'r'): ('q15', 'r', 'RIGHT'),
            ('q15', 'w'): ('q15', 'w', 'RIGHT'),
            ('q15', '+'): ('q23', '+', 'RIGHT'),
            ('q15', 'b'): ('q15', 'b', 'RIGHT'),

            ('q16', '+'): ('q16', '0', 'RIGHT'),
            ('q16', 'b'): ('q12', '+', 'LEFT'),
            ('q16', '0'): ('q15', '0', 'RIGHT'),
            ('q16', '1'): ('q15', '1', 'RIGHT'),

            ('q17', 'b'): ('q17', 'b', 'RIGHT'),
            ('q17', 's'): ('q18', 's', 'RIGHT'),

            ('q18', 'w'): ('q18', 'w', 'RIGHT'),
            ('q18', 'b'): ('q14', 'w', 'LEFT'),

            ('q19', 'b'): ('q19', 'b', 'RIGHT'),
            ('q19', 's'): ('q19', 's', 'RIGHT'),
            ('q19', 'w'): ('q19', 'w', 'RIGHT'),
            ('q19', 'r'): ('q20', 'r', 'RIGHT'),

            ('q20', '+'): ('q20', '+', 'RIGHT'),
            ('q20', '1'): ('q20', '1', 'RIGHT'),
            ('q20', '0'): ('q20', '0', 'RIGHT'),
            ('q20', 'b'): ('q21', 'b', 'LEFT'),

            ('q21', '+'): (qY, 'b', 'STAY'),
            
            ('q22', '*'): ('q22', '*', 'LEFT'),
            ('q22', 'x'): ('q22', '1', 'LEFT'),
            ('q22', 'y'): ('q22', '0', 'LEFT'),
            ('q22', 'b'): ('q0', 'b', 'RIGHT'),
            ('q22', '0'): ('q22', '0', 'LEFT'),
            ('q22', '1'): ('q22', '1', 'LEFT'),

            ('q23', 'b'): ('q16', 'b', 'LEFT'),
            ('q23', '1'): ('q23', '1', 'RIGHT'),
            ('q23', '0'): ('q23', '0', 'RIGHT'),
            ('q23', '+'): ('q16', '+', 'STAY'),

        }
        tape = (['b']  * 100) + list(input_string) + ['b'] + ['s'] + ['b']  * 50 + ['r'] + ['b']  * 50
        head_position = 100 #Set starting head position at the actual start of input not appended b's

    # Simulation
    print(f"Initial tape state: {''.join(tape).strip('b')}")  # Remove trailing blanks for readability
    while current_state not in {qY, qN}:
        WhileLoopSteps += 1
        symbol_read = tape[head_position] if head_position < len(tape) else 'b'
        if (current_state, symbol_read) in delta:
            next_state, symbol_to_write, move_direction = delta[(current_state, symbol_read)]
            tape[head_position] = symbol_to_write
            if move_direction == 'RIGHT':
                head_position += 1
            elif move_direction == 'LEFT':
                head_position -= 1
                if head_position < 0:  # Ensure head doesn't go beyond the tape's start
                    tape.insert(0, 'b')
                    head_position = 0
            current_state = next_state
        else:
            print("No transition defined for this state and symbol.")
            break
        steps += 1
        stepArray.append(f"Step {steps}: {''.join(tape).strip('b')} Current state: {current_state}")
        print(f"Step {steps}: {''.join(tape).strip('b')} Current state: {current_state}")
        if steps == 30:
            with open('dbishop7PA2/dtm_simulation_output.txt', 'a') as f:
                for step in stepArray:
                    f.write(step + "\n")
        if steps > 30:
            with open('dbishop7PA2/dtm_simulation_output.txt', 'a') as f:
                f.write(f"Step {steps}: {''.join(tape).strip('b')} Current state: {current_state}\n")

    # Final tape state
    tape = ''.join(tape)
    tape = tape.replace('b', "")
    tape = tape.replace('s', "")
    tape = tape.replace('w', "")
    tape = tape.replace('x', "")
    tape = tape.replace('y', "")
    tape = tape.replace('r', "")
    tape = tape.replace('+', "")
    print(f"Final tape state: {tape} Number of steps: {steps}")
    if steps > 30:
        print(f"The state of the tape after each transition has been written to 'dtm_simulation_output.txt'")
    print("Number of Times While Loop Ran: ", WhileLoopSteps)

# Setup argument parser
parser = argparse.ArgumentParser(description='Process command-line options for finding closest point pairs.')
parser.add_argument('--input', type=str, help='Input you want the DTM to run')
parser.add_argument('--operation', type=str, help='What operation you want the code to perform', default="None")

# Parse arguments
args = parser.parse_args()

# Check if --input is set
if args.input == "":
    parser.error("Please use --input to give the DTM the string you want it to run")
    
if(args.input != ""):
	input = args.input
if(args.operation != ""):
    operationtype = args.operation

simulate_dtm(input, operationtype)