file1 = open("tests/Asymptotic_Analysis.txt", "a") #File For Asymptotic Analysis
file2 = open("tests/Required_Cases.txt", "a") #File For Required Test Cases

def interweaved(s: str, x: str, y: str):
    # Variables For Asymptotic Analysis
    loop_counter = 0  
    compare_counter = 0
    append_counter = 0
    reset_counter = 0
    # Two pointers for each sequence
    len_x, len_y = len(x), len(y)
    result_x, result_y, result_noise = [], [], []

    # The potential indices to be cleared if needed
    potential_x, potential_y = [], []
    potential_noise = []

    # Pointers for tracking positions in x and y
    idx_x, idx_y, idx_noise = 0, 0, 0
    i = 0  # Index for s

    
    for i in range(len(s)):
        loop_counter += 1  # Count each iteration of the loop
        if(idx_x >= idx_y):
            compare_counter += 1  # Compare idx_x and idx_y
            if(idx_x < len_x and s[i] == x[idx_x]):
                append_counter += 1
                potential_x.append(i+1)
                idx_x += 1
            elif(idx_y < len_y and s[i] == y[idx_y]):
                append_counter += 1
                potential_y.append(i+1)
                idx_y += 1
            else:
                append_counter += 1
                potential_noise.append(i+1)
                idx_noise += 1
        elif(idx_y > idx_x):
            compare_counter += 1
            if(idx_y < len_y and s[i] == y[idx_y]):
                append_counter += 1
                potential_y.append(i+1)
                idx_y += 1
            elif(idx_x < len_x and s[i] == x[idx_x]):
                append_counter += 1
                potential_x.append(i+1)
                idx_x += 1
            else:
                append_counter += 1
                potential_noise.append(i+1)
                idx_noise += 1
        if(len(potential_x) == len_x):
            reset_counter += 1
            append_counter += 1
            result_x.append(potential_x)
            potential_x = []
            idx_x = 0
            if(len(potential_noise) > 0):
                reset_counter += 1
                append_counter += 1
                result_noise.append(potential_noise)
                potential_noise = []
            idx_noise = 0
        if(len(potential_y) == len_y):
            reset_counter += 1
            append_counter += 1
            result_y.append(potential_y)
            potential_y = []
            idx_y = 0
            if(len(potential_noise) > 0):
                reset_counter += 1
                append_counter += 1
                result_noise.append(potential_noise)
                potential_noise = []
            idx_noise = 0
        if(idx_noise >= 1 and len(potential_x) <= 1 and len(potential_y) <= 1):
            reset_counter += 3
            if(len(potential_x) > 0):
                potential_noise.insert(potential_x[0]-1, potential_x[0])
                potential_x = []
            if(len(potential_y) > 0):
                potential_noise.insert(potential_y[0]-1, potential_y[0])
                potential_y = []
            idx_noise = 0
            idx_y = 0
            idx_x = 0
    append_counter += 1
    result_noise.append(potential_noise)

    file1.write(f"Loop iterations: {loop_counter}" + "\n")
    file1.write(f"Comparisons: {compare_counter}" + "\n")
    file1.write(f"Appends: {append_counter}" + "\n")
    file1.write(f"Resets and copies: {reset_counter}" + "\n")

    is_interweave = False
    if(result_noise == [[]]):
        is_interweave = True
    elif(len(result_noise) <= 2): #Accounts for if there is only noise at beginning and end 
        if(len(result_noise) == 1):
            if(int(result_noise[0][-1]) < int(result_x[0][0]) or int(result_noise[0][0]) > int(result_x[-1][-1])):
                is_interweave = True
        else:
            if(int(result_noise[0][-1]) < int(result_x[0][0]) or int(result_noise[-1][0]) < int(result_x[-1][-1])):
                is_interweave = True
    return result_x, result_y, result_noise, is_interweave



# Required Testing Cases:
s = "100010101"
x = "101"
y = "0"
file1.write("Input Variable s: " + s + "\n")
file1.write("Input Length: " + str(len(s)) + "\n")
file2.write("s1 \n")
file2.write("s: " + s + ", x: " + x + ", y: " + y + "\n")
indices_x, indices_y, indices_noise, is_interweave = interweaved(s, x, y)
file2.write("Is Interweaving: " + str(is_interweave) + "\n")
file2.write("Indices for x: {" + str(indices_x)[1:-1] + "}\n")
file2.write("Indices for y: {" + str(indices_y)[1:-1] + "}\n")
file2.write("Indices for noise: {" + str(indices_noise)[1:-1] + "}\n\n")

s = "101010101010101"
x = "101"
y = "010"
file1.write("Input Variable s: " + s + "\n")
file1.write("Input Length: " + str(len(s)) + "\n")
file2.write("s2 \n")
file2.write("s: " + s + ", x: " + x + ", y: " + y + "\n")
indices_x, indices_y, indices_noise, is_interweave = interweaved(s, x, y)
file2.write("Is Interweaving: " + str(is_interweave) + "\n")
file2.write("Indices for x: {" + str(indices_x)[1:-1] + "}\n")
file2.write("Indices for y: {" + str(indices_y)[1:-1] + "}\n")
file2.write("Indices for noise: {" + str(indices_noise)[1:-1] + "}\n\n")

s = "001100110101011001100110010101111"
x = "101"
y = "010"
file1.write("Input Variable s: " + s + "\n")
file1.write("Input Length: " + str(len(s)) + "\n")
file2.write("s3 \n")
file2.write("s: " + s + ", x: " + x + ", y: " + y + "\n")
indices_x, indices_y, indices_noise, is_interweave = interweaved(s, x, y)
file2.write("Is Interweaving: " + str(is_interweave) + "\n")
file2.write("Indices for x: {" + str(indices_x)[1:-1] + "}\n")
file2.write("Indices for y: {" + str(indices_y)[1:-1] + "}\n")
file2.write("Indices for noise: {" + str(indices_noise)[1:-1] + "}\n\n")

s = "100110011001"
x = "101"
y = "010"
file1.write("Input Variable s: " + s + "\n")
file1.write("Input Length: " + str(len(s)) + "\n")
file2.write("s4 \n")
file2.write("s: " + s + ", x: " + x + ", y: " + y + "\n")
indices_x, indices_y, indices_noise, is_interweave = interweaved(s, x, y)
file2.write("Is Interweaving: " + str(is_interweave) + "\n")
file2.write("Indices for x: {" + str(indices_x)[1:-1] + "}\n")
file2.write("Indices for y: {" + str(indices_y)[1:-1] + "}\n")
file2.write("Indices for noise: {" + str(indices_noise)[1:-1] + "}")

#Extra Test Cases for Length Asymptotic Anaylsis
s = "101010101010101010101010101010101010101010101010101"
x = "101"
y = "010"
file1.write("Input Variable s: " + s + "\n")
file1.write("Input Length: " + str(len(s)) + "\n")
indices_x, indices_y, indices_noise, is_interweave = interweaved(s, x, y)

s = "10101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101"
x = "101"
y = "010"
file1.write("Input Variable s: " + s + "\n")
file1.write("Input Length: " + str(len(s)) + "\n")
indices_x, indices_y, indices_noise, is_interweave = interweaved(s, x, y)

s = "101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101"
x = "101"
y = "010"
file1.write("Input Variable s: " + s + "\n")
file1.write("Input Length: " + str(len(s)) + "\n")
indices_x, indices_y, indices_noise, is_interweave = interweaved(s, x, y)

file1.close()
file2.close()