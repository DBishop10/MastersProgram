import argparse

#These imports were utilized in testing, not part of normal code

# import random 
# import copy
# import sys

comparisons = 0 #Utilized for black box testing
swaps = 0 #Utilized for black box testing
# file1 = open(r"Trace_Runs.txt", "a")


def median_of_three_partition(A, p, r):
    """Partitions the Array Utilizing Median Of Three Partitioning

	Arguments:
	A -- Array
    p -- starting index
    r -- ending index
	"""
    global comparisons, swaps
    k = (p + r) // 2

    # Ensure the median of A[p], A[k], A[r] is at position A[r]
    if A[p] > A[k]:
        A[p], A[k] = A[k], A[p]
        swaps += 1
    if A[p] > A[r]:
        A[p], A[r] = A[r], A[p]
        swaps += 1
    if A[k] > A[r]:
        A[k], A[r] = A[r], A[k]
        swaps += 1
    comparisons += 3
    # Exchange A[k] with A[r-1] to put the pivot (A[r]) into the right position
    A[k], A[r-1] = A[r-1], A[k]
    swaps += 1
    pivot = A[r-1]
    # Partitioning step
    i = p - 1
    for j in range(p, r-1):
        comparisons += 1
        if A[j] <= pivot:
            i += 1
            A[i], A[j] = A[j], A[i]
            swaps += 1
    A[i+1], A[r-1] = A[r-1], A[i+1]
    return i+1

def partition(A, p, r):
    """Partitions the Array Not Utilizing Median Of Three Partitioning

	Arguments:
	A -- Array
    p -- starting index
    r -- ending index
	"""
    global comparisons, swaps
    x = A[r]  # Using the last element as the pivot
    i = p - 1  # Place for the next element in the low side
    for j in range(p, r):  # Process each element other than the pivot
        comparisons += 1
        if A[j] <= x:  # Does this element belong on the low side?
            i = i + 1  # Index of a new slot in the low side
            A[i], A[j] = A[j], A[i]  # Put this element there
            swaps += 1
    # Place the pivot after the last element of the low side
    A[i+1], A[r] = A[r], A[i+1]
    swaps += 1
    # Return the index of the new pivot
    return i+1

def quicksort_with_median_of_three(A, p, r):
    """Quicksort Code That utilizes Median of Three Partitioning

	Arguments:
	A -- Array
    p -- starting index
    r -- ending index
	"""
    global file1
    if p < r:
        # file1.write("Array At Current Step: " + str(A) + "\n")
        q = median_of_three_partition(A, p, r) #Calls Median Of Three Partitioning To partition
        quicksort_with_median_of_three(A, p, q-1) 
        quicksort_with_median_of_three(A, q+1, r)

def quicksort(A, p, r):
    """Quicksort Code That utilizes Median of Three Partitioning

	Arguments:
	A -- Array
    p -- starting index
    r -- ending index
	"""
    global file1
    if p < r:
        # file1.write("Array At Current Step: " + str(A) + "\n")
        q = partition(A, p, r) #Calls Normal Partitioning To partition
        quicksort(A, p, q-1)
        quicksort(A, q+1, r)


############
#This Is For Running from Command line
############

# Setup argument parser
parser = argparse.ArgumentParser(description='Process command-line options for finding closest point pairs.')
parser.add_argument('--A', help='List of points provided by the user.', default=[])
parser.add_argument('--MOT', type=bool, help='Set if you want MOT mode or regular parsing mode', default=True)

# Parse arguments
args = parser.parse_args()

A = args.A.replace('[', '').replace(']', '').split(',')
A = [eval(i) for i in A]

useMOT = args.MOT

if(useMOT):
    quicksort_with_median_of_three(A, 0, len(A) - 1)
else:
    quicksort(A, 0, len(A) - 1)
print("Sorted Array: \n" + str(A))



######################
#The Below Was utilized for Testing Purposes
######################


# def generate_array(n):
#     return [random.randint(1, n) for _ in range(n)]


# A = generate_array(10000)
# A.sort()
# file1.write("Array Used For Testing: " + str(A) + "\n")
# file1.write("Array Size For Testing: " + str(len(A)) + "\n\n")
# sys.setrecursionlimit(10000000)
# B = copy.deepcopy(A)
# C = copy.deepcopy(A)
# file1.write("MOT: \n\n")
# quicksort_with_median_of_three(B, 0, len(B) - 1)
# file1.write("Number of Swap Calls: " + str(swaps) + "\n")
# file1.write("Number of Comparisons Calls: " + str(comparisons) + "\n")
# file1.write("Number of Overall Calls: " + str(swaps + comparisons) + "\n\n")
# print(comparisons)
# swaps = 0
# comparisons = 0
# file1.write("Reuglar Quicksort: \n\n")
# quicksort(C, 0, len(C) - 1)
# file1.write("Number of Swap Calls: " + str(swaps) + "\n")
# file1.write("Number of Comparisons Calls: " + str(comparisons) + "\n")
# file1.write("Number of Overall Calls: " + str(swaps + comparisons) + "\n")
# print(comparisons)