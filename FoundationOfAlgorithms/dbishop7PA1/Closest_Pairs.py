import random
import argparse
import ast

distance_calculations = 0 #Utilized for black box testing
merge_sort_runs = 0 #Utilized for black box testing

def point_list(amount):
    """Create a list of points with a length equal to input amount

	Arguments:
	amount -- Size of requested list
	"""
    pointlist = []
    for i in range(amount):
        point1 = random.randrange(1, 1000)
        point2 = random.randrange(1, 1000)
        pointlist.append((point1, point2))
    f.write(f"Generated Points List: {pointlist}\n")
    return(pointlist)

def calculate_manhattan_distance(point1, point2):
    """Calculate the manhatten distance between point 1 and point 2.

	Arguments:
	point1 -- The first provided point
	point2 -- The second provided point
	"""
    global distance_calculations #Utilized for black box testing
    distance_calculations += 1 #Utilized for black box testing
    distance = abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
    # print(f"Calculating Manhattan distance between {point1} and {point2}: {distance}") Utilized in running Trace Runs
    return distance

def find_closest_pairs(points, m):
    # print(f"Starting find_closest_pairs with points={points} and m={m}") Utilized in running Trace Runs
    """Find the closest m pairs of points

	Arguments:
	points -- set of points of n size
	m -- number of closest points we want returned
	"""
    n = len(points)
    distance_list = []
	#Ensure that m meets required criteria of greater tha 0 or less than or equal to n/2
    if not (0 < m <= n // 2):
        raise Exception("m does not meet requirements, m has to be greater than 0 or less than length of the list divided by 2.")
    # Calculate all pairwise Manhattan distances
    for i in range(n - 1):
        for j in range(i + 1, n):
            distance = calculate_manhattan_distance(points[i], points[j])
            distance_list.append((distance, points[i], points[j]))
    
    # Sort the list by distances in ascending order
    merge_sort(distance_list)
    # print(f"Dinstances sorted, sorted list:{distance_list}") Utilized in running Trace Runs
    # Select the first m entries from the sorted list
    closest_pairs = distance_list[:m]
    # Return the closest m pairs and their distances
    f.write(f"Final closest {m} pairs found: {closest_pairs}\n") #Utilized in running Trace Runs
    return closest_pairs

def merge(A, p, q, r):
	"""Merge two sorted sublists/subarrays to a larger sorted sublist/subarray.

	Arguments:
	A -- a list/array containing the sublists/subarrays to be merged
	p -- index of the beginning of the first sublist/subarray
	q -- index of the end of the first sublist/subarray;
	the second sublist/subarray starts at index q+1
	r -- index of the end of the second sublist/subarray
	"""
	global merge_sort_runs #Utilized for black box testing
	merge_sort_runs += 1 #Utilized for black box testing
	# Copy the left and right sublists/subarrays.
	# If A is a list, slicing creates a copy.
	if type(A) is list:
		left = A[p: q+1]
		right = A[q+1: r+1]
	# Otherwise a is a np.array, so create a copy with list().
	else:
		left = list(A[p: q+1])
		right = list(A[q+1: r+1])

	i = 0    # index into left sublist/subarray
	j = 0    # index into right sublist/subarray
	k = p    # index into a[p: r+1]

	# Combine the two sorted sublists/subarrays by inserting into A
	# the lesser exposed element of the two sublists/subarrays.
	while i < len(left) and j < len(right):
		if left[i] <= right[j]:
			A[k] = left[i]
			i += 1
		else:
			A[k] = right[j]
			j += 1
		k += 1

	# After going through the left or right sublist/subarray, copy the 
	# remainder of the other to the end of the list/array.
	if i < len(left):  # copy remainder of left
		A[k: r+1] = left[i:]
	if j < len(right):  # copy remainder of right
		A[k: r+1] = right[j:]


def merge_sort(A, p=0, r=None):
	"""Sort the elements in the sublist/subarray a[p:r+1].

	Arguments:
	A -- a list/array containing the sublist/subarray to be merged
	p -- index of the beginning of the sublist/subarray (default = 0)
	r -- index of the end of the sublist/subarray (default = None)
	"""
	# If r is not given, set to the index of the last element of the list/array.
	if r is None:
		r = len(A) - 1
	if p >= r:  # 0 or 1 element?
		return
	q = (p+r) // 2            # midpoint of A[p: r]
	merge_sort(A, p, q)       # recursively sort A[p: q]
	merge_sort(A, q + 1, r)   # recursively sort A[q+1: r]
	merge(A, p, q, r)         # merge A[p: q] and A[q+1: r] into A[p: r]

# Function to parse a list of points input as a string
def parse_points(points_str):
    """Parses a string of points that the user can pass

	Arguments:
	points_str -- A string that contains points passed in by the user
	"""
    try:
        # Safely evaluate the string as a list
        points = ast.literal_eval(points_str)
        if isinstance(points, list) and all(isinstance(p, tuple) and len(p) == 2 for p in points):
            return points
        else:
            raise ValueError
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Invalid points format. Please provide a list of point tuples.")


# Setup argument parser
parser = argparse.ArgumentParser(description='Process command-line options for finding closest point pairs.')
parser.add_argument('--numPoints', type=int, help='Number of points to generate randomly.', default=0)
parser.add_argument('--points', type=parse_points, help='List of points provided by the user.', default=[])
parser.add_argument('--m', type=int, help='Number of closest point pairs to find.', required=True)

# Parse arguments
args = parser.parse_args()

# Check if neither --numPoints nor --points is provided by the user
if args.numPoints == 0 and args.points == []:
    parser.error("One of --numPoints or --points is required.")
    
if(args.numPoints != 0):
	closest_pairs = find_closest_pairs(point_list(args.numPoints), args.m)
	print(f"Final closest {args.m} pairs found: {closest_pairs}")
elif(args.points != []):
    closest_pairs = find_closest_pairs(args.points, args.m)
    print(f"Final closest {args.m} pairs found: {closest_pairs}")
else:
    print("Ensure that you added either the argument --numPoints or --points, as well as --m")
    
# f = open("black_box.txt", "a")
# # This for loop was utilized for black box testing
# for n in [100, 1000, 5000, 10000]:
# 	closest_pairs = find_closest_pairs(point_list(n), 50)
# 	f.write(f"For n={n}, distance calculations: {distance_calculations}, merge_sort runs: {merge_sort_runs}\n")
# 	distance_calculations = 0
# 	merge_sort_runs = 0

