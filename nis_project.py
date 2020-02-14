import math
import random
import copy

# Class to store node data.
class Node:
	def __init__(self, _id, _x, _y):
		self.id = _id
		self.x  = _x
		self.y  = _y

def main():

	nodes = read_tsp("ATT48.tsp")
	opt_tour = read_tour("att48.opt.tour", nodes)
	opt_tour_dist = calc_route_dist(opt_tour)

	max_step = 3000
	iterations = 1

	test_search_algo(sim_ann, nodes, max_step, iterations, opt_tour_dist)
	test_search_algo(tabu_search, nodes, max_step, iterations, opt_tour_dist)
	# test_search_algo(genetic, nodes, max_step, iterations, opt_tour_dist)

	genetic(nodes, max_step)


"""
------------------------------------------------------------------------------------------------------------------------------------------------

READ FILE FUNCTIONS

------------------------------------------------------------------------------------------------------------------------------------------------
"""

"""
NAME:   read_tsp
RETURN: List of Node objects.
DESC:   Given a filename that ends in '.tsp', it will read the file and construct a list of node objects for each node in the file.
"""
def read_tsp(filename):

	# Checks file is of correct type.
	if not filename.endswith('.tsp'): raise Exception("Filename should end with '.tsp'.")

	# Gets node data.
	f = read_file_remove_header(filename, "NODE_COORD_SECTION\n")

	# Creates a node object for each node in file, and appends to list.
	nodes = []
	for i in f:
		line = i.split()
		nodes.append(Node(line[0], int(line[1]), int(line[2])))

	return nodes


"""
NAME:   read_tour
RETURN: List of Node objects.
DESC:   Given a filename that ends in '.tour', and a list of nodes, it will read the file and construct a list of node in the tour order.
"""
def read_tour(filename, nodes):

	# Checks filename is of correct type.
	if not filename.endswith('.tour'): raise Exception("Filename should end with '.tour'.")

	# Gets tour order.
	f = read_file_remove_header(filename, "TOUR_SECTION\n")

	# Constructs ordered node list according to the specified tour in the file.
	tour = []
	for i in f:
		node_id = i.split()[0]
		for n in nodes: 
			if n.id == node_id: tour.append(n)

	return tour

"""
NAME:   read_file_remove_header
RETURN: List of strings.
DESC:   Given a filename and a string indicating the end of the header, it will return a list of all lines in the file between the header and EOF.
"""
def read_file_remove_header(filename, header_end):
	f = open(filename).readlines()
	tour_begin = f.index(header_end) + 1
	return f[tour_begin:-1]

"""
------------------------------------------------------------------------------------------------------------------------------------------------

CALCULATE ROUTE DISTANCE FUNCTIONS

------------------------------------------------------------------------------------------------------------------------------------------------
"""

"""
NAME:   calc_route_dist
RETURN: Float value
DESC:   Given a list of nodes which form a route, it will calculate the distance and return the value.
"""
def calc_route_dist(route):

	distance = 0

	for i in range(0, len(route)-1):
		distance += ped(route[i], route[i+1])
	
	distance += ped(route[-1], route[0])
	return distance

"""
NAME:   ped
RETURN: Float value
DESC:   Calulcated the pseudo-euclidean distance between two nodes.
"""
def ped(n1, n2):
	
	xd = n1.x - n2.x
	yd = n1.y - n2.y

	rij = math.sqrt( (xd*xd + yd*yd) / 10.0 )
	tij = math.ceil( rij )

	if (tij<rij): dij = tij + 1
	else: dij = tij
	return dij

"""
------------------------------------------------------------------------------------------------------------------------------------------------

UNIVERSAL FUNCTIONS

------------------------------------------------------------------------------------------------------------------------------------------------
"""

"""
NAME:   print_route
RETURN: None
DESC:   Outputs the given route. Used for debugging.
"""
def print_route(route):
	r = ""
	for i in route: r += i.id + " > "
	print(r)

"""
NAME:   test_search_algo
RETURN: None
DESC:   Runs a given search algorithm a given amount of times and outputs all the solutions given, as well as the average.
"""
def test_search_algo(algo, nodes, max_step, iterations, opt_tour_dist):

	solutions = []

	# Runs the search algorithm repeatedly for the amount of iterations specified, and stores all solutions.
	for i in range(0, iterations):
		solutions.append( algo(nodes, max_step) )

	# Checks each solution is valid, and outputs each solution distance.
	average = 0
	for i in range(0, iterations):

		dist_colour, valid_colour = "\33[31m", "\33[31m"

		d = calc_route_dist(solutions[i])
		valid = valid_route(nodes, solutions[i])

		if d <= (opt_tour_dist * 1.1): dist_colour = "\33[32m"
		elif d <= (opt_tour_dist * 1.2): dist_colour = "\33[34m"
		if valid == True: valid_colour = "\33[32m"

		average += d

		print("Iteration %d - Distance: %s%d\33[0m - Valid: %s%r\33[0m" % (i, dist_colour, d, valid_colour, valid))
	
	average /= iterations

	dist_colour = "\33[31m"
	if average <= (opt_tour_dist * 1.1): dist_colour = "\33[32m"
	elif average <= (opt_tour_dist * 1.2): dist_colour = "\33[34m"

	# Outputs the average distance obtained over all iterations.
	print("\n\nAverage Distance: %s%d\33[0m\n\n" % (dist_colour, average))

"""
NAME:   valid_route
RETURN: Boolean value
DESC: Checks that each node has been visited on the route once and only once.
"""
def valid_route(nodes, route):

	node_check = [False] * len(nodes)

	if len(route) != len(nodes): return False

	for n in route:
		i = int(n.id) - 1
		if node_check[i] == True: return False
		else: node_check[i] = True

	for i in node_check:
		if i == False: return False

	return True

"""
------------------------------------------------------------------------------------------------------------------------------------------------

SIMULATED ANNEALING FUNCTIONS

------------------------------------------------------------------------------------------------------------------------------------------------
"""

"""
NAME:   sim_ann
RETURN: List of Node objects
DESC:   The simulated annealing algorithm.
"""
def sim_ann(nodes, max_step):

	route = copy.deepcopy(nodes)

	# Generates random initial solution.
	random.shuffle(route)
	curr_dist = calc_route_dist(route)

	# Sets the best route and distance to the initial solution
	best_route = copy.deepcopy(route)
	best_dist = copy.deepcopy(curr_dist)

	for i in range(0, max_step):

		t = temperature(i/max_step)

		# Generated new route and calculates distance.
		new_route = generate_neighbour_solution(route)
		new_dist = calc_route_dist(new_route)

		# Accepts the new solution with a probability given by function P.
		if P(curr_dist, new_dist, t) > random.uniform(0, 1):
			route = copy.deepcopy(new_route)
			curr_dist = copy.deepcopy(new_dist)

		# If the new route is better than the best route, updates the best route.
		if new_dist < best_dist:
			best_route = copy.deepcopy(new_route)
			best_dist = copy.deepcopy(new_dist)

	return best_route


"""
NAME:   temperature
RETURN: Float value
DESC:   Returns the temperature value according to the schedule defined as: (initial temp)/(1 + (cooling factor ^ current step)). The less amount of steps remaining the lower the value returned.
"""
def temperature(k):
	t_0 = 1
	# a = 0.8
	# return t_0  * (a ** k)
	a = 0.7
	return t_0 / 1 + (a ** k)

"""
NAME:   P
RETURN: Float value
DESC:   Returns a probability value for selecting a new route. If the route is better 1 is retured, else the probability is given as e^(difference/temperature).
"""
def P(curr_dist, new_dist, t): 
	if new_dist < curr_dist: return 1
	else: return math.exp((curr_dist-new_dist)/t)

"""
NAME:   generate_neighbour_solution
RETURN: List of Node objects.
DESC:   Randomly reverses a section of the route to generate a neighbour solution.
"""
def generate_neighbour_solution(route):

	# Generates two random numbers, which represent indexs in the route list.
	start = random.randrange(0, len(route)-1, 1)
	end = random.randrange(start, len(route)-1, 1)

	# Flips the list between the two random indexs.
	rev = route[start:end]
	rev.reverse()

	# Returns a new route which contains the flipped list.
	return route[0:start] + rev + route[end:]


"""
------------------------------------------------------------------------------------------------------------------------------------------------

TABU SEARCH FUNCTIONS

------------------------------------------------------------------------------------------------------------------------------------------------
"""

"""
NAME:   tabu_search
RETURN: List of Node objects.
DESC:   The tabu search algorithm.
"""
def tabu_search(nodes, max_step):

	max_tabu_size = 15

	best_cand = copy.deepcopy(nodes)

	# Generates random initial solution.
	random.shuffle(best_cand )
	best_cand_dist = calc_route_dist(best_cand)

	# Sets the best route and distance to the initial solution
	best_route = copy.deepcopy(best_cand )
	best_dist = copy.deepcopy(best_cand_dist)

	# Creates tabu list and appends the current best solution.
	tabu = []
	tabu.append(best_route[:])

	for i in range(0, max_step):

		# Generates neighbourhood solutions of the current route.
		neighbourhood = get_neighbourhood(best_cand)

		# Finds the best candidate solution in the neighbourhood that is not present in the tabu list.
		best_cand = copy.deepcopy(neighbourhood[0])
		best_cand_dist = calc_route_dist(best_cand)
		for cand in neighbourhood:
			cand_dist = calc_route_dist(cand)
			if not in_tabu(tabu, cand) and (cand_dist < best_cand_dist):
				best_cand = copy.deepcopy(cand)
				best_cand_dist = calc_route_dist(best_cand)

		# Compares the best candidate solution to the current best solution, and updates the current best accordingly.
		if best_cand_dist < best_dist:
			best_route = copy.deepcopy(best_cand)
			best_dist = copy.deepcopy(best_cand_dist)

		# Updates tabu list by appending the best candiate solution. If the tabu list is above the maximum size the oldest candidate solution is removed.
		tabu.append(best_cand[:])
		if len(tabu) > max_tabu_size:
			tabu = tabu [:-1]

	return best_route

"""
NAME:   get_neighbourhood
RETURN: 2D List of Node objects.
DESC:   Generates all possible neighbour solutions of a given route by reversing the list between every two unique element pairs.
"""
def get_neighbourhood(route):

	neighbourhood = []

	# Iterates over each unique cobination of nodes in the route.
	for i in range(0, len(route)):
		for j in range(i, len(route)):

			# Reverses route between the two selected nodes.
			rev = route[i:j]
			rev.reverse()

			# Adds new route to the neighbourhood.
			new_route = route[:i] + rev + route[j:]
			neighbourhood.append(new_route[:])

	return neighbourhood

"""
NAME:   compare_routes
RETURN: Boolean value
DESC:   Iterates over two given routes and compares then. If they're the same returns the value true, else it returns false.
"""
def compare_routes(r1, r2):
	for i in range(0, len(r1)):
		if r1[i].id != r2[i].id: return False
	return True 

"""
NAME:   in_tabu
RETURN: Boolean value
DESC:   Compares a candidate route to every route in the tabu list. If the candidate is found in the tabu list the value true is returned, else false.
"""
def in_tabu(tabu, cand):
	for i in tabu:
		if compare_routes(i, cand): return True
	return False


"""
------------------------------------------------------------------------------------------------------------------------------------------------

GENETIC ALGORITHM FUNCTIONS

------------------------------------------------------------------------------------------------------------------------------------------------
"""

def genetic(nodes, max_generations):

	route = copy.deepcopy(nodes)

	pop_size = 30
	population, pop_fitness = [], []

	# Generates initial random population, and calculates fitness.
	for i in range(0, pop_size):
		random.shuffle(route)
		population.append(route[:])
		pop_fitness.append(calc_route_dist(route))

	for i in range(0, max_generations):

		p1, p2 = selection(pop_fitness)

		# new_pop = variation(population[p1], population[p2])

		# new_fitness = [0] * pop_size
		# for i in range(0, pop_size):
		# 	new_fitness[i] = calc_route_dist(new_pop[i])

		# population, pop_fitness = reproduction(population, pop_fitness, new_pop, new_fitness)

	best_ind = 0
	for i in range(0, pop_size):
		if pop_fitness[i] < pop_fitness[best_ind]:
			best_ind = i

	return population[best_ind]



def selection(pop_fitness):
	return 0, 0

def variation(p1, p2):
	return [None] * 30

def reproduction(population, pop_fitness, new_pop, new_fitness):
	return [None] * 30





		































if __name__ == "__main__":
	main()
