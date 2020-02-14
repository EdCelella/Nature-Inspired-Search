from __future__ import division
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
	iterations = 30

	optimise_parameters(nodes, max_step, iterations)

	# test_search_algo(sim_ann, nodes, max_step, iterations, opt_tour, opt_tour_dist)
	# test_search_algo(tabu_search, nodes, max_step, iterations, opt_tour, opt_tour_dist)
	# test_search_algo(genetic, nodes, max_step, iterations, opt_tour, opt_tour_dist)


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
def test_search_algo(algo, nodes, max_step, iterations, opt_tour, opt_tour_dist):

	print(opt_tour_dist)

	
	average = 0

	# Runs the search algorithm repeatedly for the amount of iterations specified, and stores all solutions.
	for i in range(0, iterations):
		solution = algo(nodes, max_step)

		# Checks each solution is valid, and outputs each solution distance.

		dist_colour, valid_colour = "\33[31m", "\33[31m"

		d = calc_route_dist(solution)
		valid = valid_route(nodes, solution)

		if d <= (opt_tour_dist * 1.1): dist_colour = "\33[32m"
		elif d <= (opt_tour_dist * 1.2): dist_colour = "\33[34m"
		if valid == True: valid_colour = "\33[32m"

		average += d

		print("Iteration %d - Distance: %s%d\33[0m - Valid: %s%r\33[0m" % (i, dist_colour, d, valid_colour, valid))

		if d <= opt_tour_dist:
			print("----------------------------")
			print_route(solution)
			print_route(opt_tour_dist)
			print(compare_routes(solution, opt_tour))
			# print("\33[34mOptimal Solution Found!\33[0m")
			print("----------------------------")
			break
	
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

def optimise_parameters(nodes, max_step, iterations):

	print("--------------------------\nSimulated Annealing - Optimisation\n--------------------------")

	a_opt, t_0_opt = 0.8, 0.1

	best_dist = calc_route_dist(sim_ann(nodes, max_step, 1, a_opt))
	for i in range(0, 10):

		a = 0.8 + float(i)/100

		print(0.8 + i/100)

		curr_dist = calc_route_dist(sim_ann(nodes, max_step, 1, a))
		print("Cooling rate %d ≈ %d" % (a, curr_dist))

		if curr_dist < best_dist: a_opt = a

	# for t in range(0.2, 1.1, 0.1):

	# 	curr_dist = calc_route_dist(sim_ann(nodes, max_step, t, 0.8))
	# 	print("Initial temperature %d ≈ %d" % (a, curr_dist))

	# 	if curr_dist < best_dist: t_0_opt = t

	print()
	print(a_opt)
	print(t_0_opt)


	# sim_ann(nodes, max_step, t_0 = 1, a = 0.7)

	return 0

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
def sim_ann(nodes, max_step, t_0 = 1, a = 0.8):

	route = copy.deepcopy(nodes)

	# Generates random initial solution.
	random.shuffle(route)
	curr_dist = calc_route_dist(route)

	# Sets the best route and distance to the initial solution
	best_route = copy.deepcopy(route)
	best_dist = copy.deepcopy(curr_dist)

	for i in range(0, max_step):

		t = temperature(i, t_0, a)

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
def temperature(k, t_0, a):
	# a = 0.8
	return t_0  * (a ** k)
	# return t_0 / 1 + (a ** k)

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
	start = random.randrange(0, len(route), 1)
	end = random.randrange(start, len(route), 1)

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
def tabu_search(nodes, max_step, max_tabu_size = 15):

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

class Individual:
	def __init__(self, _route):
		self.route = _route
		self.fitness = calc_route_dist(_route)

"""
NAME:   genetic
RETURN: List of Node objects.
DESC:   The genetic evolutionary algorithm.
"""
def genetic(nodes, max_generations, pop_size = 30, tour_size = 2, mutation_p = 0.5, elite = 0.1):

	# Sets the elitism constant for each generation.
	elite = math.ceil(pop_size * elite)

	# Generates initial random population.
	route = copy.deepcopy(nodes)
	population = []
	for i in range(0, pop_size):
		random.shuffle(route)
		population.append(Individual(route[:]))

	for i in range(0, max_generations):

		# Generates children from the current population.
		# Selects two parents via tournament selection, and applies an order crossover operator which generates two new children.
		children = []
		while len(children) < pop_size:
			p1, p2 = selection(population, pop_size, tour_size)
			c1, c2 = variation(p1.route, p2.route)
			children.append(c1)
			children.append(c2)

		# Forms now population based on elitism constant and mutation probability.
		population = reproduction(population, children, pop_size, elite, mutation_p)

	# Finds and returns the solution with the lowest distance in the final population.
	best_ind = 0
	for i in range(0, pop_size):
		if population[i].fitness < population[best_ind].fitness:
			best_ind = i
	return population[best_ind].route

"""
NAME:   selection
RETURN: Two Individual objects (two routes with fitness).
DESC:   Implementation of tournament selection.
"""
def selection(population, pop_size, tour_size):

	# Selects a subset of unique parents from the population. The amount selected is determined by the set tournament size (tour_size).
	selected = set()
	while len(selected) < tour_size:
		p = random.randrange(0, pop_size, 1)
		selected.add(p)

	# The two parents with the lowest distances in the subset are found and returned.
	p1 = selected.pop()
	p2 = selected.pop()
	for i in selected:
		if population[i].fitness < population[p1].fitness:
			if population[p1].fitness < population[p2].fitness: p2 = p1
			p1 = i
		elif population[i].fitness < population[p2].fitness: p2 = i

	return population[p1], population[p2]

"""
NAME:   variation
RETURN: Two Individual objects (two routes with fitness).
DESC:   Given two routes, produces two offspring using the order crossover operator.
"""
def variation(p1, p2):

	route_len = len(p1)

	# Selects two random cutoff points.
	start = random.randrange(0, route_len, 1)
	end = random.randrange(start, route_len, 1)

	# The section of the route between the cutoff points is retained.
	off1 = p1[start:end]
	off2 = p2[start:end]

	# Each parents route starting from the second cut off point is retained, with existing bits between the cutoff points removed.
	temp1 = remove_duplicates(p2[end:] + p2[:end], off1)
	temp2 = remove_duplicates(p1[end:] + p1[:end], off2)

	# The middle cutoff section is merged with the remaining route of the other parent. Generating two offspring.
	off1 = temp1[route_len-end:] + off1 + temp1[:route_len-end]
	off2 = temp2[route_len-end:] + off2 + temp2[:route_len-end]

	return Individual(off1), Individual(off2)

"""
NAME:   remove_duplicates
RETURN: List of Node objects.
DESC:   Removes any nodes in route 1 that are present in route 2 (utilised for ordered crossover variation).
"""
def remove_duplicates(r1, r2):

	new_r = []
	for i in r1:
		duplicate = False
		for j in r2:
			if i.id == j.id: 
				duplicate = True
				break
		if not duplicate:
			new_r.append(i)
	return new_r

"""
NAME:   reproduction
RETURN: List of Individual objects.
DESC:   Produces a new population from the parents and children.
"""
def reproduction(parents, children, pop_size, elite, mutation_p):

	# Sorts the parents and children according their fitness.
	parents.sort(key=lambda x: x.fitness)
	children.sort(key=lambda x: x.fitness)

	# The new population retains the best individuals from the parents. The amount retained is set by the 'elite' constant.
	new_pop = parents[:elite]

	# The best children are added to the new population until it is full. Each child undergoes a random mutation with probability 'mutation_p'.
	c = 0
	while(len(new_pop) < pop_size):
		c1 = mutate(children[c].route, mutation_p)
		new_pop.append(c1)

	return new_pop

"""
NAME:   mutate
RETURN: Individual object.
DESC:   Mutates a route with probabiliy 'mutation_p' by applying the 2-opt algorithm to the route.
"""
def mutate(route, mutation_p):
	if mutation_p > random.uniform(0, 1): route = generate_neighbour_solution(route) # Function defined for simmulated annealing.
	return Individual(route)

if __name__ == "__main__":
	main()
