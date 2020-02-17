from __future__ import division
import math
import random
import copy
from itertools import product
import time

# Class to store node data.
class Node:
	def __init__(self, _id, _x, _y):
		self.id = _id
		self.x  = _x
		self.y  = _y

def main():

	node_file = input("Enter ATT48.tsp filepath (including the filename). If file is in current directory leave blank): ")
	opt_file = input("Enter att48.opt.tour filepath (including the filename). If file is in current directory leave blank): ")
	run_opt = input("Do you wish to run the optimal parameter tests (input character 'y' for yes, any other for no)? ")

	# Checks .tsp filename is of correct type. If found reads file and creates list of nodes.
	try:
		# If incorrect file path is give, the file is searched for in current directory.
		if not node_file.endswith('/ATT48.tsp'): nodes = read_tsp()
		else: nodes = read_tsp(node_file)
	except:
		print("File not found.")
		raise

	# Checks .tour filename is of correct type, reads file and calculaes optimal distance.
	try: 
		# If incorrect file path is give, the file is searched for in current directory.
		if not opt_file.endswith('/att48.opt.tour'): opt_tour = read_tour(nodes)
		else: opt_tour = read_tour(nodes, opt_file)
		opt_tour_dist = calc_route_dist(opt_tour)
	except:
		# If file is not found, optimal distance is set manually.
		opt_tour_dist = 10628

	
	max_step = 3000
	iterations = 30

	if run_opt == 'y':
		optimise_parameters(nodes, max_step, iterations)

	print("\n--------------------------\nSimulated Annealing\n--------------------------\n")
	test_search_algo(sim_ann, nodes, max_step, iterations, opt_tour, opt_tour_dist)

	print("\n--------------------------\nTabu Search\n--------------------------\n")
	test_search_algo(tabu_search, nodes, max_step, iterations, opt_tour, opt_tour_dist)

	print("\n--------------------------\nGenetic Algorithm\n--------------------------\n")
	test_search_algo(genetic, nodes, max_step, iterations, opt_tour, opt_tour_dist)


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
def read_tsp(filename="ATT48.tsp"):

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
def read_tour(nodes, filename="att48.opt.tour"):

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
	
	average, average_time = 0, 0

	# Runs the search algorithm repeatedly for the amount of iterations specified, and stores all solutions.
	for i in range(0, iterations):

		start = time.time()
		solution = algo(nodes, max_step)
		end = time.time()

		# Checks each solution is valid, and outputs each solution distance.
		d = calc_route_dist(solution)
		valid = valid_route(nodes, solution)
		time_taken = end - start

		# Used to colour the distance and validity values.
		dist_colour, valid_colour = "\33[31m", "\33[31m"          # Initial colour red.
		if d <= (opt_tour_dist * 1.1): dist_colour = "\33[32m"    # If distance is within 10% of the optimal set to green.
		elif d <= (opt_tour_dist * 1.2): dist_colour = "\33[34m"  # If distance is within 20% of the optimal set to blue.
		if valid == True: valid_colour = "\33[32m"                # If route is valid set to green.

		average += d
		average_time += time_taken

		print("Iteration %d - Distance: %s%d\33[0m - Valid: %s%r\33[0m - Time: %.2f" % (i, dist_colour, d, valid_colour, valid, time_taken))
	
	average /= iterations
	average_time /= iterations

	dist_colour = "\33[31m"
	if average <= (opt_tour_dist * 1.1): dist_colour = "\33[32m"
	elif average <= (opt_tour_dist * 1.2): dist_colour = "\33[34m"

	# Outputs the average distance obtained over all iterations.
	print("\n\nAverage Distance: %s%d\33[0m\nAverage Time = %.2f\n\n" % (dist_colour, average, average_time))

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
def sim_ann(nodes, max_step, t_0 = 1, a = 0.8, seed_route=False):

	route = copy.deepcopy(nodes)

	# Generates random initial solution, unless a seed route is given (for optimisation).
	if not seed_route: random.shuffle(route)
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
def tabu_search(nodes, max_step, max_tabu_size = 15, seed_route=False):

	best_cand = copy.deepcopy(nodes)

	# Generates random initial solution, unless a seed route is provided.
	if not seed_route: random.shuffle(best_cand)
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
def genetic(nodes, max_generations, pop_size = 90, tour_size = 2, mutation_p = 0.5, elite = 0.1):

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

"""
------------------------------------------------------------------------------------------------------------------------------------------------

HYPER PARAMETER OPTIMISATION FUNCTIONS

------------------------------------------------------------------------------------------------------------------------------------------------
"""

"""
NAME:   optimise_parameters
RETURN: None
DESC:   Generates a seed route and runs each parameter optimisation function for each algorithm.
"""
def optimise_parameters(nodes, max_step, iterations):

	# Produces a seed route which is used as the start for each algorithm.
	seed_route = copy.deepcopy(nodes)
	random.shuffle(seed_route)

	# optimise_sim_ann(seed_route, max_step, iterations)
	# optimise_tabu(seed_route, max_step, iterations)
	optimise_genetic(seed_route, max_step, iterations)

"""
NAME:   optimise_sim_ann
RETURN: None
DESC:   Generates a set of hyper-parameters to test over. Outputs the average distance of the simmulated annealing algorithm given each set of hyper-parameters, and then outputs the hyper-parameters with the best results. Due to a maximum of 30 iterations for all tests these values are given as approximates.
"""
def optimise_sim_ann(seed_route, max_step, iterations):

	print("\n--------------------------\nSimulated Annealing - Optimisation\n--------------------------\n")

	# Generates lists of test values for the cooling rate and initial temerature.
	a_base, t_0_base = 0.8, 0.8
	a_list = [round(x * 0.025 + a_base, 3) for x in range(0, 5)]
	t_0_list = [round(x * 0.1 + t_0_base, 2) for x in range(0, 3)]

	# Produces list of all unique combinations of the hyper paramter test values.
	simm_ann_hyper = list(product(t_0_list, a_list))[1:]

	# Defines the amount od trials for each set of hyper-parameters.
	trials = int(iterations / len(simm_ann_hyper) + 1)

	# Calculates the average route distance using the base hyper-parameter values.
	a_opt, t_0_opt, best_dist = a_base, t_0_base, 0
	for i in range(0, trials): best_dist += calc_route_dist(sim_ann(seed_route, max_step, t_0_base, a_base, True))
	best_dist /= trials

	print("Initial Temp: %f - Cooling rate: %f ≈ %d" % (a_base, t_0_base, best_dist))

	# Iterates over the hyper parameter set.
	for i in simm_ann_hyper:

		# Calculates the average distance over the amount of given trials for each set of hyper-parameters.
		dist = 0
		for j in range(0, trials): dist += calc_route_dist(sim_ann(seed_route, max_step, i[0], i[1], True))
		dist /= trials

		print("Initial Temp: %f - Cooling rate: %f ≈ %d" % (i[0], i[1], dist))

		# Updates the best hyper-parameters if the average distance is better than the current best.
		if dist < best_dist:
			t_0_opt = i[0]
			a_opt = i[1]
			best_dist = dist

	print("\nOptimal Parameters for Simulated Annealing:\n\tCooling Rate ≈ %f\n\tInitial Temp ≈ %f\n" % (a_opt, t_0_opt))

"""
NAME:   optimise_tabu
RETURN: None
DESC:   Generates a set of hyper-parameters to test over. Outputs the average distance of the tabu search algorithm given each set of hyper-parameters, and then outputs the hyper-parameters with the best results. Due to a maximum of 30 iterations for all tests these values are given as approximates.
"""
def optimise_tabu(seed_route, max_step, iterations):

	print("\n--------------------------\nTabu Search - Optimisation\n--------------------------\n")

	# Produces a list of different tabu list sizes to test.
	tabu_hyper = [x * 10 for x in range(1, 10)]

	# Calculates the amount of trials per tabu list size allowed within the maxiumum 30 trials.
	trials = int(iterations / len(tabu_hyper))

	# Calculates the average route distance over the amount of trials for the base tabu list size.
	best_dist = calc_route_dist(tabu_search(seed_route, max_step, tabu_hyper[0], True))
	tabu_opt = tabu_hyper[0]

	print("Tabu list size: %d ≈ %d" % (tabu_opt, best_dist))

	# Iterates over the different tabu list sizes.
	for i in tabu_hyper[1:]:

		# Calculates the average distance obtained for each tabu list size.
		dist = 0
		for j in range(0,trials): dist += calc_route_dist(tabu_search(seed_route, max_step, i, True))
		dist /= trials

		print("Tabu list size: %d ≈ %d" % (i, dist))

		# Updates the best tabu list size if the average distance is better than the current best.
		if dist < best_dist:
			tabu_opt = i
			best_dist = dist

	print("\nOptimal Parameters for Tabu Search:\n\tTabu List Size ≈ %d\n" % (tabu_opt))

"""
NAME:   optimise_Genetic
RETURN: None
DESC:   Generates a set of hyper-parameters to test over. Outputs the average distance of the genetic algorithm given each set of hyper-parameters, and then outputs the hyper-parameters with the best results. Due to a maximum of 30 iterations for all tests these values are given as approximates.
"""
def optimise_genetic(seed_route, max_step, iterations):

	print("\n--------------------------\nGenetic Algorithm - Optimisation\n--------------------------\n")

	tour_size = 2
	pop_size_base, mutation_p_base, elite_base = 30, 0.3, 0

	pop_size_list = [x for x in range(30,91,30)]
	mutation_list = [round(mutation_p_base + (x * 0.2), 1) for x in range(0, 3)]
	elite_list    = [round(elite_base + (x * 0.1), 1) for x in range(0, 3)]

	genetic_hyper = list(product(pop_size_list, mutation_list, elite_list))[1:]

	# Calculates the average route distance using the base hyper-parameter values.
	pop_size_opt, mutation_p_opt, elite_opt = pop_size_base, mutation_p_base, elite_base
	best_dist = calc_route_dist(genetic(seed_route, max_step, pop_size_opt, tour_size, mutation_p_opt, elite_opt))

	print("Population size: %d - Mutation probability: %.1f - Elite percentage: %.1f ≈ %d" % (pop_size_opt, mutation_p_opt, elite_opt, best_dist))

	# Iterates over the hyper parameter set.
	for i in genetic_hyper:

		# Calculates the distance for the current hyper-parameter set.
		dist = calc_route_dist(genetic(seed_route, max_step, i[0], tour_size, i[1], i[2]))

		print("Population size: %d - Mutation probability: %.1f - Elite percentage: %.1f ≈ %d" % (i[0], i[1], i[2], dist))

		# Updates the best hyper-parameters if the average distance is better than the current best.
		if dist < best_dist:
			pop_size_opt, mutation_p_opt, elite_opt = i[0], i[1], i[2]
			best_dist = dist

	print("\nOptimal Parameters for Genetic Algorithm:\n\tPopulation Size ≈ %d\n\tMutation Probability ≈ %.1f\n\tElite Percentage: %.1f\n" % (pop_size_opt, mutation_p_opt, elite_opt))



if __name__ == "__main__":
	main()
