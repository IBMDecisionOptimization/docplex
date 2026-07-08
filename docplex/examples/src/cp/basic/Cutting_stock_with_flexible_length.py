'''
Problem description

The problem solved in this example is a two-dimensional (2D) Two-Stage Cutting Stock Problem with Guillotine Cuts
and flexible length. This problem involves cutting large stock sheets into smaller, ordered items using a 2-stage
process to optimize material utilization. The aim is to fulfill specific order requirements by dividing each stock
into levels and allocating ordered items within each level, while minimizing unused material. All cuts are guillotine
cuts, which span the entire width or length of the stock.
The problem is described in the following paper :
"Luo, Yiqing L., and J. Christopher Beck. "Packing by Scheduling: Using Constraint Programming to Solve a Complex 2D
Cutting Stock Problem."

Problem Constraints:
    1. **Level Width Constraint**: This constraint ensures that each level within a stock can accommodate the width
    of all items assigned to it without exceeding the stock’s total width. The constraint is only active for levels
    that have items allocated, ensuring that material usage is limited to what's needed for active levels and avoiding
    unnecessary use of stock width.

    2. **Stock Length Constraint**: This constraint guarantees that the sum of lengths of all levels within a stock
    remains within the stock’s overall length. It’s applied only to stocks that are in use, making sure that material
    is efficiently allocated and that unused stock remains unaffected, thus optimizing space.

    3. **Stock Area Constraint**: This constraint controls the total area of each order placed within a stock, ensuring
    that the area assigned meets each order’s area requirements range. It enforces efficient use of stock by balancing
	order needs with stock capacity, reducing waste while fully satisfying each order’s area specifications.

    4. **Level Length Constraint**: The length of each level within a stock must be appropriately sized to fit the
    items allocated to it. This is achieved by enforcing a range bounds on the level's length:

    Each level in a stock must be at least as long as the largest or the minimum required length of each order.
    Similarly, each level in a stock must be at most as long as the smallest of the maximum required length of each
    order.

    This ensures that all items fit within the level without exceeding its boundaries.

Objective:
	The objective is to minimize unused material by reducing the waste area within each stock after fulfilling all
	order requirements. This optimization ensures efficient utilization of resources and cost savings.

Secondary Considerations:
	- Guillotine cuts are mandatory, meaning each cut goes entirely from one stock edge to the opposite edge, either
	  horizontally or vertically, to align with practical cutting machinery requirements.
	- The solution also seeks to minimize the number of stock pieces used, promoting economical use of material.

'''


import docplex.cp.model as cp
from sys import stdout

import numpy as np
import math
import random
import time

# Parameters
TIME_LIMIT = 60
DISPLAY_SOLUTION = 0
LOG_VERBOSITY = 'Terse'

# Objective weights
ALPHA = 1
BETA = 1

# Precision for integer approximation of continuous variables
PRECISION = 100


# Randomly generation of an instance
def generate_instance(random_seed, num_stock, num_orders):
    random_numbers = np.random.default_rng(seed=random_seed)
    # Length of each stock
    L = random_numbers.integers(170, 200) 
    # Width of each stock 
    W = random_numbers.integers(100, 160, size=num_stock)  
    # Width of each item in orders respectively
    O_w = random_numbers.integers(20, 60, size=num_orders)  
    # Length of each item in orders respectively
    O_l = [np.sort(random_numbers.integers(30, 70, size = 2)) for i in range (num_orders)] 
    O_an = [np.sort(random_numbers.integers(1, num_orders // 2 + 2, size = 2)) for i in range (num_orders)]
    # Area of each order respectively
    O_a = [[O_an[i][0] * O_w[i] * O_l[i][0], O_an[i][1] * O_w[i] * O_l[i][1]] for i in range(num_orders)]  
    # Number of cuts of the cutting machine
    Cutters = math.floor(L/(min(map(min, O_l)))) 
    return Cutters, num_stock, num_orders, L, W, O_w, O_l, O_a

# Define general and restricted domains
def y_i_domain(dom_restriction, data):
    # Recover data
    (NUM_CUTS, NUM_STOCK, NUM_ORDERS, STOCK_LENGTH, STOCK_WIDTH, ORDER_WIDTH, ORDER_LENGTH, ORDER_AREA) = data
    if dom_restriction == "restricted":
        return ((0, (int(min(map(min, ORDER_LENGTH))) * PRECISION, int(max(map(max, ORDER_LENGTH))) * PRECISION)))
    else:
        return ((0, (1, int(max(map(max, ORDER_LENGTH))) * PRECISION)))
    


def create_decision_variables(dom_restriction, mdl, data):
    # Recover data
    (NUM_CUTS, NUM_STOCK, NUM_ORDERS, STOCK_LENGTH, STOCK_WIDTH, ORDER_WIDTH, ORDER_LENGTH, ORDER_AREA) = data

    MAX_LEVEL = NUM_CUTS + 1
    eta = math.ceil(max(STOCK_WIDTH) / min(ORDER_WIDTH))  # Maximum number of partition in each level

    # Maximum number of partition in a level
    maxPartition = math.ceil(max(STOCK_WIDTH) / min(ORDER_WIDTH))

    # Order allocation variables
    x = [[[mdl.integer_var(0, maxPartition, "x[{}][{}][{}]".format(i, j, k)) for k in range(0, NUM_STOCK)]
          for j in range(0, MAX_LEVEL)] for i in range(0, NUM_ORDERS)]
    
    # Level length: Float variables 
    y = [[mdl.float_var(0, max(map(max, ORDER_LENGTH)), "y[{}][{}]".format(j, k))
          for k in range(0, NUM_STOCK)]
         for j in range(0, MAX_LEVEL)]
    
    # Scaled level length: Integer variables 
    y_i_domain = (0, (int(min(map(min, ORDER_LENGTH))) * PRECISION, int(max(map(max, ORDER_LENGTH))) * PRECISION))
    y_integer = [
        [mdl.integer_var(domain=y_i_domain, name="y_integer[{}][{}]".format(j, k))
         for k in range(0, NUM_STOCK)]
        for j in range(0, MAX_LEVEL)]

    # Connect level length(y) and scaled level length(y_integer)
    for k in range(NUM_STOCK):
        for j in range(MAX_LEVEL):
            mdl.add(y[j][k] == y_integer[j][k] / PRECISION)

    # Binary variable s[k][j] is true (1) if stock level j on stock k is used, false (0) otherwise
    s = [[mdl.binary_var("s[{}][{}]".format(j, k)) for k in range(0, NUM_STOCK)] for j in range(0, MAX_LEVEL)]
    # Binary variable c[k] is true (1) if stock k is used, false (0) otherwise
    c = [mdl.binary_var("c[{}]".format(k)) for k in range(0, NUM_STOCK)]

    return x, y, s, c


def create_cutting_stock_model(dom_restriction, mdl, indicator, vars, data):
    # Recover variables
    (x, y, s, c) = vars

    # Recover data
    (NUM_CUTS, NUM_STOCK, NUM_ORDERS, STOCK_LENGTH, STOCK_WIDTH, ORDER_WIDTH, ORDER_LENGTH, ORDER_AREA) = data

    MAX_LEVEL = NUM_CUTS + 1

    # Recover variables 
    (x, y, s, c) = vars

    # s_jk (binary variable definition) 
    for j in range(MAX_LEVEL):
        for k in range(NUM_STOCK):
            mdl.add(s[j][k] == mdl.any([x[i][j][k] > 0 for i in range(NUM_ORDERS)]))

    # c_k (binary variable definition) 
    for k in range(NUM_STOCK):
        mdl.add(c[k] == mdl.any([s[j][k] == 1 for j in range(MAX_LEVEL)]))

    # 1-Total width of items in each level should be within the width of the stock 
    for j in range(MAX_LEVEL):
        for k in range(NUM_STOCK):
            mdl.add(sum([x[i][j][k] * ORDER_WIDTH[i] for i in range(NUM_ORDERS)]) <= STOCK_WIDTH[k] * s[j][k])

    # 2-Total length of levels in each stock should be within length of the stock 
    for k in range(NUM_STOCK):
        mdl.add(sum([y[j][k] for j in range(MAX_LEVEL)]) <= STOCK_LENGTH * c[k])

    # 3-Total area of orders in each stock
    for i in range(NUM_ORDERS):
        mdl.add(mdl.range(sum(x[i][j][k] * y[j][k] for j in range(MAX_LEVEL) for k in range(NUM_STOCK)),
                      min(ORDER_AREA[i]) / ORDER_WIDTH[i],
                      max(ORDER_AREA[i]) / ORDER_WIDTH[i]))

    # 4-Level bounds depend on orders assigned to that level
    for j in range(MAX_LEVEL):
        for k in range(NUM_STOCK):
            for i in range(NUM_ORDERS):
                mdl.add((x[i][j][k] >= 1) <= mdl.range(y[j][k], min(ORDER_LENGTH[i]), max(ORDER_LENGTH[i])))

    # 5-Optional symmetry-breaking constraints.
    # Breaking symmetries can make solution finding more complex (because this removes solutions
    # to the model) but helps to have optimality proofs faster.
    # 5a - Ordering levels by length on each stock
    for k in range(NUM_STOCK):
        for j in range(NUM_CUTS):
            mdl.add(y[j][k] >= y[j + 1][k])

    # 5b - Order used stock first.
    for k in range(NUM_STOCK):
        for k_ in range(k+1, NUM_STOCK):
            if STOCK_WIDTH[k] == STOCK_WIDTH[k_]:
                mdl.add(c[k] >= c[k_])
    
    # Objective function setup 
    term1 = STOCK_LENGTH * mdl.sum(STOCK_WIDTH[k] * c[k] for k in range(NUM_STOCK))
    term2 = mdl.sum(
        ORDER_WIDTH[i] * x[i][j][k] * y[j][k] for i in range(NUM_ORDERS) for j in range(MAX_LEVEL) for k in
        range(NUM_STOCK))

    objective = ALPHA * term1 - BETA * term2
    # Minimize the total wastage
    mdl.minimize(objective)


def solveDisplay_singleData(dom_restriction, indicator, data, time_lmt):
    # Recover data
    (NUM_CUTS, NUM_STOCK, NUM_ORDERS, STOCK_LENGTH, STOCK_WIDTH, ORDER_WIDTH, ORDER_LENGTH, ORDER_AREA) = data
    MAX_LEVEL = NUM_CUTS + 1
    mdl = cp.CpoModel('2d_2s_cutting_stock_problem')
    vars = create_decision_variables(dom_restriction, mdl, data)
    create_cutting_stock_model(dom_restriction, mdl, indicator, vars, data)
    sol = mdl.solve(LogVerbosity=LOG_VERBOSITY, TimeLimit=time_lmt)
    # Recover variables
    (x, y, s, c) = vars
    cvalues = [sol[v] for v in c]
    svalues = [[sol[v] for v in d1] for d1 in s]
    yvalues = np.array([[sol[v] for v in d1] for d1 in y])
    xvalues = [[[sol[v] for v in d2] for d2 in d1] for d1 in x]
    stdout.write('\n')
    obj = sol.get_objective_values()
    time_sl = sol.get_solve_time()
    if not obj:
        print("Search terminated by limit, no solution found.")
    if obj:
        print("Solution found")
        print("~~~~~~~~~~~~~~")
        print("Area of trim loss: {}".format(obj[0]))
        print("Solve time: ", sol.get_solve_time())
        print()
        print("Cutting strategy")
        print("~~~~~~~~~~~~~~~~~~~~")
        for k in range(NUM_STOCK):
            if cvalues[k] != 0:
                print("Levels of stock {}:".format(k + 1))
                for j in range(MAX_LEVEL):
                    if svalues[j][k] != 0:
                        print(" Level", j, "has length", yvalues[:, k][j], ", orders assigned :")
                        for i in range(NUM_ORDERS):
                            if xvalues[i][j][k] == 1:
                                print(" . 1 order of type", i)
                            elif xvalues[i][j][k] > 1:
                                print(" .", xvalues[i][j][k], "orders of type", i)


data = generate_instance(3, 3, 5)
solveDisplay_singleData("restricted", "with_symmetry", data, 120)

