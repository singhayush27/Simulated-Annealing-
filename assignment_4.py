import random
import sys
from copy import deepcopy
from datetime import datetime
from math import exp
import numpy


# Keeps track of visited states
visited_state = {}
# Keeps track of parent state
parent_state = {}


"""This function is used to convert string format to matrix format"""
def string_to_matrix(state):
    grid = [[0 for row in range(3)] for column in range(3)]
    grid_state = state
    for row in range(3):
        for column in range(3):
            grid[row][column] = int(grid_state[row * 3 + column])
    return grid


"""This function is used to convert matrix to string format"""
def matrix_to_string(grid):
    string_format = ""
    for row in range(3):
        for column in range(3):
            string_format += str(grid[row][column])
    return string_format

"""This function is used to replace character 'T' and 'B' from the given string"""
def character_replacement_of_T_B(txt):
    txt = txt.replace("T", "")
    txt = txt.replace("B", "0")
    return txt


"""Creating a class for state handling"""
class State:
    def __init__(self, stateInfo, heuristic_value=0):
        self.puzzleState = stateInfo
        self.hvalue = heuristic_value

    """This function is used to find all the successors for a given configuration"""
    def find_all_successor(self, heuristic_choice, final_config):
        x = [1, -1, 0, 0]
        y = [0, -0, 1, -1]

        puzzle_grid = string_to_matrix(self.puzzleState)
        for row in range(3):
            for column in range(3):
                if puzzle_grid[row][column] == 0:
                    blank_x = row
                    blank_y = column
                    break
        
        """Maintaining a list for successor grid"""
        successor_grid = []
        for (move_in_x, move_in_y) in zip(x, y):
            if 0 <= blank_x + move_in_x < 3 and 0 <= blank_y + move_in_y < 3:
                successor_puzzle_grid = deepcopy(puzzle_grid)
                temp = successor_puzzle_grid[blank_x + move_in_x][blank_y + move_in_y]
                successor_puzzle_grid[blank_x + move_in_x][blank_y + move_in_y] = 0
                successor_puzzle_grid[blank_x][blank_y] = temp
                new_state = matrix_to_string(successor_puzzle_grid)
                successor_grid.append(
                    State(new_state, Heuristic(final_config).heuristic_estimation(
                        new_state, heuristic_choice))
                )

        return successor_grid

    """Checking for equality"""
    def __eq__(self, other):
        return self.puzzleState == other.puzzleState

    """Checking for less than comparison"""
    def __lt__(self, other):
        return self.hvalue < other.hvalue


"""Created a class for simulated annealing"""
class simulated_annealing:
    def __init__(self, intial_state, goal_state, temperature):
        self.temperature = temperature
        self.max_temperature = temperature
        self.initial_config = intial_state
        self.final_config = goal_state
        self.step = 0.0000001
        self.x = -temperature / 5
        self.explored_states = 0
        self.admissible = True
        sys.setrecursionlimit(181440)


    """Function to obtain change in energy"""
    def energy_difference(self, current, new):
        return -1 * (current.hvalue - new.hvalue)


    """This Function is used to choose for temperature drop function"""
    def temperature_function_selection(self, choice):
        return {
            1: self.linear_strategy(),
            2: self.random_strategy(),
            3: self.negative_exponential(),
            4: self.positive_exponential(),
        }[choice]


    """This function is used to solve the eight puzzle problem"""
    def puzzle_solve(self, current_state, heuristic_choice, cooling_choice):
        stack = [current_state]
        while len(stack) != 0:
            current_state = stack.pop()
            print(self.explored_states, current_state)
            if current_state == self.final_config:
                self.explored_states += 1
                out = output_results(
                    self.initial_config, "simulated_annealing_output.txt", parent_state, self.explored_states
                )
                out.write_output_path(current_state, self.admissible)
                print("******** Reached Goal state ********")
                return 0
            elif current_state in visited_state:
                continue
            else:
                self.temperature_function_selection(cooling_choice)
                self.explored_states += 1
                visited_state[current_state] = 1
                state = State(
                    current_state, Heuristic(self.final_config).heuristic_estimation(
                        current_state, heuristic_choice)
                )
                neighbours = state.find_all_successor(
                    heuristic_choice, self.final_config)
                neighbours.sort()
                cur = 0
                sz = len(neighbours)
                li = []
                mark = [0] * sz
                count = 0
                while count < len(neighbours):
                    heuristic_value = Heuristic(self.final_config).heuristic_estimation(
                        neighbours[cur].puzzleState, heuristic_choice
                    )
                    if state.hvalue > heuristic_value + 1:  # Monotonicity implies admissibility
                        self.admissible = False

                    e = self.energy_difference(state, neighbours[cur])
                    if mark[cur] == 1:
                        cur = (cur + 1) % sz
                        continue
                    if neighbours[cur].puzzleState in visited_state:
                        mark[cur] = 1
                        count += 1
                    elif e <= 0:
                        mark[cur] = 1
                        count += 1
                        parent_state[neighbours[cur].puzzleState] = current_state
                        li.append(neighbours[cur].puzzleState)
                    elif exp(-e / self.temperature) < random.uniform(0, 1):
                        mark[cur] = 1
                        count += 1
                        parent_state[neighbours[cur].puzzleState] = current_state
                        li.append(neighbours[cur].puzzleState)
                    cur = (cur + 1) % sz
                li.reverse()
                stack.extend(li)

        return 1

    def linear_strategy(self):
        self.temperature = abs(self.max_temperature + self.x)
        self.x += self.step

    def random_strategy(self):
        self.temperature = random.uniform(
            0, 1) * abs(self.max_temperature + self.x)
        self.x += self.step

    def negative_exponential(self):
        self.temperature = exp(-1 * self.x) * self.max_temperature
        self.x += self.step

    def positive_exponential(self):
        self.temperature = exp(self.x) * self.max_temperature
        self.x += self.step


"""Created a class for Heuristic"""
class Heuristic:
    def __init__(self, goal_state="123456780"):
        self.goal_state = goal_state

    """This function is used to select displaced tile heuristics """
    def displaced_tile_heuristics(self, state):
        current_puzzle_grid = string_to_matrix(state)
        goal_puzzle_grid = string_to_matrix(self.goal_state)
        heuristic_value = 0
        for row in range(3):
            for column in range(3):
                if current_puzzle_grid[row][column] != goal_puzzle_grid[row][column]:
                    heuristic_value += 1
                if (
                    current_puzzle_grid[row][column] == 0
                    and current_puzzle_grid[row][column] != goal_puzzle_grid[row][column]
                    and tile_inclusion is False
                ):
                    heuristic_value -= 1
        return heuristic_value


    """This function is used to select manhattan heuristics"""
    def manhattan_heuristic(self, state):
        current_puzzle_grid = string_to_matrix(state)
        goal_puzzle_grid = string_to_matrix(self.goal_state)
        current_grid_coordinate = numpy.arange(18).reshape((9, 2))

        for row in range(3):
            for column in range(3):
                current_grid_coordinate[current_puzzle_grid[row][column]][0] = row
                current_grid_coordinate[current_puzzle_grid[row][column]][1] = column

        """Initiating a variable for heuristic value with 0"""
        heuristic_value = 0
        for row in range(3):
            for column in range(3):
                if goal_puzzle_grid[row][column] != 0:
                    heuristic_value += abs(row - current_grid_coordinate[goal_puzzle_grid[row][column]][0]) + abs(
                        column - current_grid_coordinate[goal_puzzle_grid[row][column]][1]
                    )
                if goal_puzzle_grid[row][column] == 0 and tile_inclusion:
                    heuristic_value += abs(row - current_grid_coordinate[goal_puzzle_grid[row][column]][0]) + abs(
                        column - current_grid_coordinate[goal_puzzle_grid[row][column]][1]
                    )
        return heuristic_value

    """This Function is used for getting heuristic estimation on the choice selection"""
    def heuristic_estimation(self, state, heuristic_choice):
        return {
            1: self.displaced_tile_heuristics(state),
            2: self.manhattan_heuristic(state),
            3: 3 * self.displaced_tile_heuristics(state) - 2 * self.manhattan_heuristic(state),
            4: 2 * self.displaced_tile_heuristics(state) * self.manhattan_heuristic(state),
        }[heuristic_choice]


"""This class is created to get the results in orderly manner"""
class output_results:
    def __init__(self, start_state, file_name, parent_map, explored_states):
        self.start_state = start_state
        self.file_name = file_name
        self.parent = parent_map
        self.explored_states = explored_states
        self.path_length = 0
        sys.setrecursionlimit(181440)


    """This function is used to obtain a path from start state to goal state"""
    def write_output_path(self, puzzle_state, admissibilty):
        stack = [puzzle_state]

        while puzzle_state != self.start_state:
            stack.append(self.parent[puzzle_state])
            puzzle_state = self.parent[puzzle_state]
            self.path_length += 1
        stack.pop()
        with open(self.file_name, "a") as f:
            if admissibilty:
                f.write("This heuristic is admissible!\n")
            else:
                f.write("This heuristic is not admissible!\n")
            f.write("Total Number of state explored : {}\n".format(
                str(self.explored_states)))
            f.write("Search Status : Successful \n(sub) optimal Path length: {} \n".format(
                str(self.path_length)))
            f.write("Sub Optimal Path \n")
            f.write(puzzle_state[:3] + "\n" + puzzle_state[3:6] +
                    "\n" + puzzle_state[6:] + "\n" + " v" + "\n")
            f.close()

        while len(stack) != 0:
            puzzle_state = stack.pop()
            with open(self.file_name, "a") as f:
                f.write(puzzle_state[:3] + "\n" + puzzle_state[3:6] +
                        "\n" + puzzle_state[6:] + "\n" + " v" + "\n")
                f.close()


"""This function is used to write output in external file"""
def write_output_in_file(file, heuristic_choice, start_state, goal_state, cooling_function, temp):
    file.write("Chosen Heuristic: ")
    if heuristic_choice == 1:
        file.write("Number of tiles displaced from their final position: \n")
    elif heuristic_choice == 2:
        file.write("Total Manhattan distance: \n")
    else:
        file.write(
            "Total Manhattan distance * Number of tiles displaced from their final position:\n")
    file.write("Start State : \n")
    file.write(start_state[:3] + "\n" +
               start_state[3:6] + "\n" + start_state[6:] + "\n")
    file.write("Goal State : \n")
    file.write(goal_state[:3] + "\n" + goal_state[3:6] +
               "\n" + goal_state[6:] + "\n")
    if cooling_function == 1:
        file.write("Cooling Function : Linear Function \n")
    elif cooling_function == 2:
        file.write("Cooling Function : Random Strategy \n")
    elif cooling_function == 3:
        file.write("Cooling Function : Negative Exponential Function \n")
    else:
        file.write("Cooling Function : Positive Exponential function \n")
    file.write("TMAX : {}\n".format(str(temp)))
    file.close()


"""This function is used to initiate our program"""
def start(starting_state, goal_state, file):
    print("Enter your choice depending on the heuristic :")
    print("1. h1(n) = Number of tiles displaced")
    print("2. h2(n) = Total Manhattan distance")
    print("3. h3(n) = h1(n) * h2(n)")

    heuristic_choice = int(input("Enter your choice between 1 to 3\n"))

    blank_tile_inclusion = int(
        input("For considering blank tile, press 1 else 0\n"))
    if blank_tile_inclusion:
        Heuristic.tile_inclusion = True
    print(
        "Choose the cooling function:\n 1.Linear Function \n 2.Random Strategy \n"
        " 3.Negative Exponential Function \n 4.Positive Exponential function\n"
    )
    cooling_function_choice = int(input("Enter your choice:\n"))
    initial_temperature = int(input("Enter the value of Tmax: \n"))

    write_output_in_file(file, heuristic_choice, starting_state, goal_state,
                  cooling_function_choice, initial_temperature)

    """Starting timer for calculation of execution time"""
    start_time = datetime.now()
    puzzle_solver = simulated_annealing(
        starting_state, goal_state, initial_temperature)
    search_status = puzzle_solver.puzzle_solve(
        starting_state, heuristic_choice, cooling_function_choice)
    file = open("simulated_annealing_output.txt", "a")
    if search_status == 1:
        file.write("Search Status : Failed\n")
    file.write("Time Taken : {} ".format(str(datetime.now() - start_time)))
    file.close()


"""By default keeping the tile inclusion as false"""
tile_inclusion = False


"""From here our main function starts"""
if __name__ == "__main__":
    starting_state = ""
    goal_state = ""

    """Getting input from external file in write format"""
    file = open("simulated_annealing_output.txt", "w+")

    """Converting start state in string format"""
    with open("StartState") as f:
        for line in f:
            line = line.strip()
            line = line.replace(" ", "")
            starting_state += line

    """Converting goal state in string format"""
    with open("GoalState") as f:
        for line in f:
            line = line.strip()
            line = line.replace(" ", "")
            goal_state += line

    starting_state = character_replacement_of_T_B(starting_state)
    goal_state = character_replacement_of_T_B(goal_state)

    """Start execution of function for solving the puzzle problem"""
    start(starting_state, goal_state, file)
