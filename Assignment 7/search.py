import util
from sudoku import SudokuSearchProblem
from maps import MapSearchProblem

################ Node structure to use for the search algorithm ################
class Node:
    def __init__(self, state, action, path_cost, parent_node, depth):
        self.state = state
        self.action = action
        self.path_cost = path_cost
        self.parent_node = parent_node
        self.depth = depth

########################## DFS for Sudoku ########################
## Choose some node to expand from the frontier with Stack like implementation
def sudokuDepthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Return the final values dictionary, i.e. the values dictionary which is the goal state  
    """

    def convertStateToHash(values):
        """ 
        values as a dictionary is not hashable and hence cannot be used directly in the explored/visited set.
        This function changes values dict into a unique hashable string which can be used in the explored set.
        You may or may not use this
        """
        l = list(sorted(values.items()))
        modl = [a+b for (a, b) in l]
        return ''.join(modl)

    ## YOUR CODE HERE
    
    frontier = util.Stack()
    frontier.push(Node(problem.getStartState(), None, 0, None, 0))
    explored = set()

    while not frontier.isEmpty():
        node = frontier.pop()
        if convertStateToHash(node.state) not in explored:
            if problem.isGoalState(node.state):
                return node.state
            explored.add(convertStateToHash(node.state))
            for child_state, action, step_cost in problem.getSuccessors(node.state):
                child = Node(child_state, action, node.path_cost + step_cost, node, node.depth + 1)
                if convertStateToHash(child.state) not in explored:
                    frontier.push(child)

    # util.raiseNotDefined()

######################## A-Star and DFS for Map Problem ########################
## Choose some node to expand from the frontier with priority_queue like implementation

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def heuristic(state, problem):
    # It would take a while for Flat Earther's to get accustomed to this paradigm
    # but hang in there.

    """
        Takes the state and the problem as input and returns the heuristic for the state
        Returns a real number(Float)
    """

    current_node_attr, end_node_attr = problem.G.node[state], problem.G.node[problem.end_node]
    current_node_coordinates = ((current_node_attr['x'], 0, 0), (current_node_attr['y'], 0, 0))
    end_node_coordinates = ((end_node_attr['x'], 0, 0), (end_node_attr['y'], 0, 0))
    return util.points2distance(current_node_coordinates, end_node_coordinates)

    # util.raiseNotDefined()

def AStar_search(problem, heuristic=nullHeuristic):

    """
        Search the node that has the lowest combined cost and heuristic first.
        Return the route as a list of nodes(Int) iterated through starting from the first to the final.
    """

    def solution(node):
        path = [node]
        while path[-1].parent_node is not None:
            path.append(path[-1].parent_node)
        path.reverse()
        path = map(lambda node: node.state, path)
        return path

    frontier = util.PriorityQueue()
    node = Node(problem.getStartState(), None, 0, None, 0)
    frontier.push(node, heuristic(node.state, problem))
    explored = set()
    path_cost = {node.state: node.path_cost}

    while not frontier.isEmpty():
        node = frontier.pop()
        if node.state not in explored:
            if problem.isGoalState(node.state):
                return solution(node)
            explored.add(node.state)
            for child_state, action, step_cost in problem.getSuccessors(node.state):
                child = Node(child_state, action, node.path_cost + step_cost, node, node.depth + 1)
                if child.state not in explored:
                    if child.state in path_cost and child.path_cost > path_cost[child.state]:
                        continue
                    path_cost[child.state] = child.path_cost
                    frontier.update(child, child.path_cost + heuristic(child.state, problem))

    # util.raiseNotDefined()