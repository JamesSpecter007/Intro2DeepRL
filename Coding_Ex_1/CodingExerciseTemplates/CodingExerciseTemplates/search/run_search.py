import config
import environment
import util


class Agent():
    def __init__(self):
        self.env = environment.Environment()
        self.graph = util.Graph(self.env)

    #implementHere
    def depth_first_search(self):
        # TODO initialize a stack
        stack = util.Stack()
        # TODO get the start state of the environment and push it on the stack
        start_state = self.env.get_start_state()
        stack
        # TODO initialize a ClosedList called closed_list
        closed_list = util.ClosedList()
        while stack.get_number_of_entries() > 0:
            # TODO get the first element of the stack
            s = stack.pop()
            print("Expand node {}".format(s))
            # TODO if the closed_list does not contain state s already
            if closed_list.contains(s) is False:
                # TODO add state s to the closed_list
                closed_list.contains(s)
                # TODO for every child_tuple in the list of successors from s
                for child_tuple in self.env.get_successors(s):
                    # TODO extract the child, action, reward from the variable child_tuple
                    child, action, reward = child_tuple
                    # TODO add the child and parent node s to the graph
                    self.graph.add(child=child, parent=s)
                    # TODO if child node is goal node
                    if self.env.is_goal(child):
                        # TODO return actual solution from the graph with the actual child as input
                        return self.graph.get_actual_solution(child)
                    # TODO push the child on the stack
                    stack.push(child)
                    print("Add node {} to stack".format(child))

    # implementHere
    def breadth_first_search(self):
        # TODO implement breadth first search algorithm


    # implementHere
    def uniform_cost_search(self):
        # TODO implement uniform cost search algorithm



    def run_search_algorithm(self, algorithm):
        """
        Function calls the correct algorithm depending on the command args
        """
        if algorithm == "BFS":
            solution = self.breadth_first_search()
        elif algorithm == "DFS":
            solution = self.depth_first_search()
        elif algorithm == "UCS":
            solution = self.uniform_cost_search()
        else:
            raise Exception("Wrong algorithm determined")

        return solution



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = config.parser.parse_args()
    search_algorithm = args.search_algorithm
    agent = Agent()
    solution = agent.run_search_algorithm(search_algorithm)

    print("Your solution when using {} is the action sequence {}.".format(args.search_algorithm, solution))

