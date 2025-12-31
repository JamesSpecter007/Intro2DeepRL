import numpy as np

class Environment:
    """
    This class defines the environment the agent is acting in
    """
    def __init__(self, args):
        self.num_rows = 3
        self.num_cols = 4
        self.win_state = [(3, 2)]
        self.lose_state = [(3, 1)]
        self.start_state = (0, 0)
        self.list_of_blocked_states = [(1, 1)]
        self.board = np.zeros([self.num_cols, self.num_rows])
        self.state = self.start_state
        self.counter_epoch = 0
        self.number_epochs = args.number_epochs
        np.random.seed(0)


    def giveReward(self, state):
        """
        returns the reward for entering a certain state
        :param state: state to enter
        :return: reward
        """
        if state in self.win_state:
            return 1
        elif state in self.lose_state:
            return -1
        else:
            return 0

    def is_terminal_state(self, state=None):
        """
        checks if the state is a terminal state
        :param state: state to check. If state is None we assume the actual state known to the environment as state
        :return: boolean indicating if state is terminal state or not
        """
        if state is None:
            state = self.state
        if (state in self.win_state) or (state in self.lose_state):
            return True
        else:
            return False


    def transition(self, action, state=None):
        """
        defines the transition from a state+action to a new state and transition probabilities.
        This function also calculates the possibility that the agent performs a different action as intended
        :param action: action the agent takes
        :param state:  state the agent is in
        :return: next state, reward
        """
        if state is None:
            state = self.state
        if action == "up":
            action_transit = np.random.choice(["up", "left", "right"], p=[0.8, 0.1, 0.1])
        elif action == "down":
            action_transit = np.random.choice(["down", "left", "right"], p=[0.8, 0.1, 0.1])
        elif action == "left":
            action_transit = np.random.choice(["left", "up", "down"], p=[0.8, 0.1, 0.1])
        elif action == "right":
            action_transit = np.random.choice(["right", "up", "down"], p=[0.8, 0.1, 0.1])
        else:
            raise Exception("Wrong action")

        if action_transit == "up":
            nxtState = (state[0], state[1] + 1)
        elif action_transit == "down":
            nxtState = (state[0], state[1] - 1)
        elif action_transit == "left":
            nxtState = (state[0] - 1, state[1])
        elif action_transit == "right":
            nxtState = (state[0] + 1, state[1])
        else:
            raise Exception("Wrong transition action")


        # if next state is legal
        if nxtState[0] > self.num_cols - 1 or nxtState[0] < 0 or nxtState[1] > self.num_rows - 1 or nxtState[1] < 0 or nxtState in self.list_of_blocked_states:
            self.state = state
            return state, self.giveReward(state)
        else:
            self.state = nxtState
            return nxtState, self.giveReward(nxtState)
