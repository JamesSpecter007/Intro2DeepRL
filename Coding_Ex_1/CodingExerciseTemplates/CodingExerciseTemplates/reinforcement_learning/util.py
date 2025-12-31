import numpy as np

class Agent:
    """
    class defining the agent
    """
    def __init__(self, env, args):
        self.actions = ["up", "down", "left", "right"]
        self.lr = args.learning_rate
        self.exp_rate = args.exploration_rate
        self.decay_gamma = args.decay_gamma
        self.init_q_values(env=env)
        np.random.seed(0)

    def init_q_values(self, env):
        """
        sets all Q-Values to 0
        :param env: environment the agent is acting in to get the dimensions for the Q-Values
        :return: void
        """
        # initial Q values
        self.Q_values = {}
        for row in range(env.num_rows):
            for col in range(env.num_cols):
                self.Q_values[(col, row)] = {}
                for a in self.actions:
                    self.Q_values[(col, row)][a] = 0  # Q value is a dict of dict

    def get_greedy_action(self, state):
        """
        returns the action with the highest Q-Value
        :param state: get the action for this state
        :return: action with highest Q-Value, when being in this state
        """
        mx_nxt_reward = 0
        action = ""
        # greedy action
        for a in self.actions:
            nxt_reward = self.Q_values[state][a]
            if nxt_reward >= mx_nxt_reward:
                action = a
                mx_nxt_reward = nxt_reward
        return action

    def chooseAction(self, state):
        """
        regulates exploration and exploitation using command args exp_rate
        :param state: actual state of the agent
        :return: action that the agent takes
        """

        # choose action with most expected value
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            action = self.get_greedy_action(state=state)
        return action

    def extract_policy(self, env):
        """
        extracts the policy for every state in the environment and the optimal path to maximize the reward
        when starting in the start state
        :param env: environment the agent is acting in
        :return: void - as the function prints out the best action for every state in the environment and the
        best solution path
        """
        print("****** optimal policy *******")
        # print policy
        for row in range(env.num_rows):
            for col in range(env.num_cols):
                state = (col, row)
                if (state not in env.win_state) and (state not in env.lose_state) and (state not in env.list_of_blocked_states):
                    action = self.get_greedy_action(state)
                    print("State: {} --> action: {}".format(state, action))
        print("****** optimal path *******")
        state = env.start_state
        counter = 0
        print("The best path is:")
        while not env.is_terminal_state(state):
            counter = counter + 1
            if counter > 100:
                print("This path seems to be infinite")
                break
            action = self.get_greedy_action(state=state)
            print("State: {} --> action: {}".format(state, action))
            state_new, _ = env.transition(state=state, action=action)
            state = state_new


