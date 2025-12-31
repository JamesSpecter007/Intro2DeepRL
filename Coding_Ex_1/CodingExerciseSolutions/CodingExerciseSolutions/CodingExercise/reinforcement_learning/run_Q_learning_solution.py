import config
import environment
import util

# https://towardsdatascience.com/implement-grid-world-with-q-learning-51151747b455

# implementation of the reinforcement learning algorithm
#implementHere
def train(agent, env, epochs=10):
    # initialize the q-values of the agent
    agent.init_q_values(env)
    env.counter_epoch = 0
    while env.counter_epoch < epochs:
        print("-------------------------------------- We start in round : {} ---------------------------------".format(env.counter_epoch))
        # TODO set the env.state to the start state
        env.state = env.start_state
        # TODO set state to the env.state
        state = env.state
        while not env.is_terminal_state():
            # TODO the agent chooses the action for the state, and saves it in variable "action"
            action = agent.chooseAction(state=state) # e-greedy
            print("current position {} and agent chooses action {}".format(state, action))
            # TODO forward the action to the environemnt and save output in state_new and reward
            state_new, reward = env.transition(action=action)
            # update Q
            # TODO set the value for the new Q-value for Q(state, action)
            reward = agent.Q_values[state][action] \
                     + agent.lr * (reward + agent.decay_gamma * agent.Q_values[state_new][agent.get_greedy_action(state_new)] -
                                    agent.Q_values[state][action])
            agent.Q_values[state][action] = round(reward, 3)
            print("nxt state", state_new)
            print("---------------------")
            # TODO set state to state_new
            state = state_new
        # TODO update the epoch
        env.counter_epoch += 1
    agent.extract_policy(env=env)


if __name__ == "__main__":
    args = config.parser.parse_args()

    # create the new environment
    env = environment.Environment(args)
    # create the new agent
    agent = util.Agent(env=env, args=args)
    print("initial Q-values ... \n")
    print(agent.Q_values)

    # run reinforcement learning algorithm
    train(agent=agent, env=env, epochs=args.number_epochs)
    print("latest Q-values ... \n")
    print(agent.Q_values)
    