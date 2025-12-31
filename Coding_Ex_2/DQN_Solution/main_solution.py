import tensorflow as tf
import gym
from collections import deque
import numpy as np
import random


class GymEnvironment:
    def __init__(self, env_id, monitor_dir, max_timesteps=10000):
        self.max_timesteps = max_timesteps

        self.env = gym.make(env_id)

    def trainDQN(self, agent, no_episodes):
        self.runDQN(agent, no_episodes, training=True)

        # Automatically save weights of trained network
        agent.model.save_weights("cartpole-v0.h5", overwrite=True)


    def runDQN(self, agent, no_episodes, training=False):

        # Define array and list to store collected rewards and losses
        rew = np.zeros(no_episodes)
        losses = []
        target_counter = 0 # Initialize counter for target network synchronization

        for episode in range(no_episodes):

            # Initialize state and rewards for episode
            state = self.env.reset()[0].reshape(1, self.env.observation_space.shape[0])
            tot_rew = 0

            for t in range(self.max_timesteps):

                # Epsilon greedy action selection
                if training and np.random.rand() <= agent.epsilon:
                    action = random.randrange(agent.action_size)
                else:
                    action = np.argmax(agent.select_action(state))

                # Execute the action and observe the transition which the environment gives you, i.e., next state and reward
                next_state, reward, done, _ = self.env.step(action)[:4]
                next_state = next_state.reshape(1, self.env.observation_space.shape[0])

                if training:
                    # Record the results of the step
                    agent.record(state, action, reward, next_state, done)

                    # Update weights based on sampled transitions of experience replay
                    if len(agent.memory) >= agent.batch_size: # This updates the weights every timestep, other schedules are also possible
                        minibatch = random.sample(agent.memory, agent.batch_size)
                        states, actions, rewards, next_states, dones = agent.prepare_batch(minibatch)
                        loss = agent.update_weights(states, actions, rewards, next_states, dones)
                        losses.append(loss)

                    # Perform synchronization of target network every certain no. of steps
                    if target_counter >= agent.target_model_time and target_counter % agent.target_model_time == 0:
                        w = agent.model.get_weights()
                        agent.target_model.set_weights(w)

                tot_rew += reward # Accumulated reward for this episode
                state = next_state # Overwrite old state with new state
                target_counter += 1 # Increment target model update counter

                # End episode if terminal state was reached
                if done:
                    break

            rew[episode] = tot_rew

            # Test performance of your trained agent every 25 epsiodes
            if training:
                if episode > 0 and episode % 25 ==0:
                    rew_test = self.runDQN(agent, 25)
                    rew_test_mean = np.mean(rew_test)
                    print(f'reward after episode {episode} is {rew_test_mean}')
                    if rew_test_mean > 200: # 200 is considered to be a good reward
                        agent.model.save_weights('cartpole-v0.h5', overwrite=True)
                        break # early stopping condition if good reward is reached before all episodes are done

            print("episode: {}/{} | score: {} | e: {:.3f}".format(
                episode + 1, no_episodes, tot_rew, agent.epsilon))
        return rew # If needed, losses can also be returned


class DQN_Agent:
    def __init__(self, no_of_states, no_of_actions,load_old_model):
        self.state_size = no_of_states
        self.action_size = no_of_actions

        # Hyperparameters
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.2  # eps-greedy exploration rate
        self.batch_size = 10  # maximum size of the batches sampled from memory
        alpha = 0.001 # learning rate

        # Define your neural network and optimizers
        self.model = self.nn_model(load_old_model)
        self.target_model = self.nn_model(load_old_model)
        w = self.model.get_weights()
        self.target_model.set_weights(w)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = alpha)

        # Times at which target weights are synchronized
        self.target_model_time = 20

        # Maximal size of memory buffer
        self.memory = deque(maxlen=2000)

    # Define the neural network and load old weights if load_old_model == 1
    def nn_model(self, load_old_model): # You can also explicitely put state_size and action_size here
        # Simple fully-connected feedforward neural network with 2 hidden layers
        observation_input = tf.keras.Input(shape=(self.state_size,), dtype=tf.float32)
        dense1 = tf.keras.layers.Dense(units=64, activation=tf.tanh)(observation_input)
        dense2 = tf.keras.layers.Dense(units=64, activation=tf.tanh)(dense1)
        output = tf.keras.layers.Dense(self.action_size)(dense2)
        nn_model = tf.keras.Model(inputs=observation_input, outputs=output)

        # If you already have a set of weights from previous training, you can load them here
        if load_old_model == 1:
            nn_model.load_weights("cartpole-v0.h5")
        return nn_model

    # Return state action values, i.e., Q(s,a = 1 & a = 0)
    @tf.function # @tf.function compiles a function into a callable tensorflow graph, significantly improving speed
    def select_action(self, state):
        return self.model(state)[0]

    # Here newly observed transitions are stored in the experience replay buffer
    def record(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Prepare data from experience replay for usage in updates of weights
    def prepare_batch(self, minibatch):
        states = np.array([t[0] for t in minibatch], dtype=np.float32).reshape(self.batch_size, 4)
        actions = np.array([t[1] for t in minibatch], dtype=np.int32).reshape(self.batch_size, 1)
        indices = np.arange(self.batch_size).reshape(-1, 1)
        actions = np.hstack((indices, actions))
        rewards = np.array([t[2] for t in minibatch], dtype=np.float32).reshape(self.batch_size, 1)
        next_states = np.array([t[3] for t in minibatch], dtype=np.float32).reshape(self.batch_size, 4)
        dones = np.array([float(t[4]) for t in minibatch], dtype=np.float32).reshape(self.batch_size, 1)

        return states, actions, rewards, next_states, dones
    
    # Perform DQN update of weights w
    @tf.function
    def update_weights(self, states, actions, rewards, next_states, dones):
        # It is also possible to sample a minibatch, prepare the batch parts and perform the update in one function, but it cannot be a tf.function then and would thus be slower

        # Calculate targets, use tf.math.reduce_prod to have vectors instead of matrices
        targets = tf.math.reduce_prod(rewards, axis=1) + self.gamma * tf.math.reduce_max(self.target_model(next_states), axis=1) * tf.math.reduce_prod(1 - dones, axis=1)

        # Write down the loss function, i.e., target - predicted values
        with tf.GradientTape() as tape: # tf.GradientTape() lets tensorflow record the intented operations as part of the differentiable loss function, enabling auto-differentiation
            Q_predicted = self.model(states)
            Q_pred_action = tf.gather_nd(Q_predicted, indices=actions)
            value_loss = tf.reduce_mean((targets - Q_pred_action) ** 2)
        
        # Update weights
        grads = tape.gradient(value_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return value_loss


if __name__ == "__main__":
    environment = GymEnvironment('CartPole-v1', 'gymresults/cartpole-v1')


    # Define the number of states and actions - you will need this information for defining your neural network
    no_of_states = 4
    no_of_actions = 2

    # If you want to load weights of an already trained model, set this flag to 1
    load_old_model = 0

    # The agent is initialized
    agent = DQN_Agent(no_of_states,no_of_actions,load_old_model)

    # Train your agent
    no_episodes = 200
    environment.trainDQN(agent, no_episodes)

    # Run your agent
    no_episodes_run = 100
    rew = environment.runDQN(agent, no_episodes_run)
    print(f'final reward: {np.mean(rew)}')



    # Here you can watch a simulation on how your agent performs after being trained.
    # NOTE that this part will try to load an existing set of weights, therefore set visualize_agent to TRUE, when you
    # already saved a set of weights from a training session
    visualize_agent = True
    if visualize_agent == True:
        print(f'cartpole rendering started. You can terminate it by going to your terminal and pressing CTRL+C (Windows/Mac/Linux)')
        env = gym.make('CartPole-v1', render_mode="human")
        load_model = 1
        state_size = 4
        action_size = 2
        agent = DQN_Agent(state_size, action_size, load_model)
        for _ in range(20):
            state = env.reset()[0].reshape(1, env.observation_space.shape[0])
            done = False
            while not done:
                action = np.argmax(agent.select_action(state))
                next_state, reward, done, _ = env.step(action)[:4]
                next_state = next_state.reshape(1, env.observation_space.shape[0])
                state = next_state
        env.close()