% lecture_DRL_coding1
# Coding Exercise

**First**: To be able to work on the coding exercise you first have to set up python and an IDE (look into slides Coding1_Introduction)

**Second**: When python is running and you set up an IDE to work with, you have to install all dependencies. Therefore:
(You find more detailed instructions on installing virtual environments and dependencies in tutorials - you find the links to these tutorials in the presentation Coding1_Introduction)

Install virtualenv
```
Unix/macOS: python3 -m pip install --user virtualenv
Windows: py -m pip install --user virtualenv
```

Create a virtual environment
 ```
 Unix/macOS: python3 -m venv environment_name
 Windows: py -m venv environment_name
 ```
 activate the virtual environment
 ```
 Unix/macOS: source environment_name/bin/activate
 Windows: .\environment_name\Scripts\activate
 ```
 and install relevant dependencies that are listed in the requirements.txt file
 ```
 pip install -r requirements.txt
 ```

For this exercise we only need the libraries (dependencies):
- numpy (Reinforcement Learning - exercise)
- tensorflow (Deep Neural Network - exercise)
- pandas (Deep Neural Network - exercise)
- matplotlib (Deep Neural Network - exercise)
- sklearn (Deep Neural Network - exercise)
- and all the dependencies from these libraries

--> you do not need any more libraries to install in order to work on this exercise.



**Third**: Work on the exercise. The Coding Exercise consists out of three sub-exercises, that are independent from each other. In each sub-exercise we provide a python project that contains relevant code blocks to frame the task. You only have to implement the missing code blocks. We marked missing code blocks with TODO comments. The sub-exercises are:


## 1. Search Algorithms (Gridworld)

```
     -------------------------
2    |     |     |     |  G  |
     -------------------------
1    |     |||||||     |     |
     -------------------------
0    |  S  |     |     |     |
     -------------------------
       0     1     2     3
```


Implement a generic Graph Search algorithms. The objective in this example is to find a path from a start state S to a goal state G in a deterministic Gridworld environment (see picture above). Implement the following Graph Search algorithms:
 - Depth-First Search
 - Breadth-First Search
 - Uniform-Cost Search
 
 The structure of the project is the following
 ```
  search
  |	config.py
  |	environment.py
  |	run_search.py
  |	util.py
 ```

 *config.py*: contains all relevant hyperparameters. You can change the hyperparameter setting either directly in the code or by adding command line args when running from terminal. For example: 
 ```
 python run_search.py --search_algorithm="BFS"
 ```
 
 *environment.py*: contains the code to simulate the example environment to test your code. You do not have to change anything in this python file. Nevertheless for the overall understanding it helps to understand the functions in this file. The environment is implemented as follows:
 
 The Gridworld environment consists out of 12 states (4 columns and 3 rows), each state is defined as tuple (x,y) (see figure above). State (1,1) is a wall. The start state is (0,0) and the terminal state is (3,2). The agent can choose in every state one of the actions {up, down, left, right} if possible. In a state, an action is not available if this actions leads the agent to run against a wall. The environment is deterministic, therefore the agent performs the action as intended.
 
 *run_search.py*: contains the relevant functions that implement Depth-First Search, Breadth-First Search, and Uniform-Cost Search. So far, the functions are empty. **You have to implement the code of these functions (depth_first_search, breadth_first_search, uniform_cost_search)**. All code lines where you have to implement something are marked with a previous TODO command
 
 *util.py*: contains the code for the classes Queue, Stack, Priority Queue, ClosedList, and Graph. You will need objects of these classes to write the code in run_search.py
  


## 2. Reinforcement Learning: Q-Learning (Gridworld)

```
     -------------------------
2    |     |     |     |  G  |
     -------------------------
1    |     |||||||     |  L  |
     -------------------------
0    |  S  |     |     |     |
     -------------------------
       0     1     2     3
```
Write a generic Q-Learning algorithm. In our example the goal ist to train an agent that maximizes its reward in a non-deterministic Gridworld environment (see picture above).


The structure of the project is the following
 ```
  reinforcement_learning
  |	config.py
  |	environment.py
  |	run_Q_learning.py
  |	util.py
 ```


 *config.py*:	contains all relevant hyperparameters. You can change the hyperparameter setting either directly in the code or by adding command line args when running from terminal. For example: 
 ```
 python run_Q_learning.py --learning_rate=0.02
 ```
  For example, you can adapt the discount factor via the command line parameter decay_gamma. 
 
 *environment.py*: contains the code to simulate the environment. You do not have to change anything in this python file. Nevertheless for the overall understanding it helps to understand the functions in this file.
 
 The environment is implemented as follows:
 The Gridworld environment consists out of 12 states (4 columns and 3 rows), each state is defined as tuple (x,y) (see figure above). State (1,1) is a wall. In each epoch the agent starts from start state S (0,0). Every state has a reward of 0, whereas state (3,2) has a reward of 1 and state (3,1) has a reward of -1. The agent can choose in every state one of the actions {up, down, left, right}. If the agent runs into a wall the agent stays in the original state. The environment is non-deterministic. With a probability 0.8 the agent performs the action as intended, whereas with a probability of 0.1 (0.1) the agent moves in the right angle left (right) to the intended direction. 
 
 *run_Q_learning.py*: contains the function train() where you have to implement the training of the Q-learning agent. So far, the function train() is empty **You have to implement the code of the train() function into this file**. All code lines where you have to implement something are marked with a previous TODO command
 
 *util.py*: contains the code defining the functionalities of the agent. The Q-Values of the agent are initialized with value 0.


## 3. Neural Networks using TensorFlow 2
Write code to train a Deep Neural Network that predicts the species in the Iris data set / predicts if a client will subscribe for a bank term deposit / predicts the value of a house.

 The structure of the project is the following
 ```
  deep_neural_network
  |	config.py
  |	train_deep_neural_network.py
  |	data/
  |	------	bank_prediction.csv
  |	------	bank_test.csv
  |	------	bank_training.csv
  |	------	wine_prediction.csv
  |	------	wine_test.csv
  |	------	wine_training.csv
  |	------	iris_prediction.csv
  |	------	iris_test.csv
  |	------	iris_training.csv
 ```
 
  *config.py*:	contains all relevant hyperparameters. You can change the hyperparameter setting either directly in the code or by adding command line args when running from terminal. For example:
  ``` 
  python train_deep_neural_network.py --data="iris"
  ```
  
  *train_deep_neural_network.py*: contains the relevant part to implement the Deep Neural Network. **You have to implement the code into this file**. All code lines where you have to implement something are marked with a previous TODO command. For a better overview we have summarized the most important implementing steps in this content list (You find the relevant sections in the code train_deep_neural_network.py by searching for the respective #)
  1. Define the deep neural network models (#defineModel)
     1. as subclass of Model (#defineModel_model)
     2. with using sequential API (#defineModel_sequential)
     3. with using functional API (#defineModel_functional)
  2. Read in the data sets (#readInData)
  Option1 – args.implementation=‘detail’
  3. Define metrics to track the learning progress (#defineMetrics)
  4. Define the loss function (#defineLoss)
  5. Define gradient (#defineGradient) 
  6. Define the optimization algorithm (#defineOptimization)
  7. Train the model (#trainModel)
  8. Evaluate the model (#evaluateModel)
  9. Use the model to make predictions (#predictTarget)
  Option2 – args.implementation=‘no_detail’
  10. Define metrics to track the learning progress (#defineMetrics2)
  11. Compile the model (#compileModel2)
  12. Fit the model (#fitModel2)
  13. Evaluate model (#evaluateModel2)

  *data/*:
  1. Bank: predict if a client will subscribe for a bank term deposit or not | Classification | Output: {yes, no} | Features: Numerical (one-hot encoded)
  2. Wine: predict the quality of wine | Regression | Output: continuous | Features: Numerical
  3. Iris: predict the species of a flower | Classification | Output: {Iris Setosa, Iris Versicolour, Iris Virginica} | Features: Numerical





If you do not want to follow the predefined functions or the code structure of the projects, feel free to write your own code / projects. There exist several different solutions to implement search algorithms / deep neural networks / Q-learning.
The predefined functions and the TODO comments are only meant to help you with the tasks. 
