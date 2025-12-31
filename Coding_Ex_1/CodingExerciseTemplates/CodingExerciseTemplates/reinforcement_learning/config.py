import argparse

parser = argparse.ArgumentParser(description="read input parameters")

parser.add_argument('--number_epochs', type=int, help="number of learning epochs - one epoche is one round till agent reaches terminal state", default=3000)

parser.add_argument('--learning_rate', type=float, help="learning rate - alpha - for the Q update", default=0.2)

parser.add_argument('--exploration_rate', type=float, help="exploration rate --> gives probability to choose random action", default=0.2)

parser.add_argument('--decay_gamma', type=float, help="decay factor - gamma - to decay the Q(s',a) value in temporal difference learning", default=0.9)