import argparse

parser = argparse.ArgumentParser(description="read input parameters")

parser.add_argument('--data', type=str, help="data set", choices=["iris", "bank", "wine"], default="iris")

parser.add_argument('--implementation', type=str, help="implementation style", choices=["detail", "no_detail"], default="detail")

parser.add_argument('--optimizer', type=str, help="optimizer to use for training", choices=["Adam", "SGD", "RMSprop"], default="SGD")

parser.add_argument('--loss', type=str, help="optimizer to use for training", choices=["CategoricalCrossEntropy", "MeanSquaredErrors"], default="CategoricalCrossEntropy")

parser.add_argument('--num_epochs', type=int, help="number of learning epochs", default=70)

parser.add_argument('--batch_size', type=int, help="number of instances per batch", default=32)

parser.add_argument('--network_type', type=str, help="define the API to model the network",
                    choices=["sequential", "functional", "model"], default="functional")
