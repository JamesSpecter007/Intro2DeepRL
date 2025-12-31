import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import config


# https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
# https://github.com/lucone83/deep-learning-with-python/blob/master/notebooks/chapter_03/03%20-%20Logistic%20regression.ipynb
# https://www.tensorflow.org/guide/effective_tf2

'''
CONTENT LIST
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
'''



# define the Neural Network for the wine data set
#defineModel_model
# TODO define a model using the subclass Model (https://www.tensorflow.org/api_docs/python/tf/keras/Model)
class WineModel(tf.keras.Model):
    #  Define a model with the following layers:
    #  1. Dense layer (input_dimension -> 64, relu activation)
    #  1. Dense layer (64 -> 64, relu activation)
    #  1. Dense layer (64 -> 1)
    def __init__(self):
        super().__init__()
        self.dense1 =
        self.dense2 =
        self.out =

    def call(self, inputs, training=False):

        return output

#defineModel_model
# TODO define a model using the subclass Model (https://www.tensorflow.org/api_docs/python/tf/keras/Model)
class BankingModel(tf.keras.Model):
    # Define a model with the following layers:
    #  1. Dense layer (input_dimension -> 100, sigmoid activation)
    #  1. Dense layer (100 -> 100, sigmoid activation)
    #  1. Dense layer (100 -> 10, sigmoid activation)
    #  1. Dense layer (64 -> 2)
    def __init__(self):
        super().__init__()
        self.dense1 =
        self.dense2 =
        self.dense3 =
        self.out =

    def call(self, inputs, training=False):

        return output

#defineModel_model
# TODO define a model using the subclass Model (https://www.tensorflow.org/api_docs/python/tf/keras/Model)
class IrisModel(tf.keras.Model):
    # Define a model with the following layers:
    #  1. Dense layer (input_dimension -> 10, relu activation)
    #  1. Dense layer (10 -> 10, relu activation)
    #  1. Dense layer (10 -> 3)
    def __init__(self):
        super().__init__()
        self.dense1 =
        self.dense2 =
        self.out =

    def call(self, inputs, training=False):

        return output



#defineModel
def get_model_wine(input_dimension):
    #  Define a model with the following layers:
    #  1. Dense layer (input_dimension -> 64, relu activation)
    #  1. Dense layer (64 -> 64, relu activation)
    #  1. Dense layer (64 -> 1)

    #defineModel_sequential
    if args.network_type == "sequential":
        # TODO define a Sequential model (https://www.tensorflow.org/api_docs/python/tf/keras/Sequential)
        model =

        return model

    # the advantage of a functional API is that you can handle non-linear topology, shared layers,
    # and even multiple inputs or outputs
    #defineModel_functional
    if args.network_type == "functional":
        # TODO define a Functional model (https://www.tensorflow.org/guide/keras/functional)
        inputs =
        dense1 =
        dense2 =
        outputs =

        model =
        return model

    if args.network_type == "model":
        model = WineModel()
        model.build((args.batch_size, input_dimension))
        return model



# define the Neural Network for the banking data set
def get_model_banking(input_dimension):
    # Define a model with the following layers:
    #  1. Dense layer (input_dimension -> 100, sigmoid activation)
    #  1. Dense layer (100 -> 100, sigmoid activation)
    #  1. Dense layer (100 -> 10, sigmoid activation)
    #  1. Dense layer (64 -> 2)

    #defineModel_sequential
    if args.network_type == "sequential":
        # TODO define a Sequential model (https://www.tensorflow.org/api_docs/python/tf/keras/Sequential)
        model =
        return model

    # the advantage of a functional API is that you can handle non-linear topology, shared layers,
    # and even multiple inputs or outputs
    #defineModel_functional
    if args.network_type == "functional":
        # TODO define a Functional model (https://www.tensorflow.org/guide/keras/functional)
        inputs =
        dense1 = 
        dense2 = 
        dense3 =
        outputs =

        model =
        return model

    # here we define a model as a subclass of Model
    if args.network_type == "model":
        model = BankingModel()
        model.build((args.batch_size, input_dimension))
        return model




# define the Neural Network for the iris data set
def get_model_iris(input_dimension):
    # Define a model with the following layers:
    #  1. Dense layer (input_dimension -> 10, relu activation)
    #  1. Dense layer (10 -> 10, relu activation)
    #  1. Dense layer (10 -> 3)

    #defineModel_sequential
    if args.network_type == "sequential":
        # TODO define a Sequential model (https://www.tensorflow.org/api_docs/python/tf/keras/Sequential)
        model =

        return model

    # the advantage of a functional API is that you can handle non-linear topology, shared layers,
    # and even multiple inputs or outputs
    #defineModel_functional
    if args.network_type == "functional":
        # TODO define a Functional model (https://www.tensorflow.org/guide/keras/functional)
        inputs =
        dense1 =
        dense2 =
        outputs =

        model =
        return model

    # here we define a model as a subclass of Model
    if args.network_type == "model":
        model = IrisModel()
        model.build((args.batch_size, input_dimension))
        return model


if __name__ == '__main__':
    args = config.parser.parse_args()

    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    # read in the selected data
    if args.data == "iris":
        path_training = "data/iris_training.csv"
        path_testing = "data/iris_test.csv"
        path_prediction = "data/iris_prediction.csv"
        class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
        if args.loss != "CategoricalCrossEntropy":
            raise Exception("We recommend to use *args.loss=CategoricalCrossEntropy* as we want to apply logistic "
                            "regression on the data set. You can change this in the config.py file or as terminal args "
                            "python train_deep_neural_network.py --loss='CategoricalCrossEntropy'")
    elif args.data == "bank":
        path_training = "data/bank_training.csv"
        path_testing = "data/bank_test.csv"
        path_prediction = "data/bank_prediction.csv"
        class_names = ['yes', 'no']
        if args.loss != "CategoricalCrossEntropy":
            raise Exception("We recommend to use *args.loss=CategoricalCrossEntropy* as we want to apply logistic "
                            "regression on the data set on the data set. You can change this in the config.py file or as terminal args "
                            "python train_deep_neural_network.py --loss='CategoricalCrossEntropy'")
    elif args.data == "wine":
        path_training = "data/wine_training.csv"
        path_testing = "data/wine_test.csv"
        path_prediction = "data/wine_prediction.csv"
        class_names = ['continuous target variable']
        if args.loss != "MeanSquaredErrors":
            raise Exception("We recommend to use *args.loss=MeanSquaredErrors* as we want to apply regression on the "
                            "data set on the data set. You can change this in the config.py file or as terminal args "
                            "python train_deep_neural_network.py --loss='MeanSquaredErrors'")
    else:
        raise Exception("This data set does not exist")

    # read in training data as table (pandas)
    training_pandas = pd.read_csv(path_training, delimiter=",")
    # print the first columns of the data set
    print(training_pandas.head())
    # get the column names
    column_names = list(training_pandas.columns)
    # get the feature names
    feature_names = column_names[:-1]
    # get the label names
    label_name = column_names[-1]

    #readInData
    # TODO read in data set stored in path_training as tf.data.Dataset (https://www.tensorflow.org/api_docs/python/tf/data/experimental/make_csv_dataset)
    #  and pass args.batch_size, column_names, label_name and num_epochs=1 to the method
    train_dataset =

    # TODO read in data set stored in path_testing as tf.data.Dataset (https://www.tensorflow.org/api_docs/python/tf/data/experimental/make_csv_dataset)
    #  and pass args.batch_size, column_names, label_name and num_epochs=1 and shuffle=False to the method
    test_dataset =

    # TODO read in the data set stored in path_prediction as numpy array
    predict_dataset =

    # choose the correct model according to the data set
    if args.data == "iris":
        model = get_model_iris(input_dimension=len(column_names) - 1)
    elif args.data == "bank":
        model = get_model_banking(input_dimension=len(column_names) - 1)
    elif args.data == "wine":
        model = get_model_wine(input_dimension=len(column_names) - 1)
    else:
        raise Exception("no model defined")

    model.summary()

    # Pack the features into a single array
    def pack_features_vector(features, labels):
        features = tf.stack(list(features.values()), axis=1)
        return features, labels

    train_dataset = train_dataset.map(pack_features_vector)
    test_dataset = test_dataset.map(pack_features_vector)

    # Here we implement a detailed learning implementation. Most of the code that we implement here is called
    # automatically by using *args.implementation="no_detail* which uses the function
    # model.fit according to https://keras.io/guides/customizing_what_happens_in_fit/
    if args.implementation == "detail":

        # select the correct metrics according to the objective and loss function
        #defineMetrics
        if args.loss == "CategoricalCrossEntropy":
            # TODO define a SparseCategoricalCrossentropy loss with from_logits=True
            #  (https://www.tensorflow.org/api_docs/python/tf/keras/losses) and keep it in variable loss_object
            loss_object =
            # TODO define test_accuracy with the Accuracy metric and epoch_accuracy with the SparseCategoricalAccuracy
            #  (https://www.tensorflow.org/api_docs/python/tf/keras/metrics)
            test_accuracy =
            epoch_accuracy =
            # TODO define epoch_loss_avg with the Mean metric
            #  (https://www.tensorflow.org/api_docs/python/tf/keras/metrics)
            epoch_loss_avg =
        elif args.loss == "MeanSquaredErrors":
            # TODO define a MeanSquaredError loss
            #  (https://www.tensorflow.org/api_docs/python/tf/keras/losses) and keep it in variable loss_object
            loss_object =
            # TODO define test_accuracy with the Accuracy metric and epoch_accuracy with the MeanAbsoluteError
            #  (https://www.tensorflow.org/api_docs/python/tf/keras/metrics)
            test_accuracy =
            epoch_accuracy =
            # TODO define epoch_loss_avg with the Mean metric
            #  (https://www.tensorflow.org/api_docs/python/tf/keras/metrics)
            epoch_loss_avg =
        else:
            raise Exception("The selected loss is not implemented so far")

        #defineLoss
        def loss(model, x, y, training):
            """
            calculates the loss value of this training iteration
            :param model: model to train
            :param x: input features
            :param y: correct output features
            :param training: is only needed, if there exist layers in the model that behave differently in training and evaluation
            :return: loss object (https://www.tensorflow.org/api_docs/python/tf/keras/losses)
            """
            # TODO get the prediction y_ of input x using the model
            y_ =

            return loss_object(y_true=y, y_pred=y_)

        #defineGradient
        def grad(model, inputs, targets):
            """
            calculates the gradient of this training iteration
            :param model: model to train
            :param inputs: input to the model
            :param targets: target output of the model
            :return: loss value and gradient (https://www.tensorflow.org/api_docs/python/tf/GradientTape)
            """
            with tf.GradientTape() as tape:
                # TODO get the loss_value, using the implemented loss function
                loss_value =
            return loss_value,

        # define the different optimizers (https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)
        # according to the problem
        #defineOptimization
        if args.optimizer == "SGD":
            # TODO define SGD optimizer
            optimizer =
        elif args.optimizer == "Adam":
            # TODO define Adam optimizer
            optimizer =
        elif args.optimizer == "RMSprop":
            # TODO define RMSprop optimizer
            optimizer =
        else:
            raise Exception("The selected optimizer is not implemented")

        # storage to keep results for plotting later
        train_loss_results = []
        train_accuracy_results = []

        # run over every training epoche
        #trainModel
        for epoch in range(args.num_epochs):

            # run over every batch in the training data set
            for x, y in train_dataset:
                # TODO get the loss_value and gradients of the actual model. The loss_value and the gradients
                #  depend on the model itself, input x, and target y
                loss_value, grads =

                # TODO optimize the model by applying the actual gradients on the training variables of the model


                # TODO update the epoch_loss_avg with the current batch loss_value
                epoch_loss_avg.

                # TODO update the epoch_accuracy with the specified metric for the difference between the predicted
                #  label and the actual label
                epoch_accuracy.

            # TODO after every learning epoch add the epoch_loss_avg results to the train_loss_results and add the
            #  epoch_accuracy results to the train_accuracy_results for plotting later
            train_loss_results.
            train_accuracy_results.

            # print out actual state of the learning process
            if epoch % 10 == 0:
                if args.loss == "CategoricalCrossEntropy":
                    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                                epoch_loss_avg.result(),
                                                                                epoch_accuracy.result()))
                elif args.loss == "MeanSquaredErrors":
                    print("Epoch {:03d}: Loss: {:.3f}, MSE: {:.3}".format(epoch,
                                                                                epoch_loss_avg.result(),
                                                                                epoch_accuracy.result()))
                else:
                    raise Exception("The chosen loss is not implemented so far")

        # print evolution of learning metrics during training
        fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
        fig.suptitle('Training Metrics')

        axes[0].set_ylabel("Loss", fontsize=14)
        axes[0].plot(train_loss_results)

        if args.loss == "CategoricalCrossEntropy":
            axes[1].set_ylabel("Accuracy", fontsize=14)
        elif args.loss == "MeanSquaredErrors":
            axes[1].set_ylabel("MSE", fontsize=14)
        else:
            raise Exception("Wrong loss defined")
        axes[1].set_xlabel("Epoch", fontsize=14)
        axes[1].plot(train_accuracy_results)
        plt.show()

        # Get the accuracy of model for testing data set
        #evaluateModel
        if args.loss == "CategoricalCrossEntropy":
            for (x, y) in test_dataset:
                # TODO get the logits output of the model with input x
                logits =
                # TODO get the class with the maximum logits value
                prediction =
                # TODO calculate the accuracy between the prediction and target y using test_accuracy

                print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
        elif args.loss == "MeanSquaredErrors":
            # TODO get the regression output of the model with input x
            prediction =
            # TODO calculate the accuracy between the prediction and target y using test_accuracy

            print("Test set MSE: {:.3}".format(test_accuracy.result()))


        # TODO predict the output of the Neural Network for the instances of the prediction data set
        predictions =

        # plot the predictions of the instances from the prediction data set
        #predictTarget
        for i, pred in enumerate(predictions):
            if args.loss == "CategoricalCrossEntropy":
                # TODO use tf.argmax to extract the class with the highest prediction and save it in class_idx
                class_idx =
                # TODO calculate the softmax output p for the highest prediction using tf.nn.softmax
                p =
                name = class_names[class_idx]
                print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100 * p))
            elif args.loss == "MeanSquaredErrors":
                print("Example {} prediction: {}".format(i, pred))

    else:
        # implementation using model.fit()

        if args.optimizer == "Adam":
            optimizer_object = "adam"
        elif args.optimizer == "SGD":
            optimizer_object = "sgd"
        elif args.optimizer == "RMSprop":
            optimizer_object = "rmsprop"
        else:
            raise Exception("Wrong optimizer declared")

        #defineMetrics2
        if args.loss == "CategoricalCrossEntropy":
            # TODO define a SparseCategoricalCrossentropy loss with from_logits=True
            #  (https://www.tensorflow.org/api_docs/python/tf/keras/losses) and keep it in variable loss_object
            loss_object =
            metrics_object = ['accuracy']
        elif args.loss == "MeanSquaredErrors":
            # TODO define a MeanSquaredError loss with from_logits=True
            #  (https://www.tensorflow.org/api_docs/python/tf/keras/losses) and keep it in variable loss_object
            loss_object =
            metrics_object = ['mae']
        else:
            raise Exception("The selected loss is not implemented so far")

        #compileModel2
        # TODO compile the model with using the optimizer_object, loss_object and the metrics_object
        model.

        #fitModel2
        # TODO fit the model with the train_dataset and epochs=args.num_epochs
        model.

        #evaluateModel2
        # TODO evaluate the trained model with the test_dataset
        loss, acc =

        if args.loss == "CategoricalCrossEntropy":
            print("Loss {}, Accuracy {}".format(loss, acc))
            predictions = model(predict_dataset)
            print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
            print("    Labels: {}".format(class_names))
        elif args.loss == "MeanSquaredErrors":
            print("Loss {}, MAE {}".format(loss, acc))
            predictions = model(predict_dataset)
            print("Prediction: {}".format(predictions))
