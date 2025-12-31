import pandas as pd
import sklearn.model_selection
from sklearn.model_selection import train_test_split

def preprocess_bank_data():
    # from https://archive.ics.uci.edu/ml/datasets/bank+marketing
    bank_data_set = pd.read_csv("data_unprocessed/bank-additional-full.csv", delimiter=";")

    # drop NaN rows
    bank_data_set = bank_data_set.dropna()

    # get information about the data set
    bank_data_set.head()
    print("Length of data set: {}".format(len(bank_data_set)))

    # apply one hot encoding for the following columns
    # job
    job_one_hot = pd.get_dummies(bank_data_set.job)
    # merital
    marital_one_hot = pd.get_dummies(bank_data_set.marital)
    # education
    education_one_hot = pd.get_dummies(bank_data_set.education)
    # default
    default_one_hot = pd.get_dummies(bank_data_set.default)
    # housing
    housing_one_hot = pd.get_dummies(bank_data_set.housing)
    # loan
    loan_one_hot = pd.get_dummies(bank_data_set.loan)
    # contact
    contact_one_hot = pd.get_dummies(bank_data_set.contact)
    # month
    month_one_hot = pd.get_dummies(bank_data_set.month)
    # day_of_week
    day_of_week_one_hot = pd.get_dummies(bank_data_set.day_of_week)
    # poutcome
    poutcome_one_hot = pd.get_dummies(bank_data_set.poutcome)

    bank_data_set = bank_data_set.drop(columns=["job", "marital", "education", "default", "housing", "loan", "contact",
                                                "month", "day_of_week", "poutcome"])

    bank_data_set["y"] = bank_data_set.y.map(dict(yes=1, no=0))

    bank_data_set = pd.concat(
        [bank_data_set, job_one_hot, marital_one_hot, education_one_hot, default_one_hot, housing_one_hot,
         loan_one_hot, contact_one_hot, month_one_hot, day_of_week_one_hot, poutcome_one_hot], axis=1)

    bank_data_set["target"] = bank_data_set["y"]
    bank_data_set.drop(columns=["y"])

    bank_data_set = bank_data_set.astype('float64')
    bank_data_set = bank_data_set.sample(n = 4000)

    training_data, testing_data = train_test_split(bank_data_set, test_size=0.2)
    _, prediction_data = train_test_split(testing_data, test_size=0.2)
    training_data.to_csv("./data/bank_training.csv", index=False)
    testing_data.to_csv("./data/bank_test.csv", index=False)
    prediction_data = prediction_data.drop(columns=["target"])
    prediction_data.to_csv("./data/bank_prediction.csv", index=False)

def preprocess_iris_data():
    #train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
    #train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
    #                                           origin=train_dataset_url)
    #print("Local copy of the dataset file: {}".format(train_dataset_fp))

    iris_training = pd.read_csv("data_unprocessed/iris_training.csv", delimiter=",")
    iris_training.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    iris_training.to_csv("./data/iris_training.csv", index=False)
    iris_test = pd.read_csv("data_unprocessed/iris_test.csv", delimiter=",")
    iris_test.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    iris_test.to_csv("./data/iris_test.csv", index=False)
    _, iris_prediction = train_test_split(iris_test, test_size=0.4)
    iris_prediction.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    iris_prediction = iris_prediction.drop(columns=["species"])
    iris_prediction.to_csv("./data/iris_prediction.csv", index=False)

def preprocess_wine():
    # from https://archive.ics.uci.edu/ml/datasets/wine+quality
    wine_data_set = pd.read_csv("data_unprocessed/winequality-red.csv", delimiter=";")

    # drop NaN rows
    wine_data_set = wine_data_set.dropna()

    training_data, testing_data = train_test_split(wine_data_set, test_size=0.1)
    training_target = training_data["quality"]
    testing_target = testing_data["quality"]
    training_target.columns = ["target"]
    testing_target.columns = ["target"]
    training_data = training_data.drop(columns=["quality"])
    testing_data = testing_data.drop(columns=["quality"])

    mean = training_data.mean(axis=0)
    training_data -= mean
    std = training_data.std(axis=0)
    training_data /= std

    testing_data -= mean
    testing_data /= std

    training_data = pd.DataFrame(training_data)
    training_target = pd.DataFrame(training_target)
    testing_data = pd.DataFrame(testing_data)
    testing_target = pd.DataFrame(testing_target)
    _, prediction_dataset = sklearn.model_selection.train_test_split(testing_data, test_size=0.4)
    training_dataset = pd.concat([training_data, training_target], axis=1)
    testing_dataset = pd.concat([testing_data, testing_target], axis=1)
    training_dataset.to_csv("./data/wine_training.csv", index=False)
    testing_dataset.to_csv("./data/wine_test.csv", index=False)
    prediction_dataset.to_csv("./data/wine_prediction.csv", index=False)



if __name__ == '__main__':
    #preprocess_bank_data()
    #preprocess_iris_data()
    preprocess_wine()






