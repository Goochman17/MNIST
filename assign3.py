# Keegan Gauthier
# 201703003
# x2017tpd@stfx.ca
from pathlib import Path
from typing import Union

# import isort
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import ravel
from numpy.core.fromnumeric import transpose
from pandas import DataFrame
from scipy.io.matlab.mio import loadmat
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_validate  # noqa
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPool2D,
    BatchNormalization,
    ReLU,
)  # noqa
from keras.models import Input, Sequential
from keras.callbacks import LearningRateScheduler
from tensorflow.python.util.nest import flatten

NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)
OUT_CHANNELS = 5
BATCH_SIZE = 8
EPOCHS = 15


# this function will be called later to load in my data set for questions
#  2 and 3
def load() -> Union[pd.DataFrame, np.ndarray]:
    # store the path of the file we're loading in
    ROOT = Path(__file__).resolve().parent
    # now add the file name to the end of that path
    DATA_FILE = ROOT / "data/features_30_sec.csv"
    # read the data and store it as a dataframe
    data = pd.read_csv(DATA_FILE)
    # return the dataframe so we can manipulate it later
    return data


def save_mnist_kfold(kfold_scores: pd.DataFrame) -> None:
    COLS = sorted(["cnn", "knn1", "knn5", "knn10"])
    df = kfold_scores
    if not isinstance(df, DataFrame):
        raise ValueError(
            "Argument `kfold_scores` to `save` must be a pandas DataFrame."
        )  # noqa
    if kfold_scores.shape != (1, 4):
        raise ValueError("DataFrame must have 1 row and 4 columns.")
    if not np.alltrue(sorted(df.columns) == COLS):
        raise ValueError("Columns are incorrectly named.")
    if not df.index.values[0] == "err":
        raise ValueError(
            "Row has bad index name. Use `kfold_scores.index = ['err']` to fix."
        )  # noqa

    if df.loc["err", ["knn1", "knn5", "knn10"]].min() > 0.06:
        raise ValueError(
            "One of your KNN error rates is likely too high. There is likely an error in your code."
        )  # noqa

    outfile = Path(__file__).resolve().parent / "kfold_mnist.json"
    df.to_json(outfile)
    print(f"K-Fold error rates for MNIST data successfully saved to {outfile}")


def save_data_kfold(kfold_scores: pd.DataFrame) -> None:
    KNN_COLS = sorted(["knn1", "knn5", "knn10"])
    df = kfold_scores
    for knn_col in KNN_COLS:
        if knn_col not in df.columns:
            raise ValueError(
                "Your DataFrame is missing a KNN error rate or is misnamed."
            )  # noqa
    if not isinstance(df, DataFrame):
        raise ValueError(
            "Argument `kfold_scores` to `save` must be a pandas DataFrame."
        )  # noqa
    if not df.index.values[0] == "err":
        raise ValueError(
            "Row has bad index name. Use `kfold_score.index = ['err']` to fix."
        )  # noqa

    outfile = Path(__file__).resolve().parent / "kfold_data.json"
    df.to_json(outfile)
    print(
        f"K-Fold error rates for individual dataset successfully saved to {outfile}"
    )  # noqa


# this function loads in the number recognition data file
# and uses the classification models for this assignment
# to make predictions
def question1():
    # load in the number recognition file
    data = loadmat(
        "/Users/k/Desktop/4th_Year/First_Semester/CSCI 444 - Machine Learning/Asn3/NumberRecognitionBiggest.mat"
    )  # noqa
    # store all the training data
    cnn_x_train = data["X_train"]
    cnn_y_train = data["y_train"]
    # do some data preprocessing for our models
    cnn_y_train = transpose(cnn_y_train)
    cnn_x_train = cnn_x_train.astype("float32") / 255
    cnn_x_train = np.expand_dims(cnn_x_train, -1)
    cnn_y_train = tf.keras.utils.to_categorical(cnn_y_train, NUM_CLASSES)

    kf = KFold(5, shuffle=True, random_state=1)
    cvscores = []
    fold_num = 1
    for train_index, test_index in kf.split(cnn_x_train, cnn_y_train):
        print("KFold number: " + str(fold_num) + "/5")
        model = None
        model = Sequential(
            [
                # The Input layer lets Keras figure out all the shapes and
                # sizes you need. It isn't
                # always necessary, but is almost always a
                # good idea for basic models.
                Input(shape=INPUT_SHAPE),
                # Note we don't have to calculate the padding in Keras,
                # due to precompilation.
                Conv2D(
                    OUT_CHANNELS, kernel_size=3, padding="same", use_bias=True
                ),  # noqa
                ReLU(),  # Without a non-linear activation, a Conv2D layer
                # is just linear
                Flatten(),  # see notes above
                Dense(NUM_CLASSES, activation="softmax"),
            ]
        )
        model.compile(
            loss="mean_squared_error",
            optimizer=tf.keras.optimizers.SGD(momentum=0.9, learning_rate=0.001),
            metrics=["accuracy"],
        )  # noqa
        model.fit(
            cnn_x_train[train_index], cnn_y_train[train_index], epochs=EPOCHS, verbose=1
        )  # noqa
        scores = model.evaluate(
            cnn_x_train[test_index], cnn_y_train[test_index], verbose=0
        )  # noqa
        cvscores.append(scores[1])
        fold_num += 1

    # store the training data into a variable X_train
    X_train = data["X_train"]
    # store the labels of the data into a variable called y
    y_train = data["y_train"]
    # store the samples and features into two seperate variable for
    # data preprocessing
    n_samples = X_train.shape[0]
    n_features = np.prod(X_train.shape[1:])
    # reshape the data to features by samples
    X_train = X_train.reshape([n_samples, n_features])
    # transpose the data and its labels so the shape is samples
    # by features
    y_train = transpose(y_train)
    y_train = ravel(y_train)
    knn_1 = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    knn_5 = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn_10 = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
    # initiate a dictionary to hold onto cross validation values
    # for each classifier
    Results = {}
    # now calculate and append the cross validation values for 5 kfolds
    # for each classifier
    Results.update({"cnn": 1 - np.mean(cvscores)})  # noqa
    Results.update(
        {
            "knn1": 1
            - np.mean(cross_validate(knn_1, X_train, y_train, cv=kf)["test_score"])
        }
    )  # noqa
    Results.update(
        {
            "knn5": 1
            - np.mean(cross_validate(knn_5, X_train, y_train, cv=kf)["test_score"])
        }
    )  # noqa
    Results.update(
        {
            "knn10": 1
            - np.mean(cross_validate(knn_10, X_train, y_train, cv=kf)["test_score"])
        }
    )  # noqa
    # turn our dictionary of results into a dataframe of 6
    # columns and one row
    Results = pd.DataFrame(Results, index=["err"], columns=Results.keys())
    # return the dataframe of results
    return Results


# this helper function will be used in question 2
# to sort the dataframe
def sort_aucs(dataframe: pd.DataFrame) -> None:
    return dataframe.sort_values("AUC", ascending=False)


# this question will compute the auc values
# for my dataset on wine quality
def question2():
    # load in the dataset using the function we were given
    df = load()
    # store the labels of the dataset into another array
    y = df["target"].to_numpy()
    # create a list of row names for our dataframe
    FEAT_NAMES = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "quality",
    ]  # noqa
    # create a list of column names for out dataframe
    COLS = ["Feature", "AUC"]
    # initiate a dataframe for our auc values of each feature
    aucs = pd.DataFrame(
        columns=COLS, data=np.zeros([len(FEAT_NAMES), len(COLS)])
    )  # noqa
    # now loop through the feature values
    for i in range(len(FEAT_NAMES)):
        # calculate the auc for that feature
        auc = roc_auc_score(y, df[FEAT_NAMES[i]])
        # save the auc score for that feature in the dataframe
        aucs.iloc[i] = (FEAT_NAMES[i], auc)
    # use our helper function to sort the dataframe
    aucs_sorted = sort_aucs(aucs)
    # now save the dataframe to a json file
    aucs_sorted.to_json(Path(__file__).resolve().parent / "aucs.json")
    # create a new column in the dataframe to show each
    # features' distance from 0.5 to show how intersting it is
    aucs_sorted["Distances from 0.5"] = abs(aucs_sorted["AUC"] - 0.5)
    # just store the top ten values into a new variable
    # so we can return them later
    top_ten = aucs_sorted.iloc[0:10]
    # this next line is commented out so i dont override my answers
    # in the responses.md file everytime i run my code
    # but this line just creates a table in markdown of my
    # feature auc values
    # top_ten.round(3).to_markdown('/Users/k/Desktop/4th Year/First Semester/CSCI 444 - Machine Learning/Asn3/responses.md')  # noqa
    # now return the dataframe of top ten auc values
    return top_ten


# helper function with question 3
def optimize_ANN():
    # use the function given to load my data in
    df = load()
    # create an array of labels from the data
    y = df["target"].to_numpy()
    # delete the label column from the dataset so
    # our classifiers dont try and make predictions off of it
    del df["target"]
    # initate all of our classifcation models
    # initiate our kfolds on our data
    kf = KFold(n_splits=5)
    ann = MLPClassifier(max_iter=1000)
    # create a dictionary of our comparable parameters
    parameter_space = {
        "hidden_layer_sizes": [(11, 5, 2), (11,)],  # noqa
        "activation": ["tanh", "relu"],
        "solver": ["sgd", "adam"],
        "alpha": [0.0001, 0.05],
        "learning_rate": ["constant", "adaptive"],
    }
    # initiate gridsearch across our parameters
    clf = GridSearchCV(ann, parameter_space, n_jobs=-1, cv=kf)
    # fit the model with various permutations of the parameters
    clf.fit(df, y)
    # print out the Best parameter set
    print("Best parameters found:\n", clf.best_params_)


def question3():
    # use the function given to load my data in
    df = load()
    # create an array of labels from the data
    y = df["target"].to_numpy()
    # delete the label column from the dataset so
    # our classifiers dont try and make predictions off of it
    del df["target"]
    # initate all of our classifcation models
    # initiate our kfolds on our data
    kf = KFold(n_splits=5)
    # initiate a list to store scores
    annscores = []
    # initiate the model
    ann = MLPClassifier(
        hidden_layer_sizes=(11, 5, 2),
        activation="tanh",
        solver="adam",
        learning_rate="constant",
        alpha=0.05,
    )  # noqa
    # train and score the model for every k fold
    for train_index, test_index in kf.split(df, y):
        ann.fit(df.iloc[train_index], y[train_index])
        scores = ann.score(df.iloc[test_index], y[test_index])
        annscores.append(scores)
    # initiate all the KNN classifiers
    knn_1 = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    knn_5 = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn_10 = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
    # initiate a dictionary to hold onto cross validation values
    # for each classifier
    Results = {}
    # now calculate and append the cross validation values for 5 kfolds
    # for each classifier
    Results.update({"ann": 1 - np.mean(annscores)})  # noqa
    Results.update(
        {"knn1": 1 - np.mean(cross_validate(knn_1, df, y, cv=kf)["test_score"])}
    )  # noqa
    Results.update(
        {"knn5": 1 - np.mean(cross_validate(knn_5, df, y, cv=kf)["test_score"])}
    )  # noqa
    Results.update(
        {"knn10": 1 - np.mean(cross_validate(knn_10, df, y, cv=kf)["test_score"])}
    )  # noqa
    # turn our dictionary of results into a dataframe of 6
    # columns and one row
    Results = pd.DataFrame(Results, index=["err"], columns=Results.keys())
    # return the dataframe of results
    return Results


def question4():
    # load in the number recognition dataset
    data = loadmat(
        "/Users/k/Desktop/4th_Year/First_Semester/CSCI 444 - Machine Learning/Asn3/NumberRecognitionBiggest.mat"
    )  # noqa
    # store all the training and testing data
    cnn_x_train = data["X_train"]
    cnn_x_test = data["X_test"]
    cnn_y_train = data["y_train"]
    # do some data preprocessing for our models
    cnn_y_train = transpose(cnn_y_train)
    cnn_x_train = cnn_x_train.astype("float32") / 255
    cnn_x_test = cnn_x_test.astype("float32") / 255
    cnn_x_train = np.expand_dims(cnn_x_train, -1)
    cnn_x_test = np.expand_dims(cnn_x_test, -1)
    cnn_y_train = tf.keras.utils.to_categorical(cnn_y_train, NUM_CLASSES)

    # generate some new images that are slightly
    # different from the original ones
    # to improve training
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.10,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )  # noqa
    X_train3 = cnn_x_train[
        9,
    ].reshape((1, 28, 28, 1))
    Y_train3 = cnn_y_train[
        9,
    ].reshape((1, 10))
    for i in range(30):
        X_train2, Y_train2 = datagen.flow(X_train3, Y_train3).next()
        if i == 9:
            X_train3 = cnn_x_train[
                11,
            ].reshape((1, 28, 28, 1))
        if i == 19:
            X_train3 = cnn_x_train[
                18,
            ].reshape((1, 28, 28, 1))
    # initiate all the models and a list to hold all of them
    nets = 1
    model = [0] * nets
    for j in range(nets):
        # for every new model, include nonlinear convolution layers,
        # learnable pooling layers, ReLU activation,
        # dropout, batch normalization, and adam optimization
        model[j] = Sequential()
        model[j].add(
            Conv2D(32, kernel_size=3, activation="relu", input_shape=(28, 28, 1))
        )  # noqa
        model[j].add(BatchNormalization())
        model[j].add(Conv2D(32, kernel_size=3, activation="relu"))
        model[j].add(BatchNormalization())
        model[j].add(
            Conv2D(32, kernel_size=5, strides=2, padding="same", activation="relu")
        )  # noqa
        model[j].add(BatchNormalization())
        model[j].add(Dropout(0.4))

        model[j].add(Conv2D(64, kernel_size=3, activation="relu"))
        model[j].add(BatchNormalization())
        model[j].add(Conv2D(64, kernel_size=3, activation="relu"))
        model[j].add(BatchNormalization())
        model[j].add(
            Conv2D(64, kernel_size=5, strides=2, padding="same", activation="relu")
        )  # noqa
        model[j].add(BatchNormalization())
        model[j].add(Dropout(0.4))

        model[j].add(Conv2D(128, kernel_size=4, activation="relu"))
        model[j].add(BatchNormalization())
        model[j].add(Flatten())
        model[j].add(Dropout(0.4))
        model[j].add(Dense(10, activation="softmax"))

        # comile the new model with adam optimzer and entropy cost
        model[j].compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )  # noqa
        # ignore stochatic gradient decent learning rate
        annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
    # initiate a list to hold onto the history
    # of each network
    history = [0] * nets
    # initiate the number of epochs for each network
    epochs = 5
    # initiate a list to store error rates
    cvresults = []
    for j in range(nets):
        # train and evaluate the models
        print("\n" + "model number: " + str(j + 1))
        X_train2, X_val2, Y_train2, Y_val2 = train_test_split(
            cnn_x_train, cnn_y_train, test_size=0.1
        )  # noqa
        history[j] = model[j].fit(
            datagen.flow(X_train2, Y_train2, batch_size=64),
            epochs=epochs,
            steps_per_epoch=X_train2.shape[0] // 64,
            validation_data=(X_val2, Y_val2),
            callbacks=[annealer],
            verbose=1,
        )  # noqa
        # store the error of each model
        cvresults.append(history[j].history["val_accuracy"])
    # flatten the lost of scores
    cvresults = flatten(cvresults)
    # calculate the average error rate
    err = 1 - np.mean(cvresults)
    # save the error rate to our output json file
    np.save(
        Path(__file__).resolve().parent / "kfold_cnn.npy",
        err,
        allow_pickle=False,
        fix_imports=False,
    )  # noqa
    # ensemble the predictions
    results = np.zeros((cnn_x_test.shape[0], 10))
    for j in range(nets):
        # add the predictions from each model to our list
        results = results + model[j].predict(cnn_x_test)
    # process the results of our predictions into a series for some cool
    # display later
    results = np.argmax(results, axis=1)
    results = pd.Series(results, name="Label")
    # save the results as a numpy array as well for submission
    y_pred = np.array(results)
    # save the results to our output json file
    np.save(
        Path(__file__).resolve().parent / "predictions.npy",
        y_pred,
        allow_pickle=False,
        fix_imports=False,
    )  # noqa

    # do some cool plotting to show the test images and their
    # predicted labels
    plt.figure(figsize=(15, 6))
    for i in range(40):
        plt.subplot(4, 10, i + 1)
        plt.imshow(cnn_x_test[i].reshape((28, 28)), cmap=plt.cm.binary)
        plt.title("predict=%d" % results[i], y=0.9)
        plt.axis("off")
    plt.subplots_adjust(wspace=0.3, hspace=-0.1)
    plt.show()


"""
save_mnist_kfold(question1())
save_data_kfold(question3())
print(question1())
print(question3())
print(question3())
isort.file('/Users/k/Desktop/4th_Year/First_Semester/CSCI 444 - Machine Learning/Asn3/assign3.py')  # noqa
"""
question4()
