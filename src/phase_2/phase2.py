import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm


"""
helper function to get accuracy for various models
"""
def get_accuracy(predictions, actuals):
    dist = [0] * 15

    for pred, acc in zip(predictions, actuals):
        diff = abs(int(acc) - int(pred))
        dist[diff] += 1

    return dist


""" 
attempt to get some optimal hyper parameters for our 
multi-layer perceptron
"""
def get_mean_errors(X_train, Y_train, X_test, Y_test):
    warnings.filterwarnings("ignore")

    solvers    = ["sgd", "lbfgs", "adam"]
    activators = ["logistic", "tanh", "identity", "relu"]

    for solver in solvers:
        for activator in activators:
            mlp = MLPClassifier(solver=solver, activation=activator, random_state=1, max_iter=10_000)
            pred = mlp.fit(X_train, Y_train).predict(X_test)

            # - get accuracy distribution
            dist  = get_accuracy(pred, Y_test)
            total = 0

            for i, ent in enumerate(dist):
                total += i * ent

            print(f"MlpClassifier(solver={solver}, activation={activator}) mean error {total / 14}")


""" 
split the training and testing years 
"""
def get_train_and_test_years(results_df):
    years_train = [year for year in results_df['year'].unique() if year < 2018]
    years_test  = [2018]

    return years_train, years_test


""" 
helper function used to get naive bayes data 
"""
def get_bayes_data(results_df, year, team):
    players   = results_df[(results_df['team'] == team) & (results_df['year'] == year)]
    offensive = players[players['position'].isin(["S", "OL", "DT", "DE", "DL", "DB", "CB"])]

    # - sum up total salary for the year and return record
    return offensive['salary'].sum(), players['team_record'].unique()[0].split("-")[0]


"""
helper function used to get multi-layer perceptron data  
"""
def get_mlp_data(results_df, year, team):
    players = results_df[(results_df['team'] == team) & (results_df['year'] == year)]

    offensive = players[players['position'].isin(["WR", "OT", "OG", "C", "TE", "QB", "FB", "RB"])]
    defensive = players[players['position'].isin(["S", "OL", "DT", "DE", "DL", "DB", "CB"])]
    off_tot = offensive['salary'].sum()
    def_tot = defensive['salary'].sum()

    return [off_tot, def_tot], players['team_record'].unique()[0].split("-")[0]


""" 
helper function used to get support vector machine data 
"""
def get_svm_data(results_df, year, team):
    players = results_df[(results_df['team'] == team) & (results_df['year'] == year)]
    rb_sal = players[players['position'] == "QB"]['salary'].sum()
    qb_sal = players[players['position'] == "RB"]['salary'].sum()

    more_than_10 = 1 if int(players['team_record'].unique()[0].split("-")[0]) >= 10 else 0

    return [rb_sal, qb_sal], more_than_10


"""
Get training and testing data for various models. 

Takes a function that splits data into training 
examples and labels. The way the data is split
depends on the function provided. 
"""
def get_train_and_test(results_df, years, data_func):
    X = []
    y = []
    teams = results_df['team'].unique()

    for year in years:
        for team in teams:
            example, label = data_func(results_df, year, team)
            X.append(example)
            y.append(label)

    return np.array(X), np.array(y)


"""
Run a naive bayes classifier on the  
data set to determine predict the 
number of wins in a season given the amount 
of salary cap spent on defensive positions. 
"""
def naive_bayes(f_name, verbose=False):
    results_df = pd.read_csv(f_name)

    years_train, years_test = get_train_and_test_years(results_df)

    # - get training and testing data
    X_train, Y_train = get_train_and_test(results_df, years_train, get_bayes_data)
    X_test, Y_test   = get_train_and_test(results_df, years_test, get_bayes_data)

    X_train = X_train.reshape(-1, 1)
    X_test  = X_test.reshape(-1, 1)

    # - train the naive bayes model
    gnb = GaussianNB()
    model = gnb.fit(X_train, Y_train)

    if verbose:
        pred_2018 = model.predict(X_test)
        # - create the accuracy distribution
        dist = get_accuracy(pred_2018, Y_test)

        # - plot the model accuracy distribution
        plt.plot([i for i in range(0, 15)], dist)
        plt.title(f"Absolute difference in predicted vs actual wins during the 2018 regular season")
        plt.xlabel("Absolute difference in wins")
        plt.ylabel("Frequency")
        plt.show()

    # - create the accuracy distribution
    defensive_spending = [val for val in range(1_000_000, 15_000_000, 100_000)]

    trend_data = np.array(defensive_spending).reshape(-1, 1)
    trend = model.predict(trend_data)

    # - plot the predicted number of wins given defensive spending
    fig = plt.figure()
    ax  = fig.add_subplot()

    ax.title.set_text("Naive bayes")
    ax.scatter(trend, defensive_spending)
    ax.set_ylabel("Amount spent on defense (USD)")
    ax.set_xlabel("Predicted number of wins")
    return fig


""" 
Support vector machine model used to predict 
if a team will obtain more than 10 wins in a 
given season based on running-back and 
quarter-back spending
"""
def support_vm(f_name, verbose=False):
    results_df = pd.read_csv(f_name)

    train_years, test_years = get_train_and_test_years(results_df)
    X_train, y_train = get_train_and_test(results_df, train_years, get_svm_data)
    X_test, y_test = get_train_and_test(results_df, test_years, get_svm_data)

    clf = svm.SVC(kernel="rbf")
    model = clf.fit(X_train, y_train)
    pred = model.predict(X_test)

    if verbose:
        accuracy = (pred == y_test).sum() / len(pred)
        print(f"\n svm accuracy: {accuracy}")

    qb_true = []
    rb_true = []
    qb_false = []
    rb_false = []

    for x, label in zip(X_test, pred):
        if label == 1:
            qb_true.append(x[0])
            rb_true.append(x[1])
        else:
            qb_false.append(x[0])
            rb_false.append(x[1])

    fig = figure()
    ax = fig.add_subplot()

    ax.scatter(qb_true, rb_true, marker="o")
    ax.scatter(qb_false, rb_false, marker="^")
    ax.set_xlabel("QB salary")
    ax.set_ylabel("RB salary")
    return fig


""" 
Multi-Layer perceptron classifier used to predict 
the number of wins a team will obtain in the
regular season based on different offensive 
and defensive spending distributions 
"""
def multi_lp(f_name, verbose=False):
    results_df = pd.read_csv(f_name)

    years_train, years_test = get_train_and_test_years(results_df)

    X_train, Y_train = get_train_and_test(results_df, years_train, get_mlp_data)
    X_test, Y_test = get_train_and_test(results_df, years_test, get_mlp_data)

    """     
    Try and find some optimality for our MLP hyper-parameters
    Set mean_errors to true if you want to see the mean errors
    for different hyper-parameters
    """
    if verbose:
        get_mean_errors(X_train, Y_train, X_test, Y_test)

    mlp = MLPClassifier(solver="sgd", activation="logistic", random_state=1, max_iter=500)
    classifier = mlp.fit(X_train, Y_train)

    cap = 30_000_000
    defensive_cap = []
    offensive_cap = []
    predictions   = []

    for i in range(1_000_000, 30_000_000, 1_000_000):
        cap_dist = np.array([cap - i, i]).reshape(1, -1)
        prediction = classifier.predict(cap_dist)

        offensive_cap.append(cap - i)
        defensive_cap.append(i)
        predictions.append(int(prediction[0]))

    fig = figure()
    ax  = fig.add_subplot(projection='3d')

    ax.scatter(offensive_cap, defensive_cap, predictions)
    ax.set_xlabel('Offensive cap')
    ax.set_ylabel('Defensive cap')
    ax.set_zlabel('Games won')
    return fig


"""
Functions in this file are called from both the main function and 
the backend module to obtain model results. Running the code from 
this file will produce more verbose output that pertains to hyper 
parameter tuning and accuracy of models. While code in the backend 
will simply save the figures to a buffer and send them to the front 
end

usage: 
    python3 phase2.py
        
"""
def main():
    f_name = "../../data_sets/results.csv"

    # - run ML algos on the dataset
    naive_bayes(f_name, verbose=True)
    plt.show()

    multi_lp(f_name, verbose=True)
    plt.show()

    support_vm(f_name, verbose=True)
    plt.show()
    return


if __name__ == "__main__":
    main()
