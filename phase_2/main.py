import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

"""
Just dump a data frame to std out 
"""
def _dump_frame(df):
    for row in df.iterrows():
        print(row)


"""
Get training and testing data for the naive bayes 
model 
"""
def get_test_and_train(results_df, years):
    salaries = []
    wins     = []
    teams = results_df['team'].unique()

    for year in years:
        for team in teams:
            players   = results_df[(results_df['team'] == team) & (results_df['year'] == year)]
            offensive = players[players['position'].isin(["S", "OL", "DT", "DE", "DL", "DB", "CB"])]

            # - sum up total salary for the year
            salaries.append(offensive['salary'].sum())
            wins.append(players['team_record'].unique()[0].split("-")[0])

    return np.array([salaries]).reshape(-1, 1), np.array(wins)


"""
Run a naive bayes classifier on the  
data set to determine predict the 
number of wins in a season given the amount 
of salary cap spent on defensive positions. 

We expect an increase in games won as 
the defensive spending on players increases  
"""
def naive_bayes(results_df):
    years_train = [year for year in results_df['year'].unique() if year < 2018]
    years_test  = [2018]

    # - get training and testing data
    salary_train, wins_train = get_test_and_train(results_df, years_train)
    salary_test, wins_test   = get_test_and_train(results_df, years_test)

    # - train the naive bayes model
    gnb        = GaussianNB()
    model      = gnb.fit(salary_train, wins_train)
    pred_2018  = model.predict(salary_test)

    # - create the accuracy distribution
    dist = [0] * 14
    for pred, acc in zip(pred_2018, wins_test):
        diff = abs(int(acc) - int(pred))
        dist[diff] += 1

    defensive_spending = [val for val in range(1_000_000, 15_000_000, 100_000)]

    trend_data = np.array(defensive_spending).reshape(-1, 1)
    trend      = model.predict(trend_data)

    # - plot the model accuracy distribution
    plt.plot([i for i in range(0, 14)], dist)
    plt.title(f"Absolute difference in predicted vs actual wins during the 2018 regular season")
    plt.xlabel("Absolute difference in wins")
    plt.ylabel("Frequency")
    plt.show()

    # - plot the predicted number of wins given defensive spending
    plt.scatter(trend, defensive_spending)
    plt.title("Relationship of defensive spending and wins per season")
    plt.ylabel("Amount spent on defense (USD)")
    plt.xlabel("Predicted number of wins")
    plt.show()


def main():
    results_df = pd.read_csv("../data_sets/results.csv")

    naive_bayes(results_df)
    return


if __name__ == "__main__":
    main()
