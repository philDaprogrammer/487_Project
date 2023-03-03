import csv

import pandas
import pandas as pd

""" 
add each times record at the end of each game
"""
def add_records(reader):
    # - cleaning (1) filter out 2018 games
    start = "2017-09-10"
    end   = "2017-12-31"

    mask        = (reader['game_date'] >= start) & (reader['game_date'] <= end)
    date_ranges = reader.loc[mask]

    records     = {}
    away        = []
    home        = []

    # add each teams record
    for i, row in date_ranges.iterrows():

        if row["home_team"] not in records:
            records[row["home_team"]] = [0, 0, 0]
        if row["away_team"] not in records:
            records[row["away_team"]] = [0, 0, 0]

        if row["total_home_score"] > row["total_away_score"]:
            records[row["home_team"]][0] += 1
            records[row["away_team"]][2] += 1
        elif row["total_home_score"] < row["total_away_score"]:
            records[row["home_team"]][2] += 1
            records[row["away_team"]][0] += 1
        else:
            records[row["home_team"]][1] += 1
            records[row["away_team"]][1] += 1

        home.append(records[row["home_team"]])
        away.append(records[row["away_team"]])

    home_df = pd.DataFrame(home, columns=["home_team_wins", "home_team_ties", "home_team_loses"])
    away_df = pd.DataFrame(away, columns=["away_team_wins", "away_team_ties", "away_team_loses"])
    print(len(date_ranges), len(home_df), len(away_df))

    return pandas.concat([date_ranges.reset_index(), home_df, away_df], axis=1)

""" 
call 'func' on each row in the data set.
Enumerates the header fields as well.  
"""
def iter_csv(f_name, func, *args, **kargs):
    with open(f_name, "r") as f:
        reader = csv.reader(f)

        # - enumerate the header indices
        for i, header in enumerate(next(reader)):
            print(i, header, end=", ")
        print("\n")

        for line in reader:
            if args and kargs:
                func(line, *args, **kargs)
            elif args:
                func(line, *args)
            elif kargs:
                func(line, **kargs)
            else:
                func(line)


""" 
Data cleaning ideas and some EDA, we need 10 in total for each
   * We will need to drop certain columns 
      
   * Organize players by team (EDA)
   
   * Organize players by position (EDA)
   
   * Combine the two data sets
   
   * Filter only games from 2017 - 2018 (playoffs)
 
   * Map abbreviated team names to actual team names 
   
   * Need to filter drives where a score occurs 
   
   * we will need to create a way of actually calculating if a team won or not 
     this is done by adding up all scores for each time. IDK if this is cleaning or EDA,
     but im leaning more towards cleaning
   
   * lots of data is un-formatted this will probably be 
     3 to 5 steps on its own.  
              
"""
def main():
    file = "../datasets/results.csv"
    reader = pd.read_csv(file)
    added = add_records(reader)

    print(len(added))
    for i, row in added.iterrows():
        print(row, end="\n\n")

    #iter_csv(file, print, end="\n\n")
    return


if __name__ == "__main__":
    main()
