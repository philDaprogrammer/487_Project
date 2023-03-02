import csv
import pandas as pd


def get_game_score(f_name, ID):
    """
    if line[1] == gameID:
        print(line[0:10], line[16])

        if line[22] == "1":
            scores[line[16]] += 6
        if line[23] == "Made":
            scores[line[16]] += 1
        if line[24] != "Good":
            scores[line[16]] += 2
        if line[54] == "Good":
            scores[line[16]] += 3
    """


def filter_game(row, *args):
    gameID = args[0]

    if row[1] == gameID:
        print(row[71], row[72])


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
    file = "../data_sets/NFL Play by Play 2009-2018 (v5).csv"
    # reader = pd.read_csv(file)
    iter_csv(file, print, end="\n\n")

    # game_1 = reader[reader["GameID"] == "2009091000"]
    # print(game_1)
    return


if __name__ == "__main__":
    main()
