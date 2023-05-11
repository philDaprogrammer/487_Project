import pandas as pd
import warnings
from os.path import exists
import matplotlib.pyplot as plt


team_map = {
    "jets": "NYJ", "browns": "CLE", "bears": "CHI", "tennessee": "TEN", "kansascity": "KC",
    "bucs": "TB", "hawks3": "SEA", "patriots": "NE", "49erslogo": "SF","washington": "WAS",
    "dolphins": "MIA", "colts": "IND", "buffalo": "BUF", "texans": "HOU", "arizona2": "ARI",
    "eagles1": "PHI", "lions": "DET", "rams2": "LA", "broncos": "DEN", "vikings": "MIN",
    "panthers": "CAR", "falcons": "ATL", "oakland": "OAK", "Pittsburgh-Steelers-logo-psd22874": "PIT",
    "jaguars": "JAX", "neworleans": "NO", "dallas": "DAL", "chargers2": "LAC", "NFL_Chargers_logo.svg_": "LAC",
    "ravens": "BAL", "bengals": "CIN", "nygiants": "NYG", "packers": "GB"
}

names_map = {
    "Johnathan Ford": "John Ford", "Deshawn Shead": "DeShawn Shead",
    "Jake Schum": "Jacob Schum", "Donavan Clark": "Donavon Clark",
    "Denzell Good": "Denzelle Good", "Adam-Pacman Jones": "Adam Jones",
    "LaDarius Gunter": "Ladarius Gunter", "Travis Carrie": "T.J. Carrie",
    "John Cyprien": "Johnathan Cyprien", "Seth Devalve": "Seth DeValve",
    "Chris Milton": "Christopher Milton"
}


"""
parse salary and total cash features 
"""
def parse_int(line: str):
    num = line.replace("$", "").replace(",", "").replace("-", "")
    return 0 if num == "" else int(num)


"""
parse a players full name
"""
def split_name(name):
    names = name.split(" ")
    return (names[0], names[1] + " " + names[2]) if len(names) > 2 else (names[0], names[1])


""" 
I have to read the file in chunks, or my cpu gets bricked
"""
def read_play_by_play(file_name):
    games_df = pd.DataFrame()

    for chunk in pd.read_csv(file_name, chunksize=2000):
        # - cleaning (10): Drop redundant columns
        filtered = chunk[['play_id', 'game_id', 'home_team', 'away_team', 'total_home_score', 'total_away_score', 'game_date']]
        games_df = pd.concat([games_df, filtered])

    return games_df


"""
make sure the team name abbreviations are consistent in 
the play by play data set
"""
def fix_team_names(games_df):
    mapping = {"STL": "LA", "JAC": "JAX", "SD": "LAC"}

    games_df['home_team'] = games_df['home_team'].replace(mapping)
    games_df['away_team'] = games_df['away_team'].replace(mapping)
    return games_df


def get_unique_games(games_df):
    game_ids       = games_df['game_id'].unique()
    filtered_games = pd.DataFrame()

    for uid in game_ids:
        temp           = games_df[games_df['game_id'] == uid]
        filtered_games = pd.concat([filtered_games, temp.tail(1)])

    return filtered_games


"""
clean the play by play data set and calculate 
each teams record for a given season
"""
def clean_records(file_name: str):
    print("Getting team records from 2011 to 2018 ... ")

    games_by_year = {}
    games_df      = read_play_by_play(file_name)

    # - cleaning (11): some team name abbreviations are incorrect
    games_df = fix_team_names(games_df)

    # - cleaning (12): get only unique games (last score in the final quarter)
    unique_games = get_unique_games(games_df)

    # - calculate all games scores from 2009 to 2018 season
    for i, row in unique_games.iterrows():
        year = str(row["game_id"])[0:4]

        if year not in games_by_year:
            games_by_year[year] = {}
        if row['home_team'] not in games_by_year[year]:
            games_by_year[year][row['home_team']] = [0, 0, 0]
        if row['away_team'] not in games_by_year[year]:
            games_by_year[year][row['away_team']] = [0, 0, 0]

        if row["total_home_score"] > row["total_away_score"]:
            games_by_year[year][row["home_team"]][0] += 1
            games_by_year[year][row["away_team"]][2] += 1
        elif row["total_home_score"] < row["total_away_score"]:
            games_by_year[year][row["home_team"]][2] += 1
            games_by_year[year][row["away_team"]][0] += 1
        else:
            games_by_year[year][row["home_team"]][1] += 1
            games_by_year[year][row["away_team"]][1] += 1

    return games_by_year


"""
clean the salaries data, all cleaning operations are commented 
within the function
"""
def clean_salaries(salaries: str, players: str):
    print("Cleaning salaries data set and adding player positions ... ")

    salaries_rd = pd.read_csv(salaries)
    players_rd  = pd.read_csv(players)
    positions   = []

    # - cleaning (1): drop redundant columns
    salaries_rd = salaries_rd[["playerName", "totalCash", "team", "salary", "year"]]
    # - cleaning (2): only consider years from 2009 to 2018
    salaries_rd = salaries_rd[(salaries_rd['year'] >= 2011) & (salaries_rd['year'] <= 2018)]
    # - cleaning (3): remove dollar signs and commas from salary feature
    salaries_rd['salary']     = salaries_rd['salary'].apply(lambda v:  parse_int(v))
    # - cleaning (4): remove dollar sings and comma's from total cash
    salaries_rd['totalCash']  = salaries_rd['totalCash'].apply(lambda v: parse_int(v))
    # - cleaning (5): map team names to correct ones
    salaries_rd['team']       = salaries_rd['team'].replace(team_map)
    # - cleaning (6): fix up any incorrectly formatted player names
    salaries_rd['playerName'] = salaries_rd['playerName'].replace(names_map)
    # - cleaning (7) we have some erroneous white space that we need to get rid of
    players_rd['nameLast']    = players_rd['nameLast'].apply(lambda v: v.rstrip())
    # - cleaning (8) EDA showed some players have a 0 salary, well just ignore these entries
    filtered_rd = salaries_rd[salaries_rd['salary'] > 0]

    # - Cleaning (9): add the position feature to the data set
    for name in filtered_rd['playerName']:
        first, last = split_name(name)
        player_mask = (players_rd["nameFirst"] == first) & (players_rd["nameLast"] == last)
        positions.append(players_rd[player_mask]['position'].to_list()[0])

    return pd.concat([filtered_rd.reset_index(), pd.DataFrame(positions, columns=["position"])], axis=1)


"""
Merge the play by play data set and 
the nfl salary data set. 

Note: to run this, you'll probably 
need to change the file paths, as im 
sure the structure will be different 
then my local
"""
def create_set(read_cache=True):
    play_by_play = "../../data_sets/NFL Play by Play 2009-2018 (v5).csv"
    nfl_salaries = "../../data_sets/nfl_salaries.csv"
    players      = "../../data_sets/players.csv"
    results      = "../../data_sets/results.csv"

    if not exists(results) or not read_cache:
        print("Constructing the new data set")
        # - cleaning (13) combine the two data sets into a singular data set
        salaries_df   = clean_salaries(nfl_salaries, players)
        games_by_year = clean_records(play_by_play)
        records       = []

        for i, row in salaries_df.iterrows():
            record = games_by_year[str(row['year'])][row['team']]
            records.append(f"{record[0]}-{record[2]}-{record[1]}")

        final_df = pd.concat([salaries_df, pd.DataFrame(records, columns=["team_record"])], axis=1)
        final_df.to_csv(results)
    else:
        print("Getting data set from Disk")
        final_df = pd.read_csv(results)

    print("Done", end="\n\n")
    return final_df


""" 
EDA 1.) 
Some teams do not have entries from 2009 to 2010, 
so we will drop these now, and only consider 
entries from 2011 to 2018. We need to ensure 
the records are correct. 
"""
def team_season_wins(results):
    years = sorted(results['year'].unique())
    teams = results['team'].unique()

    """ graph each teams records """
    for team in teams:
        wins   = []

        for year in years:
            mask = (results['year'] == year) & (results['team'] == team)
            wins.append(str(results[mask]['team_record'].unique()[0]).split("-")[0])

        plt.plot(years, list(map(int, wins)))
        plt.title(f"{team} wins from 2011 to 2018")
        plt.xlabel("years")
        plt.ylabel("wins")
        plt.show()


""" 
EDA 2) 
Like the previous step, we really do not want to consider 2009 
and 2010. lets get a rough idea of team salaries throughout 
the years 
"""
def team_season_salaries(results):
    years = sorted(results['year'].unique())
    teams = results['team'].unique()

    for team in teams:
        caps  = []

        for year in years:
            mask = (results['year'] == year) & (results['team'] == team)
            caps.append(results[mask]['salary'].sum())

        plt.plot(years, caps)
        plt.title(f"{team} salary caps from 2011 to 2018")
        plt.xlabel("years")
        plt.ylabel("dollars spent")
        plt.show()


"""
EDA 3) 

Amount of salary cap spent on offensive 
vs defensive positions. This an important 
as we should consider if skewing the 
cap distribution towards either end 
results in a better record 
"""
def offensive_vs_defensive(results):
    # - pandas does some unnecessary complaining
    # - regarding index remapping
    warnings.simplefilter("ignore")

    teams = results['team'].unique()
    offensive = ["WR", "OT", "OG", "C", "TE", "QB", "FB", "RB"]
    defensive = ["S", "OL", "DT", "DE", "DL", "DB", "CB"]

    for team in teams:
        off  = results[results['team'] == team][results['position'].isin(offensive)]
        deff = results[results['team'] == team][results['position'].isin(defensive)]

        print(f"{team} total offensive cap distribution {off['salary'].sum()}")
        print(f"{team} total defensive cap distribution {deff['salary'].sum()}")
        print("\n")


""" 
EDA 4) 

Show the relationship between salary cap 
and wins to see if there is a correlation. 
"""
def cap_and_wins(results):
    years   = sorted(results["year"].unique())
    teams   = results['team'].unique()

    for team in teams:
        records = []
        caps    = []

        for year in years:
            team_mask   = (results['team'] == team ) & (results['year'] == year)
            year_record = results[team_mask]['team_record'].unique()[0]
            total_spent = results[team_mask]['salary'].sum()

            records.append(year_record)
            caps.append(total_spent)

        plt.scatter(records,  caps)
        plt.title(f"{team} salary cap vs year record")
        plt.xlabel("record")
        plt.ylabel("cap spent")
        plt.show()


"""
EDA 5.) 
 
Do some basic statistics regarding 
each teams salary. Some teams have a min salary 
of zero, which makes no sense. So these entries will 
be removed as well
"""
def basic_salary_stats(results):
    teams = results['team'].unique()

    for team in teams:
        one_team = results[results['team'] == team]
        print(one_team['salary'].describe(), end="\n\n")


""" 
usage: python3 phase1.py  

"""
def main():
    """
        set read_cache=False if you want to
        generate a new CSV file instead
        of reading it from disk. It takes
        roughly 20 secs for the csv file
        to be generated. On subsequent
        executions, the data file
        can be read from disk to reduce
        exec time. If the  file doesn't
        exist, it will be generated regardless
        of the cache argument
    """
    results = create_set(read_cache=True)

    # - my 5 distinct EDA processes
    team_season_wins(results)
    team_season_salaries(results)
    offensive_vs_defensive(results)
    cap_and_wins(results)
    basic_salary_stats(results)
    return


if __name__ == "__main__":
    main()
