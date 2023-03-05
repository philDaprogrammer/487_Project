import pandas as pd
from os.path import exists

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


def parse_int(line: str):
    return line.replace("$", "").replace(",", "")


def split_name(name):
    names = name.split(" ")
    return (names[0], names[1] + " " + names[2]) if len(names) > 2 else (names[0], names[1])


""" 
I have to read the file in chunks, or my cpu gets bricked
"""
def read_play_by_play(file_name):
    games_df = pd.DataFrame()

    for chunk in pd.read_csv(file_name, chunksize=2000):
        # - cleaning (8): Drop redundant columns
        filtered = chunk[['play_id', 'game_id', 'home_team', 'away_team', 'total_home_score', 'total_away_score', 'game_date']]
        games_df = pd.concat([games_df, filtered])

    return games_df


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


def clean_records(file_name: str):
    print("Getting team records from 2009 to 2018 ... ")

    games_by_year = {}
    games_df      = read_play_by_play(file_name)

    # - cleaning (9): some team name abbreviations are incorrect
    games_df = fix_team_names(games_df)

    # - cleaning (10): get only unique games (last score in the final quarter)
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


def clean_salaries(salaries: str, players: str):
    print("Cleaning salaries data set and adding player positions ... ")

    salaries_rd = pd.read_csv(salaries)
    players_rd  = pd.read_csv(players)
    positions   = []

    # - cleaning (1): drop redundant columns
    salaries_rd = salaries_rd[["playerName", "totalCash", "team", "salary", "year"]]
    # - cleaning (2): only consider years from 2009 to 2018
    salaries_rd = salaries_rd[(salaries_rd['year'] >= 2009) & (salaries_rd['year'] <= 2018)]
    # - cleaning (3): remove dollar signs and commas from salary feature
    salaries_rd['salary']     = salaries_rd['salary'].apply(lambda v:  parse_int(v))
    # - cleaning (4): remove dollar sings and comma's from total cash
    salaries_rd['totalCash']  = salaries_rd['totalCash'].apply(lambda v: parse_int(v))
    # - cleaning (5): map team names to correct ones
    salaries_rd['team']       = salaries_rd['team'].replace(team_map)
    # - cleaning (6): fix up any incorrectly formatted player names
    salaries_rd['playerName'] = salaries_rd['playerName'].replace(names_map)

    # - Cleaning (7): add the position feature to the data set
    for name in salaries_rd['playerName']:
        first, last = split_name(name)
        player_mask = (players_rd["nameFirst"] == first) & (players_rd["nameLast"] == last)
        positions.append(players_rd[player_mask]['position'].to_list()[0])

    return pd.concat([salaries_rd.reset_index(), pd.DataFrame(positions, columns=["position"])], axis=1)


def create_set(cache=False):
    play_by_play = "../data_sets/NFL Play by Play 2009-2018 (v5).csv"
    nfl_salaries = "../data_sets/nfl_salaries.csv"
    players      = "../data_sets/players.csv"
    results      = "../data_sets/results.csv"

    if not exists(results) or cache:
        print("Constructing the new data set")
        # - cleaning (11) combine the two data sets into a singular data set
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


def main():
    create_set(cache=True)
    return


if __name__ == "__main__":
    main()
