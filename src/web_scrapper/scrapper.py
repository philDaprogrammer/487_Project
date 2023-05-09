import requests
import logging
import pandas as pd

from bs4 import BeautifulSoup


""" years and teams we want to get """
years = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']
teams = {'arizona-cardinals': 'ARI', 'atlanta-falcons': "ATL", 'baltimore-ravens': "BAL", 'buffalo-bills': "BUF", 'carolina-panthers': "CAR",
         'chicago-bears': "CHI", 'cincinnati-bengals': "CIN", 'cleveland-browns': "CLE", 'dallas-cowboys': "DAL", 'denver-broncos': "DEN",
         'detroit-lions': "DET", 'green-bay-packers': "GB", 'houston-texans': "HOU", 'indianapolis-colts': "IND", 'jacksonville-jaguars': "JAX",
         'kansas-city-chiefs': "KC", 'las-vegas-raiders': "OAK", 'los-angeles-chargers': "LAC", 'los-angeles-rams': "LAR", 'miami-dolphins': "MIA",
         'minnesota-vikings': "MIN", 'new-england-patriots': "NE", 'new-orleans-saints': "NO", 'new-york-giants': "NYG", 'new-york-jets': "NYJ",
         'philadelphia-eagles': "PHI", 'pittsburgh-steelers':"PIT", 'san-francisco-49ers': "SF", 'seattle-seahawks': "SEA",
         'tampa-bay-buccaneers': "TB",'tennessee-titans': "TEN", 'washington-commanders': "WAS"}


""" 
Get all salary cap tables for each nfl team from the 2011 to 2018 seasons. 
We retrieve all the salary tables by requesting the html pages they reside within
all scrapped pages are placed in the web_scrapped directory  
"""
def get_salary_tables_html():
    """
    Payload for https://www.spotrac.com/signin/submit/ endpoint. Most of the tables are behind a paywall.
    But we can take advantage of the fact that signing in takes a redirect parameter to load a new
    page after login.
    """
    payload = {"redirect": "", "email": "SENSITIVE", "password": "SENSITIVE"}

    for team in sorted(teams.keys()):
        for year in years:
            """ set redirect to desired salary table page  """
            payload['redirect'] = f"nfl/{team}/cap/{year}"

            try:
                resp = requests.post("https://www.spotrac.com/signin/submit/", data=payload)

                if resp.status_code != 200:
                    print(f"Bad response for {team}_{year}: {resp}")
                    continue

                f = open(f"scrapped/{team}_{year}.html", "w")
                f.write(resp.text)
                f.close()
                print(f"successfully scrapped {team}_{year}")

            except ConnectionError:
                print(f"failed to get data for {team}:{year} -> {logging.exception('')}")


def extract_data():
    """
    Relevant data:

    Index 2   -> name
    Index 4   -> position
    Index 5   -> cap
    Index 12  -> Dead cap
    Index 13  -> cap hit
    """
    salaries = []

    for team in sorted(teams.keys()):
        for year in years:
            soup = BeautifulSoup(open(f"scrapped/{team}_{year}.html", "r").read(), "html.parser")

            for ent in soup.find_all("tr"):
                fields = ent.text.split("\n")

                # - only thing we need to test for
                if len(fields) == 16:
                    needed = [fields[2], teams[team], year, fields[4], fields[5], fields[13]]
                    salaries.append(needed)

    salaries_df = pd.DataFrame(data=salaries, columns=["playerName", "team", "year", "position", "cap", "capHit"])
    salaries_df.to_csv("../data_sets/much_better_nlf_salaries.csv")


def main():
    extract_data()
    return


if __name__ == "__main__":
    main()
