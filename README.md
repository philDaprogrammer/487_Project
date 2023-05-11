# 487 Data Science Project: 
 
   This project explores the relationship between cap space spending for 
   NFL teams and how it correlates to wins in the regular season. A good
   data set will contain data for numerous years; including player salary, 
   position, team, etc 
   


# Data Sets:
 Data sets used throughout the project. Links Below:
 
 * Found on reddit, data comes from "http://www.spotrac.com":

   https://docs.google.com/spreadsheets/d/1rds8LOv8t8HqtnM-OLzSbrNpdEy_8vUYAP39swye57I/edit#gid=0


 * Play by play 2009 - 2018: 

   https://www.kaggle.com/datasets/maxhorowitz/nflplaybyplay2009to2016?select=NFL+Play+by+Play+2009-2017+(v4).csv


 * Player positions:

   https://www.kaggle.com/datasets/toddsteussie/nfl-play-statistics-dataset-2004-to-present

# Project structure: 
```
   ├── data_sets
│   └── results.csv
├── src
│   ├── main.py
│   ├── phase_1
│   │   ├── EDA.ipynb
│   │   ├── Phase1.ipynb
│   │   └── phase1.py
│   ├── phase_2
│   │   ├── Modeling.ipynb
│   │   └── phase2.py
│   ├── phase_3
│   │   └── 487-demo
│   │       ├── angular.json
│   │       ├── package.json
│   │       ├── package-lock.json
│   │       ├── src
│   │       │   ├── app
│   │       │   │   ├── app.component.css
│   │       │   │   ├── app.component.html
│   │       │   │   ├── app.component.spec.ts
│   │       │   │   ├── app.component.ts
│   │       │   │   ├── app.module.ts
│   │       │   │   └── components
│   │       │   │       ├── dataset
│   │       │   │       │   ├── dataset.component.css
│   │       │   │       │   ├── dataset.component.html
│   │       │   │       │   ├── dataset.component.spec.ts
│   │       │   │       │   ├── dataset.component.ts
│   │       │   │       │   └── dataset.service.ts
│   │       │   │       ├── header
│   │       │   │       │   ├── header.component.css
│   │       │   │       │   ├── header.component.html
│   │       │   │       │   ├── header.component.spec.ts
│   │       │   │       │   └── header.component.ts
│   │       │   │       └── table
│   │       │   │           ├── table.component.css
│   │       │   │           ├── table.component.html
│   │       │   │           ├── table.component.spec.ts
│   │       │   │           └── table.component.ts
│   │       │   ├── assets
│   │       │   ├── favicon.ico
│   │       │   ├── index.html
│   │       │   ├── main.ts
│   │       │   └── styles.css
│   │       ├── tsconfig.app.json
│   │       ├── tsconfig.json
│   │       └── tsconfig.spec.json
│   └── web_scrapper
│       └── scrapper.py
└── user_datasets
    └── results.csv
```

results.csv is left as a test file 



# Running the code: 

   To run the code, the following must be installed on the users machine:
   python3, nodejs, npm, and angular. 
   
   All the required python packages are listed in the requirements.txt 
   file 

   After one has installed node and npm, angular can be installed 
   with the following commands: 

   ```
   npm install -g @angular/cli
   ```
   
   After the download has completed, one can navigate to the
   directory where the front-end code resides: 
   
   ```
   cd phase_3/487-demo/
   ```
   
   You will then need to install the required packages to serve 
   up the frontend, the command is: 

   ```
   npm install @angular-devkit/build-angular
   ```

   Finally, you can run: 
   
   ```
   ng serve
   ```

   To bring up the frontend up on port 4200. 
   
   If all the python requirements have been installed, 
   one can run 

   ```
   python3 main.py 
   ```

   and one can navigate to the browser and interact with the application
   

# Working with other datasets. 
   It could be challenging to find additional data on the 
   web for our project. The scrapper.py file attempts to 
   do some web scraping for data. Anyways, it will be easiest 
   to upload your own datasets to the UI in order to see results, 
   but it is very important that the headers and fields are 
   essentially error free. 
   