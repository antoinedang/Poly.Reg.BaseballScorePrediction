# BaseballOutcomePredictor

Final project for McGill AI Society Intro to ML Bootcamp (Fall 2021). 

Training data retrieved from [RetroSheet](https://www.retrosheet.org/) and [MLBStats](https://www.mlb.com/stats/).

## Members

Antoine Dangeard 260962884
Thomas Inman 260947857

## Project description

Our project is a web app that tries to predict the outcome of a baseball game using only the names of the players on each team. We built
a ridge regression model using sklearn, got the stats using BeautifulSoup, and the web app's backend using Flask. We retrieved and processed the game data and MLB stats from RetroSheet and the MLB website, standardized it, performed PCA on it, and trained several different models on the data to find which one achieved the best results.

## Running the app

To run the web app, first install all packages in imports.txt

Then, if the .finalized_model.sav, .pca.sav, or .scaler.sav files are missing, run:

```
python3 BBModel.py
```

Then,

```
python3 app.py
```

Lastly, open a browser and navigate to your http://localhost:5000.


To run hyperparameter search simply do:
```
python3 BBModel.py [args]
```
Where [args] is one or more arbitrary command line arguments

## Repository organization

This repository contains the scripts used to scrape stats, process data, train the model, and build the web app.

1. reports/
	* deliverables submitted to the MAIS Intro to ML Bootcamp organizers
2. records/
	* Contains all the game data we have for each year (from retrosheet.org)
3. data/
	* Contains the processed game data (ALLDATA) and the training, test, and validation data sets for a specific year range
4. templates/
	* HTML template for landing page
5. BBModel.py
	* python script that creates and trains the model, and performs hyperparameter search
6. app.py
	* main python script to instantiate Flask server
6. GenerateGameData.py
	* python script that processes game records and loads/writes the training data sets to/from the filesystem
7. DataCollection.py
	* python script that scrapes the MLB stats for each player for a certain year or set of years
7. bestAccuracy.txt
	* txt file containing details about the model that acheived the best accuracy
7. bestMSE.txt
	* txt file containing details about the model that acheived the best MSE
7. modelResults.txt
	* txt file containing the details, accuracy and MSE of every model during training
