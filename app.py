# Import flask
import io
import torch
from flask import Flask, render_template, request
import DataCollection
import GenerateGameData
import numpy as np
import pickle
import random
import time
import json


# Run app
app = Flask(__name__)
  



## Code to compile before
minute = 60
hour = 60*minute
secondsForRefresh = hour*1
curSeason = 2021

app = Flask(__name__)

def refreshStats():
    try:
        lastExecTime = pickle.load(open('.last.sav', 'rb'))
    except:
        lastExecTime = 0

    currentTime = time.time()

    if currentTime-lastExecTime > secondsForRefresh:
        #get latest stats
        DataCollection.scrape(curSeason)
        pickle.dump(currentTime, open('.last.sav', 'wb'))

def build_player_pitcher_dicts(playerList):
    players = {}
    pitchers = {}
    #iterate for every player
    for pi in range(18):
        player = GenerateGameData.prepareField(playerList[pi]).lower()
        if (pi == 0): #homePitcher
            numStats = GenerateGameData.numPitcherStats
            startIndex = 0
            #add to pitchers dictionary
            pitchers[player] = ("ignore", numStats, startIndex)
        elif (pi == 9): #visitingPitcher
            numStats = GenerateGameData.numPitcherStats
            startIndex = GenerateGameData.numPitcherStats*GenerateGameData.polynomialDegree + GenerateGameData.numPlayerStats*GenerateGameData.polynomialDegree*8
            #add to pitchers dictionary
            pitchers[player] = ("ignore", numStats, startIndex)
        elif (pi < 9): #homePlayer
            numStats = GenerateGameData.numPlayerStats
            startIndex = GenerateGameData.numPitcherStats*GenerateGameData.polynomialDegree + GenerateGameData.numPlayerStats*GenerateGameData.polynomialDegree*(pi-1)
            #add to players dictionary
            players[player] = ("ignore", numStats, startIndex)
        else: #visitingPlayer
            numStats = GenerateGameData.numPlayerStats
            startIndex = 2*GenerateGameData.numPitcherStats*GenerateGameData.polynomialDegree + GenerateGameData.numPlayerStats*GenerateGameData.polynomialDegree*(pi-2)
            #add to players dictionary
            players[player] = ("ignore", numStats, startIndex)
    return players, pitchers

def predictScore(playerList):

    output = ""
    #compile and preprocess the stats

    playerStatSheet = open("data/mlbPlayerStats2021.txt", encoding="ISO-8859-1").readlines()[1:]
    pitcherStatSheet = open("data/mlbPitcherStats2021.txt", encoding="ISO-8859-1").readlines()[1:]
    players, pitchers = build_player_pitcher_dicts(playerList)

    testx = np.zeros( 2*GenerateGameData.numPitcherStats*GenerateGameData.polynomialDegree+16*GenerateGameData.numPlayerStats*GenerateGameData.polynomialDegree ) #first 15*10*9 stats are polynimalized stats of home team, last 15*10*9 are visiting team
    
    success = GenerateGameData.findStats(players, playerStatSheet, 0, 2021, testx, 0, True)
    success1 = GenerateGameData.findStats(pitchers, pitcherStatSheet, 0, 2021, testx, 0, True)

    if not success or not success1:
        output += "\nError in getting stats."
        return output

    scalerx = pickle.load(open('.scaler.sav', 'rb'))
    pca = pickle.load(open('.pca.sav', 'rb'))

    testx = scalerx.transform(testx.reshape(1,-1))
    testx = pca.transform(testx)
    #load model with presaved weights
    # load the model from disk
    loaded_model = pickle.load(open('.finalized_model.sav', 'rb'))
    result = float(loaded_model.predict(testx))

    if (abs(result) < 0.45):output += "\nPredicted outcome: Tie game!"
    elif (result > 0): output += ("\nPredicted outcome: Team 1 wins!")
    else: output += ("\nPredicted outcome: Team 2 wins!")
    output += ("\nPredicted score difference (rounded): " + str(round(result)))
    output += ("\nPredicted score difference (actual) " + str(result)[:5])
    #run model and predict score
    #display to screen
    return output

def getAllPitchers():
    statSheet = open("data/mlbPitcherStats2021.txt", encoding="ISO-8859-1").readlines()[1:]

    pitchers = {}
    for p in statSheet:
        pitchers[str(p.split(",")[0])] = 0

    return pitchers.keys() # [ "pitcher1", "pitcher2", ...., "pitcherN" ]

def getAllPlayers():
    statSheet = open("data/mlbPlayerStats2021.txt", encoding="ISO-8859-1").readlines()[1:]
    
    players = {}
    for p in statSheet:
        players[str(p.split(",")[0])] = 0

    return players.keys() # [ "player1", "player2", ...., "playerN" ]

def getRandomTeam():
    randomPitchers = random.sample(getAllPitchers(), 2)
    randomPlayers = random.sample(getAllPlayers(), 16)
    return [ randomPitchers[0], randomPlayers[0], randomPlayers[1], randomPlayers[2], randomPlayers[3], randomPlayers[4], randomPlayers[5], randomPlayers[6], randomPlayers[7], randomPitchers[1], randomPlayers[8], randomPlayers[9], randomPlayers[10], randomPlayers[11], randomPlayers[12], randomPlayers[13], randomPlayers[14], randomPlayers[15] ]


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data", methods=["POST", "GET"])
def data():
    playerDict = dict()

    print(type(request.data))
    print(request.data)

    return playerDict

'''
@app.route("/")
def homePage():
    refreshStats()

    #assembles random teams and guesses outcome of game
    randomTeam = getRandomTeam()
    output = ""
    output += ("Team 1: \n")
    for x in randomTeam[:9]:
        output += (x + "\n")
        
    output += ("\n")
    output += ("Team 2: \n")
    for x in randomTeam[9:]:
        output += (x + "\n")

    output += ("\n")
    output += ("\n")
    
    output += predictScore( randomTeam )
    return output
'''

app.run(debug=False)  