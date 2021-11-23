import DataCollection
import GenerateGameData
import numpy as np
import pickle

#display UI for user to select 9 players from list to assign to either team
user_selection = ("homePitcher","homePlayers2-9","visitingPitcher","visitingPlayers2-9")

refreshStats = False

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
    #get latest stats
    if refreshStats: DataCollection.scrape(2021)
    #compile and preprocess the stats

    playerStatSheet = open("data/mlbPlayerStats2021.txt", encoding="ISO-8859-1").readlines()[1:]
    pitcherStatSheet = open("data/mlbPitcherStats2021.txt", encoding="ISO-8859-1").readlines()[1:]
    players, pitchers = build_player_pitcher_dicts(playerList)

    testx = np.zeros( 2*GenerateGameData.numPitcherStats*GenerateGameData.polynomialDegree+16*GenerateGameData.numPlayerStats*GenerateGameData.polynomialDegree ) #first 15*10*9 stats are polynimalized stats of home team, last 15*10*9 are visiting team
    
    success = GenerateGameData.findStats(players, playerStatSheet, 0, 2021, testx, 0, True)
    success1 = GenerateGameData.findStats(pitchers, pitcherStatSheet, 0, 2021, testx, 0, True)

    if not success or not success1:
        print("Error in getting stats.")
        exit()

    scalerx = pickle.load(open('.scaler.sav', 'rb'))
    pca = pickle.load(open('.pca.sav', 'rb'))

    testx = scalerx.transform(testx.reshape(1,-1))
    testx = pca.transform(testx)
    #load model with presaved weights
    # load the model from disk
    loaded_model = pickle.load(open('.finalized_model.sav', 'rb'))
    result = float(loaded_model.predict(testx))

    if (result > 0): print("Predicted outcome: Team 1 wins!")
    else: print("Predicted outcome: Team 1 wins!")
    print("Predicted score difference: " + str(result))
    #run model and predict score
    #display to screen
    return result

score = predictScore( ["Bryan Shaw", "Tommy Pham", "Shohei M Ohtani", "Vladimir V Guerrero", "Salvador S Perez", "Austin A Riley", "Dansby D Swanson", "Bo B Bichette", "Tommy T Edman", "Mike M Mayers", "Kyle K Seager", "Paul P Goldschmidt", "Carlos C Santana", "Myles M Straw", "Nolan N Arenado", "Nathaniel N Lowe", "Rafael R Devers", "Robbie R Grossman" ] )