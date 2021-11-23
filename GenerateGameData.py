import os, sys
import random
import time
import numpy as np
from difflib import SequenceMatcher

polynomialDegree = 10
gamesPerSeason = 162
minimumGameWorth = 0.2 #if the game did not take place at least 20% of the way through the season then use the previous seasons stats
fieldSeperator = ","
findStatsFromOtherYears = False #about 5% of game samples cant be used when this is True, compared to 25% when it is False
numPlayerStats = 16 #
numPitcherStats = 19 #
add_reverse_games = False #if True adds each game twice with reverse order
ignoredStats = []

startYear, endYear, max_samples = 0,0,-1 #change these in BBModel

gamesLoaded = 0
gamesDisregarded = 0

#dictionary to convert team code into team name
teams = dict({"ATL": "Atlanta Braves",
    "LAN": "Los Angeles Dodgers",
    "LAD": "Los Angeles Dodgers",
    "LA": "Los Angeles Dodgers",
    "NYN": "New York Mets",
    "NYM": "New York Mets",
    "SDN": "San Diego Padres",
    "SD": "San Diego Padres",
    "NYA": "New York Yankees",
    "NYY": "New York Yankees",
    "SFN": "San Francisco Giants",
    "SF": "San Francisco Giants",
    "PHI": "Philadelphia Phillies",
    "CHA": "Chicago White Sox",
    "CWS": "Chicago White Sox",
    "BOS": "Boston Red Sox",
    "WS4": "Washington Nationals",
    "WS5": "Washington Nationals",
    "WS7": "Washington Nationals",
    "WSU": "Washington Nationals",
    "WAS": "Washington Nationals",
    "WSH": "Washington Nationals",
    "TOR": "Toronto Blue Jays",
    "LAA": "Los Angeles Angels",
    "CAL": "Los Angeles Angels",
    "ANA": "Los Angeles Angels",
    "BL2": "Baltimore Orioles",
    "BL3": "Baltimore Orioles",
    "BLN": "Baltimore Orioles",
    "BLA": "Baltimore Orioles",
    "BAL": "Baltimore Orioles",
    "MIN": "Minnesota Twins",
    "HOU": "Houston Astros",
    "OAK": "Oakland Athletics",
    "COL": "Colorado Rockies",
    "CN1": "Cincinnati Reds",
    "CN2": "Cincinnati Reds",
    "CNU": "Cincinnati Reds",
    "CIN": "Cincinnati Reds",
    "KCA": "Kansas City Royals",
    "TB": "Kansas City Royals",
    "KC": "Kansas City Royals",
    "CHN": "Chicago Cubs",
    "CHC": "Chicago Cubs",
    "ARI": "Arizona Diamondbacks",
    "MIA": "Miami Marlins",
    "FLO": "Florida Marlins",
    "FLA": "Florida Marlins",
    "MLU": "Milwaukee Brewers",
    "ML3": "Milwaukee Brewers",
    "MLA": "Milwaukee Brewers",
    "MIL": "Milwaukee Brewers",
    "DET": "Detroit Tigers",
    "SLN": "St. Louis Cardinals",
    "SL4": "St. Louis Cardinals",
    "STL": "St. Louis Cardinals",
    "CLE": "Cleveland Indians",
    "SEA": "Seattle Mariners",
    "TEX": "Texas Rangers",
    "PT1": "Pittsburgh Pirates",
    "MON": "Montreal Expos",
    "PIT": "Pittsburgh Pirates" })

def build_player_pitcher_dicts(home, visiting):
    players = {}
    pitchers = {}
    #iterate for every player
    for pi in range(18):
        if (pi == 0): #homePitcher
            player = prepareField(home[1]).lower()
            team = prepareField(home[0]).lower()
            numStats = numPitcherStats
            startIndex = 0
            #add to pitchers dictionary
            pitchers[player] = (team, numStats, startIndex)
        elif (pi == 9): #visitingPitcher
            player = prepareField(visiting[1]).lower()
            team = prepareField(visiting[0]).lower()
            numStats = numPitcherStats
            startIndex = numPitcherStats*polynomialDegree + numPlayerStats*polynomialDegree*8
            #add to pitchers dictionary
            pitchers[player] = (team, numStats, startIndex)
        elif (pi < 9): #homePlayer
            player = prepareField(home[pi+1]).lower()
            team = prepareField(home[0]).lower()
            numStats = numPlayerStats
            startIndex = numPitcherStats*polynomialDegree + numPlayerStats*polynomialDegree*(pi-1)
            #add to players dictionary
            players[player] = (team, numStats, startIndex)
        else: #visitingPlayer
            player = prepareField(visiting[pi-8]).lower()
            team = prepareField(visiting[0]).lower()
            numStats = numPlayerStats
            startIndex = 2*numPitcherStats*polynomialDegree + numPlayerStats*polynomialDegree*(pi-2)
            #add to players dictionary
            players[player] = (team, numStats, startIndex)
    return players, pitchers

def findStats(players, playerStatSheet, _discardedSamples, year, stats, i, forWebapp=False): #find the stats for one list of players in a dictionary with their corresponding index in stats matrix
    if (forWebapp): stats = [ stats ]
    try:
        #iterate through each row to find the players stats
        for row in range(len(playerStatSheet)):
            if (len(players) == 0):
                break
            playerStats = playerStatSheet[row]

            #adding our game worth stat to stats array, this is temporary game worth should be a hyperparameter later (make the worth of this training sample less)
            individualStats = playerStats.split(",")[2:]
            tryTeam = teams.get(playerStats.split(",")[1])
            if (str(tryTeam) == "None" and not forWebapp):
                print("Missing team ID: " + playerStats.split(",")[1] + "  " + player + str(year))
            else:
                if not forWebapp: tryTeam = tryTeam.lower()
                else: tryTeam = ""
            tryPlayer = playerStats.split(",")[0].lower()

            #iterate through each player still in the dictionary
            for (player, (team, numStats, startIndex)) in players.items():
                # if row corresponds to correct player name
                if (player.split(" ")[-1] in tryPlayer and player.split(" ")[0] in tryPlayer) or (similarity(tryPlayer, player) > 0.85) and similarity(team, tryTeam) > 0.8:
                    if similarity(tryPlayer, player) < 0.6: continue
                    for s in range(numStats):
                        if (s in ignoredStats): continue
                        try:
                            float(individualStats[s])
                        except Exception as e:
                            print("failed try statement in for loop in findstats: " + e)
                            return False
                        for p in range(polynomialDegree):
                            stats[i-_discardedSamples][startIndex + p + polynomialDegree*s] = float(individualStats[s])**(p+1)

                    players.pop(player, None)
                    break
        if len(players) > 0:
            print("couldnt find all players. remaining: " + str(players))
            return False
        else: return True
    except Exception as e:
        print("failed try statement in findstats: " + e)
        return False

        

# function to turn a dataset of baseball games into a file containing stats and score differences
# for every game in our data
def createTrainingData(set, statusTitle=""):

    if max_samples < 0:
        data_selection = set
    else:
        data_selection = set[:max_samples]

    #create empty array for stats and scores
    stats = np.zeros( (len(data_selection), 2*numPitcherStats*polynomialDegree+16*numPlayerStats*polynomialDegree) ) #first 15*10*9 stats are polynimalized stats of home team, last 15*10*9 are visiting team
    scores = np.zeros(len(data_selection))

    #count how many training samples had to be skipped because of errors or other
    discardedSamples = 0

    #variable to store start time of function to predict how long computation will take
    startTime = time.time()

    log = open("log.txt", "a")

    #iterate through each game
    for i in range(len(data_selection)):

        game = data_selection[i]
        #get information about the game
        year = int(game.split(",")[0])
        if year < startYear or year > endYear:
            discardedSamples += 1
            continue

        home = game.split(",")[11:-1]
        visiting = game.split(",")[1:-11]
        scoreDifference = int(game.split(",")[21])
        
        #get stat sheet from correct year, plus a year before and after so we can find in other seasons worst case
        playerStatSheet = open("data/mlbPlayerStats" + str(year) + ".txt", encoding="ISO-8859-1").readlines()[1:]
        pitcherStatSheet = open("data/mlbPitcherStats" + str(year) + ".txt", encoding="ISO-8859-1").readlines()[1:]
        if year > startYear and findStatsFromOtherYears:
            playerStatSheet += open("data/mlbPlayerStats" + str(year-1) + ".txt", encoding="ISO-8859-1").readlines()[1:]
            pitcherStatSheet += open("data/mlbPitcherStats" + str(year-1) + ".txt", encoding="ISO-8859-1").readlines()[1:]
        if year < endYear and findStatsFromOtherYears:
            playerStatSheet += open("data/mlbPlayerStats" + str(year+1) + ".txt", encoding="ISO-8859-1").readlines()[1:]
            pitcherStatSheet += open("data/mlbPitcherStats" + str(year+1) + ".txt", encoding="ISO-8859-1").readlines()[1:]

        players, pitchers = build_player_pitcher_dicts(home, visiting)
        success = findStats(players, playerStatSheet, discardedSamples, year, stats, i)
        success1 = findStats(pitchers, pitcherStatSheet, discardedSamples, year, stats, i)
        
        if len(pitchers) > 0:
            log.write("no pitcher: " + str(pitchers) + " - " + str(year))
        if len(players) > 0:
            log.write("no PLAYER: " + str(players) + " - " + str(year))
        elif not success or not success1:
            log.write("Error in stats. Sample discarded.")
            discardedSamples += 1
        else:
            #save the outcome of the game
            scores[i-discardedSamples] = scoreDifference

        #print completion status
        if (i%100 == 0):
            timePassed = time.time() - startTime
            ratio = float((i/len(data_selection)))
            if ratio == 0: ratio = 0.0001
            totalTime = timePassed * (1.0 / ratio)
            print(statusTitle + "  >  Processing games: %.2f" % float((i/len(data_selection)*100)) + "% " + (str(discardedSamples) + " samples discarded, " + str((i+1)-discardedSamples) + " games processed. (of " + str(len(data_selection)) + ")") + (" Time left:  %.0f" % ((totalTime-timePassed)/60.0) + "m%.0f" % ((totalTime-timePassed)%60) + "s"))

    print("Done")
    log.close()
    return stats[:-discardedSamples], scores[:-discardedSamples]

def prepareField(input): #adds spaces before any capital letters and removes dashes
    output = ""
    for i in range(len(input)):
        letter = input[i]
        if i != 0 and letter.isupper() and input[i-1] != " ":
            output += " "
        output += letter
    #replace steven with steve to account for steven tolleson's stats being under the name steve tolleson
    #replace osvaldo with ozzie to account for osvaldi martinez's stats being under the name ozzie martinez
    output = output.replace('-', ' ')
    output = output.replace('Steven', 'Steve')
    output = output.replace('Osvaldo', 'Ozzie')
    return output

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

#check whether list contains 0
def doesntContainZero(s):
    for field in s.split(","):
        if "0" in str(field):
            return False
    return True

#creates a file containing the stat and score information with all the input data
def writeToFile(x, y, filename):
    file = open(filename, "w")
    for i in range(len(x)): #for each game record
        for j in range(len(x[i])): #for each stat
            file.write(str(x[i][j]) + fieldSeperator) #add the stat and a seperator between fields
        file.write(str(y[i]) + "\n") #add scoreDifference and newline after every entry
        if i % 200 == 0:
            print("Writing data to " + filename + " : %.2f" % float((i/len(x)*100)) + "%")

    file.close()

#returns stats and scores arrays when given the filename of a previously generated data set
def loadFromFile(filename, shuffle_data):
    try:
        if max_samples < 0:
            data = open(filename).readlines()
        else:
            data = open(filename).readlines()[:max_samples]
    except:
        print("Error: loadFromFile() input data file not found.")
        return None, None, 1
    
    if (shuffle_data):
        random.shuffle(data)

    #create empty arrays to fill
    stats = np.zeros( (len(data), 2*numPitcherStats*polynomialDegree + 16*numPlayerStats*polynomialDegree) ) #first 16*5 stats are polynimalized stats of home team, last 16*5 are visiting team
    scores = np.zeros(len(data))
    
    #for each data entry
    for i in range(len(data)):
        #get the individual fields
        fields = data[i].split(fieldSeperator)
        for j in range(len(fields)):
            if fields[j] != "":
                #write them to the stats array except for the last field which goes to the scores array
                if (j != len(stats[0])): stats[i][j] = float(fields[j])
                else: scores[i] = float(fields[j])
        #status update
        if i % 200 == 0:
            print("Reading data from " + filename + " : %.2f" % float((i/len(data)*100)) + "%")

    return stats, scores, 0

#############################################################################################

def processGames():
    gameData = ""
    gamesLoaded = 0
    gamesDisregarded = 0

    sY = 1990
    eY = 2020

    #generate txt file for all the game data
    newFileName = os.getcwd() + "/data/ALLGAMES_" + str(sY) + "-" + str(eY) + ".txt"
    try:
        data = open(newFileName, 'r').readlines
        if (len(data) > 300): return newFileName
    except:
        data = None

    for y in range(eY-sY+1):
        #getting data associated with current year
        filename = "records/GL" + str(sY+y) + ".TXT"
        print("Reading in games: %.2f" % float(100.0*y/(eY-sY+1)) + "%")
        #open data file and split into array of lines
        infile = open(filename, 'r')
        games = infile.read().split("\n")

        for i in range(len(games)):
            #divide line into different fields
            cur = games[i].split(',')
            
            #some games have incomplete data or aren't data entires so we skip them
            if len(cur) < 11:
                gamesDisregarded = gamesDisregarded + 1
                continue
            
            newData = ""

            year = int(cur[0].strip('"')[:4]) #get rid of the quotes around numbers and words
            visitorScore = int( cur[9].strip('"') )
            homeScore = int( cur[10].strip('"') )
            homeTeam = teams.get( cur[6].strip('"'), "" )
            visitingTeam = teams.get( cur[3].strip('"'), "" )

            visitingPlayers = [0]*9
            homePlayers = [0]*9

            homePlayers[0] = cur[104].strip('"')
            visitingPlayers[0] = cur[102].strip('"')
            
            for k in range(9):
                if len(cur) == 165 or len(cur) == 177 or len(cur) == 171:
                    player = cur[107 + (3*k)].strip('"')
                    defensivePosition = cur[108 + (3*k)].strip('"')
                else:
                    player = cur[106 + 3*k].strip('"')
                    defensivePosition = cur[107 + 3*k].strip('"')

                try:
                    pos = int(defensivePosition) #sometimes the game file is incorrectly formatted so we check to make sur defensivePosition is a number
                except:
                    gamesDisregarded = gamesDisregarded + 1
                    continue
                if (pos > 0 and pos < 10): visitingPlayers[pos-1] = player

            for k in range(9):
                if len(cur) == 165 or len(cur) == 177 or len(cur) == 171:
                    player = cur[134 + (3*k)].strip('"')
                    defensivePosition = cur[135 + (3*k)].strip('"')
                else:
                    player = cur[133 + (3*k)].strip('"')
                    defensivePosition = cur[134 + (3*k)].strip('"')
                
                try:
                    pos = int(defensivePosition) #sometimes the game file is incorrectly formatted so we check to make sur defensivePosition is a number
                except:
                    gamesDisregarded = gamesDisregarded + 1
                    continue
                if (pos > 0 and pos < 10): homePlayers[pos-1] = player

            homePlayerList = ""
            visitingPlayerList = ""

            for p in homePlayers:
                homePlayerList += str(p)
                homePlayerList += ","

            for p in visitingPlayers:
                visitingPlayerList += str(p)
                visitingPlayerList += ","

            #calculate gameWorth
            gamesPlayedHome = int(cur[8].strip('"'))
            gamesPlayedVisitors = int(cur[5].strip('"'))
            avgGamesPlayed = (gamesPlayedHome + gamesPlayedVisitors)/2.0
            gameWorth = avgGamesPlayed/gamesPerSeason

            if gameWorth < minimumGameWorth and year != 1990:
                year = year - 1

            #make sure the data isn't incomplete and isn't too early in season then add to our gameData
            if (visitorScore != "" and homeScore != "" and homeTeam != "" and visitingTeam != "" and doesntContainZero(homePlayerList) and doesntContainZero(visitingPlayerList)):
                newData += str(year) + "," + homeTeam + "," + homePlayerList + visitingTeam + "," + visitingPlayerList + str(homeScore-visitorScore)
                gameData += newData + '\n'
                gamesLoaded = gamesLoaded + 1
                if (add_reverse_games): #add the game again but in reverse order
                    newData = str(year) + "," + visitingTeam + "," + visitingPlayerList + homeTeam + "," + homePlayerList + str(visitorScore-homeScore)
                    gameData += newData + '\n'  
                    gamesLoaded = gamesLoaded + 1
            else:
                gamesDisregarded = gamesDisregarded + 1

        infile.close()

    try:
        os.remove(newFileName)
    except:
        donothing="nothing"

    output = open(newFileName, "w")
    #indicate which fields are which variables
    output.write("date,homeTeam,homePitcher,homePlayers2-9,visitingTeam,visitingPitcher,visitingPlayers2-9,scoreDifference\n")
    output.write(gameData)
    output.close()
    
    print(str(gamesLoaded) + " games loaded.")
    print(str(gamesDisregarded) + " games disregarded.")
    print("DONE")
    return newFileName

def createTrainingFiles(newFileName):
    
    #get lines of the output data except for title and shuffle them randomly
    lines = open(newFileName).readlines()[1:]
    random.shuffle(lines)

    #split our shuffled data into training, test, and validations data sets (50%, 25%, 25%)    
    trainingSet = lines[:len(lines)//2] #write half of data to training
    testSet = lines[3*len(lines)//4:] #write a quarter of data to test 
    validationSet = lines[len(lines)//2:-len(lines)//4] #write a quarter of data to validation

    # Next we convert our game data into an array of stats and score differences which we will save to a file to avoid having
    # to compute it again later

    trainingStats, trainingScores = createTrainingData(trainingSet, "TRAINING SET")
    writeToFile(trainingStats, trainingScores, "data/TrainingData" + str(startYear) + "-" + str(endYear) + ".txt")
    testStats, testScores = createTrainingData(testSet, "TEST SET")
    writeToFile(testStats, testScores, "data/TestData" + str(startYear) + "-" + str(endYear) + ".txt")
    validationStats, validationScores = createTrainingData(validationSet, "VALIDATION SET")
    writeToFile(validationStats, validationScores, "data/ValidationData" + str(startYear) + "-" + str(endYear) + ".txt")

    print(str(len(trainingStats) + len(testStats) + len(validationStats)) + " games loaded.")
    print(str(len(lines) - (len(trainingStats) + len(testStats) + len(validationStats))) + " games disregarded.")
    print("DONE")


def setup():
    gameDataFile = "/data/ALLGAMES_1990-2020.txt"
    try:
        data = open(gameDataFile, 'r')
    except:
        gameDataFile = processGames()
    createTrainingFiles(gameDataFile)

if len(sys.argv) == 2:
    setup()