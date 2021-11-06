import os, sys
import random
import numpy as np

max_samples = -1 #maximum number of games we use for training (negative for infinite)
polynomialDegree = 10
gamesPerSeason = 162
numStats = 15 + 2 #+2 for the game worth stat this is temporary
startYear = 1990
endYear = 2020
fieldSeperator = "?"

#dictionary to convert team code into team name
teams = dict({"ATL": "Atlanta Braves",
    "LAN": "Los Angeles Dodgers",
    "NYN": "New York Mets",
    "SDN": "San Diego Padres",
    "NYA": "New York Yankees",
    "SFN": "San Francisco Giants",
    "PHI": "Philadelphia Phillies",
    "CHA": "Chicago White Sox",
    "BOS": "Boston Red Sox",
    "WS4": "Washington Nationals",
    "WS5": "Washington Nationals",
    "WS7": "Washington Nationals",
    "WSU": "Washington Nationals",
    "WAS": "Washington Nationals",
    "TOR": "Toronto Blue Jays",
    "LAA": "Los Angeles Angels",
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
    "CHN": "Chicago Cubs",
    "ARI": "Arizona Diamondbacks",
    "MIA": "Miami Marlins",
    "FLO": "Florida Marlins",
    "MLU": "Milwaukee Brewers",
    "ML3": "Milwaukee Brewers",
    "MLA": "Milwaukee Brewers",
    "MIL": "Milwaukee Brewers",
    "DET": "Detroit Tigers",
    "SLN": "St. Louis Cardinals",
    "SL4": "St. Louis Cardinals",
    "CLE": "Cleveland Indians",
    "SEA": "Seattle Mariners",
    "TEX": "Texas Rangers",
    "PT1": "Pittsburgh Pirates",
    "PIT": "Pittsburgh Pirates" })

# function to turn a dataset of baseball games into a file containing stats and score differences
# for every game in our data
def createTrainingData(set):
    
    if max_samples < 0:
        data_selection = set
    else:
        data_selection = set[:max_samples]

    #create empty array for stats and scores
    stats = np.zeros( (len(data_selection), 2*numStats*polynomialDegree) ) #first 16*5 stats are polynimalized stats of home team, last 16*5 are visiting team
    scores = np.zeros(len(data_selection))

    #get minimum and maximum value for each stat (for standardization)
    mins, maxs = findMinMaxFromStatSheet(open("data/mlbStats2021.txt").readlines())

    #iterate through each game
    for i in range(len(data_selection)):
        game = data_selection[i]

        #get information about the game
        year = int(game.split(",")[0][:4])
        homeTeam = game.split(",")[1]
        visitingTeam = game.split(",")[2]
        scoreDifference = float(game.split(",")[3])
        gameWorthHome = float(game.split(",")[4])/gamesPerSeason
        gameWorthVisitors = float(game.split(",")[4])/gamesPerSeason

        #toggle variables for if the team's stats are found (1) or not (0)
        home = 0
        visiting = 0
        
        #get stat sheet from correct year
        statSheet = open("data/mlbStats" + str(year) + ".txt").readlines()

        #iterate through each row to find the teams we need
        for j in range(len(statSheet)):
            teamStats = statSheet[j]

            #adding our game worth stat to stats array, this is temporary game worth should be a hyperparameter later (make the worth of this training sample less)
            individualStats = teamStats.split(",")[3:]
            individualStats.append(gameWorthHome)
            individualStats.append(gameWorthVisitors)

            #check for home team
            if teamStats.split(",")[0] == homeTeam and home != 1:

                home = 1
                n = 0 #start position in stats array (first half is home team, second half is visiting team)

                for s in range(numStats):
                   #standardize stat using corresponding min and max
                    stdStat = float(standardize(float(individualStats[s]), mins[s], maxs[s]))

                    #polynomialize the stats
                    for p in range(polynomialDegree):
                        stats[i][n] = stdStat**p
                        n += 1

            elif teamStats.split(",")[0] == visitingTeam and visiting != 1:

                visiting = 1
                #second half of stats[i] is visiting team stats and their polynomials 
                n = (numStats+1)*polynomialDegree

                for s in range(numStats-1):
                   #standardize stat using corresponding min and max
                    stdStat = float(standardize(float(individualStats[s]), mins[s], maxs[s]))

                    #polynomialize the stats
                    for p in range(polynomialDegree):
                        stats[i][n] = stdStat**p
                        n += 1

        # make sure we found the correct team's stats
        if visiting == 0: print(str(year) + "incomplete stats: " + visitingTeam)
        if home == 0: print(str(year) + " incomplete stats: " + homeTeam)

        #save the outcome of the game
        scores[i] = scoreDifference

        #print completion status
        if (i%300 == 0):
            print("Processing games: %.2f" % float((i/len(data_selection)*100)) + "%")
    return stats, scores


#find minimums and maximums for each stat from a given stat sheet
def findMinMaxFromStatSheet(s):
    maxs = [0]*(numStats)
    mins = [100000]*(numStats) #100000 so that all stats will be less than it

    #iterate through the rows of stats
    for i in range(len(s[1:])):
        #seperate each row into individual stats
        line = s[1:][i]
        for j in range(len(line.split(",")[3:])):
            #for each stat save new maximum and minimum value
            maxs[j] = max(float(line.split(",")[3:][j]), maxs[j])
            mins[j] = min(float(line.split(",")[3:][j]), mins[j])

    return mins, maxs

#creates a file containing the stat and score information with all the input data
def writeToFile(x, y, filename):
    file = open(filename, "w")
    for i in range(len(x)): #for each game record
        for j in range(len(x[i])): #for each stat
            file.write(str(x[i][j]) + fieldSeperator) #add the stat and a seperator between fields
        file.write(str(y[i]) + "\n") #add a newline after every entry
        if i % 300 == 0:
            print("Writing data to " + filename + " : %.2f" % float((i/len(x)*100)) + "%")

    file.close()


def standardize(stat, min, max):
    return float(stat-min)/float(max-min) #scaling each feature between 0 and 1

#returns stats and scores arrays when given the filename of a previously generated data set
def loadFromFile(filename, max_entries=-6990):
    if max_entries == -6990: max_entries = max_samples

    try:
        if max_entries < 0:
            data = open(filename).readlines()
        else:
            data = open(filename).readlines()[:max_entries]
    except:
        print("Error: loadFromFile() input data file not found.")
        return [], []
        

    #create empty arrays to fill
    stats = np.zeros( (len(data), 2*numStats*polynomialDegree) ) #first 16*5 stats are polynimalized stats of home team, last 16*5 are visiting team
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
        if i % 300 == 0:
            print("Reading data from " + filename + " : %.2f" % float((i/len(data)*100)) + "%")

    return stats, scores

#############################################################################################

def generate():
    #variable for the file name
    gameData = ""

    for y in range(endYear-startYear+1):
        #getting data associated with current year
        filename = "records/GL" + str(startYear+y) + ".TXT"
        print("Reading in games: %.2f" % float(100.0*y/(endYear-startYear+1)) + "%")
        #open data file and split into array of lines
        infile = open(filename, 'r')
        games = infile.read().split("\n")


        for i in range(len(games)):
            #divide line into different fields
            cur = games[i].split(',')
            
            #some games have incomplete data or aren't data entires so we skip them
            if len(cur) < 11: continue
            
            newData = ""

            date = cur[0].strip('"') #get rid of the quotes around numbers and words
            visitorScore = int( cur[9].strip('"') )
            homeScore = int( cur[10].strip('"') )
            homeTeam = teams.get( cur[6].strip('"'), "" )
            visitingTeam = teams.get( cur[3].strip('"'), "" )
            gamesPlayedHome = int(cur[8].strip('"'))
            gamesPlayedVisitors = int(cur[5].strip('"'))

            #make sure the data isn't incomplete then add to our gameData
            if (date != "" and visitorScore != "" and homeScore != "" and homeTeam != "" and visitingTeam != "" and gamesPlayedHome != "" and gamesPlayedVisitors != ""):
                newData += date + "," + homeTeam + "," + visitingTeam + "," + str(homeScore-visitorScore) + "," + str(gamesPlayedHome) + "," + str(gamesPlayedVisitors)
                gameData += newData + "\n"
        infile.close()

    #generate txt file for all the game data
    newFileName = os.getcwd() + "/data/ALLGAMES_" + str(startYear) + "-" + str(endYear) + ".txt"
    os.remove(newFileName)
    output = open(newFileName, "w")
    #indicate which fields are which variables
    output.write("date,homeTeam,visitingTeam,scoreDifference,gamesPlayedInSeason\n")
    output.write(gameData)
    output.close()

    #get lines of the output data except for title and shuffle them randomly
    lines = open(newFileName).readlines()[1:]
    random.shuffle(lines)

    #split our shuffled data into training, test, and validations data sets (50%, 25%, 25%)    
    trainingSet = lines[:len(lines)//2] #write half of data to training
    testSet = lines[3*len(lines)//4:] #write a quarter of data to test 
    validationSet = lines[len(lines)//2:-len(lines)//4] #write a quarter of data to validation

    # Next we convert our game data into an array of stats and score differences which we will save to a file to avoid having
    # to compute it again later

    trainingStats, trainingScores = createTrainingData(trainingSet)
    testStats, testScores = createTrainingData(testSet)
    validationStats, validationScores = createTrainingData(validationSet)


    writeToFile(trainingStats, trainingScores, "data/TrainingData" + str(startYear) + "-" + str(endYear) + ".txt")
    writeToFile(testStats, testScores, "data/TestData" + str(startYear) + "-" + str(endYear) + ".txt")
    writeToFile(validationStats, validationScores, "data/ValidationData" + str(startYear) + "-" + str(endYear) + ".txt")

    print("DONE")


if len(sys.argv) > 1:
    generate()