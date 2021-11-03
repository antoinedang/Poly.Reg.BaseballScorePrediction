import os, sys
import random

#Create a variable for the file name

gameData = ""

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
    "TBA": "Tampa Bay Rays",
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
    "FLO": "Miami Marlins",
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

startYear = 1990
endYear = 2020

for y in range(endYear-startYear+1):
    filename = "records/GL" + str(startYear+y) + ".TXT"
    print(filename)
    infile = open(filename, 'r')
    games = infile.read().split("\n")

    for i in range(len(games)):
        cur = games[i].split(',')
        if len(cur) < 11: continue
        game = ""
        date = cur[0].strip('"')
        visitorScore = int( cur[9].strip('"') )
        homeScore = int( cur[10].strip('"') )
        homeTeam = teams.get( cur[6].strip('"'), "" )
        visitingTeam = teams.get( cur[3].strip('"'), "" )
        if (visitingTeam != "" and homeTeam != ""):
            game += date + "," + homeTeam + "," + visitingTeam + "," + str(homeScore-visitorScore)
            gameData += game + "\n"
    infile.close()

#open text file
newFileName = os.getcwd() + "/data/ALLGAMES_" + str(startYear) + "-" + str(endYear) + ".txt"
output = open(newFileName, "w")
 
#write string to file
output.write("date,homeTeam,visitingTeam,scoreDifference")
output.write(gameData)
 
#close file
output.close()

#shuffle lines of file randomly
lines = open(newFileName).readlines()
random.shuffle(lines)

#split our shuffled data into training, test, and validations data sets
trainingFileName = os.getcwd() + "/data/TRAINING_" + str(startYear) + "-" + str(endYear) + ".txt"
testFileName = os.getcwd() + "/data/TEST_" + str(startYear) + "-" + str(endYear) + ".txt"
validationFileName = os.getcwd() + "/data/VALIDATION_" + str(startYear) + "-" + str(endYear) + ".txt"
open(trainingFileName, 'w').writelines(lines[:len(lines)//2])
open(testFileName, 'w').writelines(lines[3*len(lines)//4:])
open(validationFileName, 'w').writelines(lines[len(lines)//2:-len(lines)//4])