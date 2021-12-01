# MAIS 202 - Data Collection

# Import packages
import os, sys
from bs4 import BeautifulSoup
import requests

startYear = 2021
endYear = 2021

## Helper funcitons
def isUpper(l):
	if l is None:
		return False
	return (l.upper() == l) and (l.lower() != l.upper())

def uniqueList(list1):
	output = []
	for x in list1:
		if (x not in output) and (containsNum(x)):
			output.append(x)
	return output

def containsNum(string):
	for i in string: 
		if i in "0123456789|/":
			return False
	return True

def strSlicer(string):
	firstChar = string[0]
	stringJoined = "".join(string.split(" ")) # "1TorontoBlueJaysBlueJays1|090239029302"
	stringList = []
	startIdx = 0

	for i in range(len(stringJoined)):
		if isUpper(stringJoined[i]):
			stringList.append(stringJoined[startIdx: i])
			startIdx = i

	#print(stringList)
	stringList = uniqueList(stringList[1:])
	return " ".join(stringList) # ['Toronto' 'Blue' 'Jays' 'Blue' 'Jays1|9019']

def getTeamNames(soup):
	teamNames = []
	isTeam = False
	## Try and find table rows and col 
	for link in soup.find_all("th"):
		if (link.text[0] == "1"): # Team name headers begin with numbers
			isTeam = True
		if not isTeam:
			continue

		newText = strSlicer(link.text)
		teamNames.append(newText)
	return teamNames

def getTableHeaders(soup, writeFile):
	# Get table headers
	headerList = []
	for headerRow in soup.find("thead").find("tr").find_all("th"):
		#.find("thead").find("tr").
		headerText = str(headerRow.text)
		if isUpper(headerText):
			headerList.append(headerText[:len(headerText)//2])
	return headerList

def getTableToCsv(soup, isElem, fileWrite):
	# Search in soup for 
	for link in soup.find("tbody").find_all("tr"):
		# Team name headers begin with numbers
		if not isElem:
			break
		fileWrite.write(strSlicer(link.find("th").text))

		for i in link.find_all("td"):
			fileWrite.write(",")
			fileWrite.write(i.text)
		fileWrite.write("\n")

def statsToCSV(soup, year, getHeader=True, getPlayers=False, getPitchers=False, appending=False):
	# Defaults to getting team data rather than player data
	# Open output text file depending on what data you are looking for
	if getPitchers:
		fileName = "data/mlbPitcherStats" + str(year) + ".txt"
	elif getPlayers:
		fileName = "data/mlbPlayerStats" + str(year) + ".txt"
	else:
		fileName = "data/mlbTeamStats" + str(year) + ".txt"

	if appending:
		teamStats = open(fileName, "a")
	else:
		teamStats = open(fileName, "w")

	# Get table headers
	if getHeader:
		headerList = getTableHeaders(soup, teamStats)
		teamStats.write(",".join(headerList))
		teamStats.write("\n")

	# Get team names
	teamNames = getTeamNames(soup)
	
	# Get data and input into csv
	isElem = True
	if (soup.find("tbody") == None):
		isElem = False
	elif (soup.find("tbody").find("tr") == None):
		isElem = False
	else:
		getTableToCsv(soup, isElem, teamStats)

	teamStats.close()
	return isElem


def getStatsForYear(year, getPlayers=False, getPitchers=False):
	# Create url
	if getPlayers:
		url = "https://www.mlb.com/stats/player/games/"
	elif getPitchers:
		url = "https://www.mlb.com/stats/player/pitching/games/"
	else:
		url = "https://www.mlb.com/stats/team/games/"
	urlYear = url + str(year) + "/regular-season"

	# Set local variables
	iter = 1
	boolVar = True
	firstIter = True
	while boolVar:
		# Get HTML and put through beautiful soup
		if (getPitchers or getPlayers):
			urlYear = url + str(year) + "/regular-season" + "?page=" + str(iter)
			print("url:  " + urlYear)
		
		# Create soup and input to csv
		response = requests.get(urlYear)
		soup = BeautifulSoup(response.text, "html.parser")
		boolVar = statsToCSV(soup, year, getHeader=firstIter, getPitchers=getPitchers, getPlayers=getPlayers, appending=(not firstIter))
		
		firstIter = False
		iter += 1
		if (not getPitchers) and (not getPlayers):
			boolVar = False

#getStatsForYear(2021, getPitchers=True)
def statsToCSVMultYears(years):
	for i in years:
		getStatsForYear(i, getPlayers=True)
		getStatsForYear(i, getPitchers=True)

def scrape(year):
	getStatsForYear(year, getPlayers=True)
	getStatsForYear(year, getPitchers=True)

def start(startYear, endYear):
	yearsList = []
	for i in range(endYear-startYear):
		#print(startYear + i)
		yearsList.append(startYear + i)
	if (startYear == endYear): yearsList = [ startYear ]
	statsToCSVMultYears(yearsList)

def setYearRange(sY, eY):
	endYear = eY
	startYear = sY

if __name__ == '__main__':
    start(startYear,endYear)