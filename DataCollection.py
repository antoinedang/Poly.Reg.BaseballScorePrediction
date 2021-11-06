# MAIS 202 - Data Collection

# Import packages
import os, sys
from bs4 import BeautifulSoup
import requests

startYear = 1990
endYear = 2021

## Helper funcitons
def isUpper(l):
	if l is None:
		return False
	return (l.upper() == l) and (l.lower() != l.upper())

def uniqueList(list1):
	output = []
	for x in list1:
		if x not in output:
			output.append(x)
	return output

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

def statsToCSV(soup, year):
	# Get team names
	teamStats = []
	isTeam = False
	teamNames = getTeamNames(soup)
	#print(teamNames)

	fileName = "data/mlbStats" + str(year) + ".txt"
	teamStats = open(fileName, "w")

	# Get table headers
	headerList = []
	for headerRow in soup.find("thead").find("tr").find_all("th"):
		#.find("thead").find("tr").
		headerText = str(headerRow.text)
		if isUpper(headerText):
			headerList.append(headerText[:len(headerText)//2])
	teamStats.write(",".join(headerList))
	teamStats.write("\n")

	# Get data and input into csv
	iterator = 0
	for link in soup.find_all("tr"):
		# Team name headers begin with numbers
		if (link.text[0] == "1"): 
			isTeam = True
		if not isTeam:
			continue
		teamStats.write(teamNames[iterator])
		iterator += 1

		for i in link.find_all("td"):
			teamStats.write(",")
			teamStats.write(i.text)
		teamStats.write("\n")
	teamStats.close()

def getStatsForYear(year):
	# Create url
	url = "https://www.mlb.com/stats/team/"
	urlYear = url + str(year) + "/regular-season"

	# Get HTML and put through beautiful soup
	response = requests.get(urlYear)
	soup = BeautifulSoup(response.text, "html.parser")
	statsToCSV(soup, year)

#getStatsForYear(2021)
def statsToCSVMultYears(years):
	for i in years:
		getStatsForYear(i)

yearsList = []
for i in range(endYear-startYear+1):
	print(startYear + i)
	yearsList.append(startYear + i)
statsToCSVMultYears(yearsList)
