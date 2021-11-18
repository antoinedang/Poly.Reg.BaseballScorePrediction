Antoine Dangeard 260962884
Thomas Inman 260947857

make games reversed once they are seperated into training and test data so as not to have the same game twice in test vs training?

Run RunFromScratch.py from terminal to scrape the web for stats, process and organize the data into 2 arrays of data (features and outcomes) and run the sklearn model on the data.

to run GenerateGameData.py from terminal, add at least 1 argument after (arbitrary)
to run DataCollection.py from terminal, add at least 1 argument after (arbitrary)

Description of python code:
BBModel: gets training data from a file using GenerateGameData, then trains and evaluates our model with that data
GenerateGameData: processes and combines the game data and the player stats to create arrays "stats" and "scores" that can be saved/loaded to/from the file system.
DataCollection.py: scraped the MLB stats website to get the hitting and pitching stats of all players in between a range of years