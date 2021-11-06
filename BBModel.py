import os, sys
import numpy as np
import pandas as pd
import sklearn.linear_model as skm

gamesPerSeason = 162
numStats = 16
polynomialDegree = 10
training_set = "data/TRAINING_1990-2020.txt"
test_set = "data/TEST_1990-2020.txt"
valid_set = "data/VALIDATION_1990-2020.txt"
learning_rate = 0.00001
epsilon = 0.001
lambda_value = 0.00001
max_samples = 1000

def getTrainingData(set):
    dataSet = open(set).readlines()

    stats = np.zeros( (len(dataSet[:max_samples]), 2*numStats*polynomialDegree) ) #first 16*5 stats are polynimalized stats of home team, last 16*5 are visiting team
    scores = np.zeros(len(dataSet[:max_samples]))
    mins, maxs = findMinMaxFromStatSheet(open("data/mlbStats2021.txt").readlines())

    for i in range(len(dataSet[:max_samples])):
        game = dataSet[:max_samples][i]
        year = int(game.split(",")[0][:4])
        homeTeam = game.split(",")[1]
        visitingTeam = game.split(",")[2]
        home = 0
        visiting = 0
        scoreDifference = int(game.split(",")[3])
        gameWorthHome = int(game.split(",")[4])
        gameWorthVisitors = int(game.split(",")[4])

        for p in range(polynomialDegree):
            stats[i][p] = (gameWorthHome/gamesPerSeason)**p
            stats[i][polynomialDegree+p] = (gameWorthVisitors/gamesPerSeason)**p

        statSheet = open("data/mlbStats" + str(year) + ".txt").readlines()
        for j in range(len(statSheet)):
            teamStats = statSheet[j]
            if teamStats.split(",")[0] == homeTeam and home != 1:
                home = 1
                n = 2*polynomialDegree
                for s in range(numStats-1): #removing one because the gameWorth stat has already been assigned
                    stdStat = float(standardize(float(teamStats.split(",")[3:][s]), mins[s], maxs[s]))
                    for p in range(polynomialDegree):
                        stats[i][n] = stdStat**p
                        n += 1
                        #first half of stats[i] is home team stats and their polynomials
            elif teamStats.split(",")[0] == visitingTeam and visiting != 1:
                visiting = 1
                n = (numStats+1)*polynomialDegree
                for s in range(numStats-1):
                    stdStat = float(standardize(float(teamStats.split(",")[3:][s]), mins[s], maxs[s]))
                    for p in range(polynomialDegree):
                        stats[i][n] = stdStat**p
                        n += 1
                        #second half of stats[i] is visiting team stats and their polynomials 
        if visiting == 0: print(str(year) + "incomplete stats: " + visitingTeam)
        if home == 0: print(str(year) + " incomplete stats: " + homeTeam)
        scores[i] = scoreDifference
        if (i%300 == 0):
            print("%.2f" % float((i/len(dataSet[:max_samples])*100)) + "%")

    return stats, scores

def predictScore(w, s): # w is (2*numStats*polyDegree)x1 matrix, s is (2 * numStats * polyDegree)x1 matrix, output is array of predicted scores (one per game)
    predictedScore = [0]*len(s)
    for i in range(len(s)):
        predictedScore[i] = s[i].dot(w)
    return predictedScore

def findMinMaxFromStatSheet(s):
    maxs = [0]*(numStats-1)
    mins = [100000]*(numStats-1)

    for i in range(len(s[1:])):
        line = s[1:][i]
        for j in range(len(line.split(",")[3:])):
            maxs[j] = max(float(line.split(",")[3:][j]), maxs[j])
            mins[j] = min(float(line.split(",")[3:][j]), mins[j])

    return mins, maxs

def standardize(stat, min, max):
    return float(stat-min)/float(max-min) #scaling each feature between 0 and 1

# function that calculates the gradient
def calculate_regularized_grad(X_poly, y, W, lambda_value):
    # let dW store dJ/dW
    dW = np.zeros((len(W),1))
    m = len(X_poly)
    y_pred = np.matmul(X_poly, W)
    
    error = y_pred - y

    for j, w_j in enumerate(W):
        ### YOUR CODE HERE - Calculate dW[j]
        # Hint: You can just copy your implementation from Q2
        # then append the L2 regularization term at the end
        
        dW[j] = (X_poly.T[j].dot(error))/m

        if j != 0:
          dW[j] += (lambda_value/m)*w_j
        
        ### ------------------------------
        


gamesPerSeason = 162
numStats = 16
polynomialDegree = 5
training_set = "data/TRAINING_1990-2020.txt"

dataSet = open(training_set).readlines()

stats = np.zeros( (len(dataSet), 2*numStats, polynomialDegree) ) #first 16 arrays are polynimalized stats of home team, last 16 are visiting team
scores = np.zeros(len(dataSet))

for i in range(len(dataSet)):
    game = dataSet[i]
    year = int(game.split(",")[0][:4])
    homeTeam = game.split(",")[1]
    visitingTeam = game.split(",")[2]
    scoreDifference = game.split(",")[3]
    gameWorth = game.split(",")[4]
    for teamStats in open("data/mlbStats" + str(year) + ".txt").readlines():
            if teamStats.split(",")[0] == homeTeam:
                for s in range(len(teamStats.split(",")[2:])):
                    for p in range(polynomialDegree): stats[i][s][p] = float(teamStats.split(",")[2:][s])**p
            if teamStats.split(",")[0] == visitingTeam:
                for s in range(len(teamStats.split(",")[2:])):
                    for p in range(polynomialDegree): stats[i][numStats+s][p] = float(teamStats.split(",")[2:][s])**p
    scores[i] = scoreDifference
    if (i%100 == 0):
        print("%.2f" % float((i/len(dataSet)*100)) + "%")

#weight matrix
weights = np.zeros( (2*numStats, polynomialDegree) )

def predictScore(w, s): # w is (2*numStats) x (polyDegree) matrix, s is (number of games) x ((2 * numStats) x (polyDegree)) matrix, output is array of predicted scores (one per game)
    prediction = np.zeros( len(s) )
    for i in range(len(s)):
        for j in range(len(w)):
            for k in range(len(w[0])):
                prediction[i] += s[i][j][k]*w[j][k]

    return prediction

print(stats[i])
print(scores[i])

###dont know how to do this part just yet

# use the above 2 functions to perform gradient descent
def train_polynomial_regression(X_poly, y, W, learning_rate, epsilon):
    epoch_count = 0
    while True:
        #calculate current gradient
        dW = calculate_grad(X_poly, y, W)
        W_prev = W.copy()

        ### YOUR CODE HERE - update each W[j] using the given learning_rate

        for i in range(len(W)):
            for j in range(len(W[0])):
                W[i][j] -= learning_rate*dW[i][j]
        ### ------------------------------

        diff = calculate_dist(W_prev, W)
        if (diff < epsilon):
            break

        epoch_count +=1
        # print train error every 50 iterations
        if epoch_count % 200 == 0:
            y_train_pred = np.matmul(X_train_poly, W)
            print('Training set Mean Squared Error: {}'.format(np.power((y_train_pred - y_train), 2).mean()))
  
    print('Training completed after {} iteration(s)'.format(epoch_count+1))

    return W

# function that calculates the gradient
def calculate_grad(X_poly, y, W):
    # let dW represent dJ/dW
    dW = np.zeros( (2*numStats*polynomialDegree, 1) )
    m = len(X_poly)
    y_pred = X_poly.dot(W)

    error = y_pred - y

    for j, w_j in enumerate(W):
      dW[j] = (X_poly.T[j].dot(error))/m 
    return dW

# function that caculates the change in W
def calculate_dist(W_prev, W_cur):
    print(W_prev)
    print(W_cur)
    return np.sqrt(np.sum((W_cur - W_prev)**2), dtype=np.float64)

# use the above 2 functions to perform regularized gradient descent
def train_regularized_polynomial_regression(X_poly, y, W, learning_rate, epsilon, lambda_value, verbose=True):
  epoch_count = 0
  while True:
      #calculate current gradient
      dW = calculate_regularized_grad(X_poly, y, W, lambda_value)

      W_prev = W.copy()

      ### YOUR CODE HERE - update W[j] using the given learning_rate
      # Hint: This should be the same as your implementation from Q2

      for j in range(W.shape[0]): 
        W[j] -= learning_rate*dW[j]

      ### ------------------------------

      diff = calculate_dist(W_prev, W)
      if (diff < epsilon):
          break

      epoch_count +=1
      # print train error every 50 iterations
      if verbose:
        if epoch_count % 100 == 0:
          y_train_pred = np.matmul(X_poly, W)
          print('Training set Mean Squared Error: {}'.format(np.power((y_train_pred - y), 2).mean()))

  print('Training complete.')
  return W
 
###################

weights = np.zeros(2*numStats*polynomialDegree)
training_x, training_y = getTrainingData(training_set)
test_x, test_y = getTrainingData(test_set)

reg = skm.LinearRegression().fit(training_x, training_y)
print(reg.score(test_x, test_y))
reg = skm.Ridge().fit(training_x, training_y)
print(reg.score(test_x, test_y))






valid_x, valid_y = getTrainingData(valid_set)
W = train_regularized_polynomial_regression(training_x, training_y, weights, learning_rate, epsilon, lambda_value)
#calculate squared error on training set
pred_scores = predictScore(training_x, W)
mse = np.power((pred_scores - training_y), 2).mean()
print('\nTraining Mean Squared Error: {}'.format(mse))


#calculate squared error on test set
pred_scores = predictScore(test_x, W)
mse = np.power((pred_scores - test_y), 2).mean()
print('\nTest Mean Squared Error: {}'.format(mse))
#calculate squared error on test set
pred_scores = predictScore(valid_x, W)
mse = np.power((pred_scores - valid_y), 2).mean()
print('\Validation Mean Squared Error: {}'.format(mse))
