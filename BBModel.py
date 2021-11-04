import os, sys
import numpy as np
import pandas as pd


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
exit()
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
    return np.sqrt(np.sum((W_cur - W_prev)**2))
