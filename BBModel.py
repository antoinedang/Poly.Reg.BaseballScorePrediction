import os, sys
import numpy as np
import pandas as pd


gamesPerSeason = 162
numStats = 16
polynomialDegree = 5
training_set = "data/TRAINING_1990-2020.txt"
test_set = "data/TEST_1990-2020.txt"
valid_set = "data/VALIDATION_1990-2020.txt"
learning_rate = 0.3
epsilon = 0.001
lambda_value = 0.3

def getTrainingData(set):
    dataSet = open(set).readlines()

    stats = np.zeros( (len(dataSet), 2*numStats*polynomialDegree) ) #first 16 arrays are polynimalized stats of home team, last 16 are visiting team
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
                    n = 0
                    for s in range(numStats):
                        for p in range(polynomialDegree):
                            stats[i][n] = float(teamStats.split(",")[2:][s])**p
                            n += 1
                            #first half of stats[i] is home team stats and their polynomials
                if teamStats.split(",")[0] == visitingTeam:
                    n = numStats*polynomialDegree
                    for s in range(numStats):
                        for p in range(polynomialDegree):
                            stats[i][n] = float(teamStats.split(",")[2:][s])**p
                            n += 1
                            #second half of stats[i] is visiting team stats and their polynomials
        scores[i] = scoreDifference
        if (i%300 == 0):
            print("%.2f" % float((i/len(dataSet)*100)) + "%")

    return stats, scores

def predictScore(w, s): # w is (2*numStats*polyDegree)x1 matrix, s is (2 * numStats * polyDegree)x1 matrix, output is array of predicted scores (one per game)
    predictedScore = [0]*len(s)
    for i in range(len(s)):
        predictedScore[i] = s[i].dot(w)
    return predictedScore

weights = np.zeros(2*numStats*polynomialDegree)
training_x, training_y = getTrainingData(training_set)

##STANDARDIZE STATS

print(training_x[0][:100])
exit()

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
        
    return dW

# function that caculates the change in W
def calculate_dist(W_prev, W_cur):
    return np.sqrt(np.sum((W_cur - W_prev)**2))

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

W = train_regularized_polynomial_regression(training_x, training_y, weights, learning_rate, epsilon, lambda_value)
#calculate squared error on training set
pred_scores = predictScore(training_x, W)
mse = np.power((pred_scores - training_y), 2).mean()
print('\nTraining Mean Squared Error: {}'.format(mse))


test_x, test_y = getTrainingData(test_set)
valid_x, valid_y = getTrainingData(valid_set)

#calculate squared error on test set
pred_scores = predictScore(test_x, W)
mse = np.power((pred_scores - test_y), 2).mean()
print('\nTest Mean Squared Error: {}'.format(mse))
#calculate squared error on test set
pred_scores = predictScore(valid_x, W)
mse = np.power((pred_scores - valid_y), 2).mean()
print('\Validation Mean Squared Error: {}'.format(mse))