import os, sys
import numpy as np
import pandas as pd
import sklearn.linear_model as skm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import GenerateGameData

# hyperparameters
learning_rate = 0.00001
epsilon = 0.001
lambda_value = 0.00001
numStats = GenerateGameData.numStats

def predictScore(w, s): # w is (2*numStats*polyDegree)x1 matrix, s is (2 * numStats * polyDegree)x1 matrix, output is array of predicted scores (one per game)
    predictedScore = [0]*len(s)
    for i in range(len(s)):
        predictedScore[i] = s[i].dot(w)
    return predictedScore

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
    print(W_cur)
    return np.sqrt(np.sum((W_cur - W_prev)**2), dtype=np.float64)

# use the above 2 functions to perform regularized gradient descent
def train_regularized_polynomial_regression(X_poly, y, W, learning_rate, epsilon, lambda_value, verbose=True):
  epoch_count = 0
  while True:
      #calculate current gradient
      dW = calculate_regularized_grad(X_poly, y, W, lambda_value)

      W_prev = W.copy()

      for j in range(W.shape[0]): 
        W[j] -= learning_rate*dW[j]

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

#function that prints the score of a model's prediction on a stat set
def score(pred, true):
    num_samples = len(pred)
    totalSquaredError = 0.0
    correct_outcomes = 0.0
    for i in range(len(pred)):
        if pred[i]*true[i] > 0: correct_outcomes += 1
        totalSquaredError += (pred[i]-true[i])**2

    print("===== SCORE =====")
    print("Mean squared error: " + str(totalSquaredError/num_samples))
    print("Model predicted outcome correctly " + str(correct_outcomes) + " times out of " + str(num_samples) + " games. (%.2f" % float(correct_outcomes/num_samples*100.0) + "% Accuracy)")
    print("=================")

################### TESTING CODE

training_x, training_y = GenerateGameData.loadFromFile("data/TrainingData1990-2020.txt")
test_x, test_y = GenerateGameData.loadFromFile("data/TestData1990-2020.txt")

model = make_pipeline(StandardScaler(), skm.LinearRegression(n_jobs=-1))
model.fit(training_x, training_y)

predictedScores = model.predict(test_x)
score(predictedScores, test_y)


exit()


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