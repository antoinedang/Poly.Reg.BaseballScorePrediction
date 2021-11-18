import os, sys
from types import TracebackType
import numpy as np
import pandas as pd
import sklearn.linear_model as skm
import sklearn.preprocessing as preprocessing
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from datetime import datetime
import GenerateGameData
time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")


print("Starting Baseball Stats Model > " + time)


max_samples = -1
use_pca = True
use_scaling = True
shuffle_data = True
startYear = 1990
endYear = 2020
num_components = 500
GenerateGameData.startYear = startYear
GenerateGameData.endYear = endYear
GenerateGameData.max_samples = max_samples


def saveScore(accuracy, mse, name, parameters, startTime):
    scores = open("modelResults.txt", 'a')
    time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    output = ""
    output += "-------------- " + startTime + " --------------\n"
    output += "\t" + name + "\n"
    output += "\t\t" + "Mean Squared Error: " + str(mse) + "\n"
    output += "\t\t" + "Accuracy: " + str(accuracy) + "%\n"
    output += "\t\t" + "Parameters: " + str(parameters)
    output += "\n\t\t" + "Hyperparameters: num_components: " + str(num_components) + ", max_samples: " + str(max_samples) + ", startYear: " + str(startYear) + ", endYear: " + str(endYear)
    output += "\n-------------- " + time + " --------------\n"

    scores.write(output)
    scores.close()

def updateHighScore(accuracy, mse, name, parameters):
    try:
        data = open("bestMSE.txt", 'r').readlines()
        lowestMSE = float(data[2][22:])
    except:
        data = [""]
        lowestMSE = 1000000000.0

    try:
        data = open("bestAccuracy.txt", 'r').readlines()
        highestAccuracy = float(data[3][12:18])
    except:
        data = [""]
        highestAccuracy = 0.0


    output = ""
    if (accuracy > highestAccuracy):
        output += "Highest Accuracy:\n"
        output += "\t" + name + "\n"
        output += "\t\t" + "Mean Squared Error: " + str(mse) + "\n"
        output += "\t\t" + "Accuracy: " + str(accuracy) + "%\n"
        output += "\t\t" + "Parameters: " + str(parameters)
        output += "\n\t\t" + "Hyperparameters: num_components: " + str(num_components) + ", max_samples: " + str(max_samples) + ", startYear: " + str(startYear) + ", endYear: " + str(endYear)

        bestAccuracy = open("bestAccuracy.txt", 'w')
        bestAccuracy.write(output)
        bestAccuracy.close()
    
    output = ""
    if (mse < lowestMSE):
        output += "Lowest Mean Squared Error:\n"
        output += "\t" + name + "\n"
        output += "\t\t" + "Mean Squared Error: " + str(mse) + "\n"
        output += "\t\t" + "Accuracy: " + str(accuracy) + "%\n"
        output += "\t\t" + "Parameters: " + str(parameters)
        output += "\n\t\t" + "Hyperparameters: num_components: " + str(num_components) + ", max_samples: " + str(max_samples) + ", startYear: " + str(startYear) + ", endYear: " + str(endYear)

        bestMSE = open("bestMSE.txt", 'w')
        bestMSE.write(output)
        bestMSE.close()

#function that prints the score of a model's prediction on a stat set
def score(pred, true, name, parameters, reverse, startTime):
    if (reverse): print (pred[:100])
    num_samples = len(pred)
    totalSquaredError = 0.0
    correct_outcomes = 0
    for i in range(len(pred)):
        if reverse and pred[i]*true[i] < 0: correct_outcomes += 1
        if pred[i]*true[i] > 0 and not reverse: correct_outcomes += 1
        totalSquaredError += (pred[i]-true[i])**2

    updateHighScore(float(correct_outcomes/num_samples*100.0), float(totalSquaredError/num_samples), name, parameters)
    saveScore(float(correct_outcomes/num_samples*100.0), float(totalSquaredError/num_samples), name, parameters, startTime)

    print("===== " + name + " =====")
    print("Mean squared error: " + str(totalSquaredError/num_samples))
    print("Model predicted outcome correctly " + str(correct_outcomes) + " times out of " + str(num_samples) + " games. (%.2f" % float(correct_outcomes/num_samples*100.0) + "% Accuracy)")

################### TESTING CODE

def start():
    training_x, training_y, status = GenerateGameData.loadFromFile("data/TrainingData" + str(startYear) + "-" + str(endYear) + ".txt", shuffle_data)
    if (status == 1):
        GenerateGameData.setup()
        training_x, training_y, status = GenerateGameData.loadFromFile("data/TrainingData" + str(startYear) + "-" + str(endYear) + ".txt", shuffle_data)
    test_x, test_y, status = GenerateGameData.loadFromFile("data/TestData" + str(startYear) + "-" + str(endYear) + ".txt", shuffle_data)

    if (use_scaling):
        scalerx = preprocessing.StandardScaler().fit(training_x)
        training_x = scalerx.transform(training_x)
        test_x = scalerx.transform(test_x)

        scalery = preprocessing.StandardScaler().fit(training_y.reshape(-1, 1))
        training_y = scalery.transform(training_y.reshape(-1, 1))
        test_y = scalery.transform(test_y.reshape(-1, 1))

    if (use_pca):
        pca = PCA(n_components = num_components)
        old_training_x = training_x
        old_test_x = test_x
        training_x = pca.fit_transform(training_x)
        test_x = pca.transform(test_x)

    
    log = open("modelResults.txt", 'a')
    output = ""
    output += "-------------- | --------------\n"
    output += "----- " + datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + " -----\n"
    output += "-------------- | --------------\n"
    log.write(output)
    log.close()

    #MODELS
    #custom parameters
    hyperparameters = [ [0], [0], [0], [0] ]
    hyperparameters[0] = [ 50, 75, 100, 200, 500, 750, 1000, 1500, 2500 ] #num_components for pca

    hyperparameters[1] = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 ] #for alpha
    hyperparameters[2] = [ True, False ] #for warm start
    hyperparameters[3] = [ 10, 20, 30, 40, 50, 100, 200 ] #for leaf size and n_neighbours and batch size

    #runDefaultModels(training_x, training_y, test_x, test_y)

    for i in hyperparameters[0]:#num_components for pca
        pca = PCA(n_components = i)
        training_x = pca.fit_transform(old_training_x)
        test_x = pca.transform(old_test_x)

        
        startTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        model = skm.LinearRegression(fit_intercept=True, normalize='deprecated', copy_X=True, n_jobs=-1, positive=False)
        model.fit(training_x, training_y)
        predictedScores = model.predict(test_x)
        score(predictedScores, test_y, "Linear", model.get_params(), False, startTime)

        startTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        model = PLSRegression(n_components=i, scale=False, max_iter=10000, tol=1e-06, copy=True)
        model.fit(training_x, training_y)
        predictedScores = model.predict(test_x)
        score(predictedScores, test_y, "Partial Least Squares", model.get_params(), False, startTime)

        for alpha in hyperparameters[1]: #for alpha

            startTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            model = skm.Ridge(alpha=alpha, max_iter=None, tol=0.00001, solver='auto')
            model.fit(training_x, training_y)
            predictedScores = model.predict(test_x)
            score(predictedScores, test_y, "Ridge", model.get_params(), False, startTime)

            for warmstart in hyperparameters[2]:#for warm start
                startTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                model = skm.Lasso(alpha=alpha, fit_intercept=False, normalize='deprecated', precompute=False, copy_X=True, max_iter=10000, tol=0.0001, warm_start=warmstart, positive=False, random_state=None, selection='random')
                model.fit(np.asfortranarray(training_x), training_y)
                predictedScores = model.predict(test_x)
                score(predictedScores, test_y, "Lasso", model.get_params(), True, startTime)

        for j in hyperparameters[3]:#for n_neighbours and batch size

            for leaf in hyperparameters[3]:#for leaf size

                startTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                model = KNeighborsRegressor(n_neighbors=j, weights='distance', algorithm='auto', leaf_size=leaf, p=2, metric='minkowski', metric_params=None, n_jobs=-1)
                model.fit(training_x, training_y)
                predictedScores = model.predict(test_x)
                score(predictedScores, test_y, "K-Nearest Neighbours", model.get_params(), False, startTime)


def runDefaultModels(trainx, trainy, testx, testy):
    startTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    model = skm.LinearRegression(n_jobs=-1)
    model.fit(trainx, trainy)
    predictedScores = model.predict(testx)
    score(predictedScores, testy, "Linear", model.get_params(), False, startTime)

    startTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    model = skm.Ridge(alpha=1.0, max_iter=None, tol=0.0001, solver='auto')
    model.fit(trainx, trainy)
    predictedScores = model.predict(testx)
    score(predictedScores, testy, "Ridge", model.get_params(), False, startTime)

    startTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    model = skm.Lasso(alpha=1.0, fit_intercept=True, normalize='deprecated', precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
    model.fit(np.asfortranarray(trainx), trainy)
    predictedScores = model.predict(testx)
    score(predictedScores, testy, "Lasso", model.get_params(), False, startTime)

    startTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    model = PLSRegression(n_components=2, scale=True, max_iter=500, tol=1e-06, copy=True)
    model.fit(trainx, trainy)
    predictedScores = model.predict(testx)
    score(predictedScores, testy, "Partial Least Squares", model.get_params(), False, startTime)

    startTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    model = SVR(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, max_iter=- 1, verbose=True)
    model.fit(trainx, trainy)
    predictedScores = model.predict(testx)
    score(predictedScores, testy, "Support Vector Machines", model.get_params(), False, startTime)

    startTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    model = RandomForestRegressor(n_estimators=100, criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, warm_start=False, ccp_alpha=0.0, max_samples=None, verbose=True)
    model.fit(trainx, trainy)
    predictedScores = model.predict(testx)
    score(predictedScores, testy, "Random Forest", model.get_params(), False, startTime)

start()