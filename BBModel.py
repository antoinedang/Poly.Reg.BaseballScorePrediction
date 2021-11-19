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


def saveScore(accuracy, mse, name, parameters, startTime, pred, true, n_comp):
    scores = open("modelResults.txt", 'a')
    time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    output = ""
    output += "-------------- " + startTime + " --------------\n"
    output += "\t" + name + "\n"
    output += "\t\t" + "Mean Squared Error: " + str(mse) + "\n"
    output += "\t\t" + "Accuracy: " + str(accuracy) + "%\n"
    output += "\t\t" + "Parameters: " + str(parameters)
    output += "\n\t\t" + "Predicted: " + str(pred[:10])
    output += "\n\t\t" + "Mean of predicted: " + str(np.mean(pred))
    output += "\n\t\t" + "True: " + str(true[:10])
    output += "\n\t\t" + "Mean of true: " + str(np.mean(true))
    output += "\n\t\t" + "Hyperparameters: num_components: " + str(n_comp) + ", max_samples: " + str(max_samples) + ", startYear: " + str(startYear) + ", endYear: " + str(endYear)
    output += "\n-------------- " + time + " --------------\n"

    scores.write(output)
    scores.close()

def updateHighScore(accuracy, mse, name, parameters, pred, true, n_comp):
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
        output += "\n\t\t" + "Predicted: " + str(pred[:10])
        output += "\n\t\t" + "Mean of predicted: " + str(np.mean(pred))
        output += "\n\t\t" + "True: " + str(true[:10])
        output += "\n\t\t" + "Mean of true: " + str(np.mean(true))
        output += "\n\t\t" + "Hyperparameters: num_components: " + str(n_comp) + ", max_samples: " + str(max_samples) + ", startYear: " + str(startYear) + ", endYear: " + str(endYear)

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
        output += "\n\t\t" + "Predicted: " + str(pred[:10])
        output += "\n\t\t" + "Mean of predicted: " + str(np.mean(pred))
        output += "\n\t\t" + "True: " + str(true[:10])
        output += "\n\t\t" + "Mean of true: " + str(np.mean(true))
        output += "\n\t\t" + "Hyperparameters: num_components: " + str(num_components) + ", max_samples: " + str(max_samples) + ", startYear: " + str(startYear) + ", endYear: " + str(endYear)

        bestMSE = open("bestMSE.txt", 'w')
        bestMSE.write(output)
        bestMSE.close()

#function that prints the score of a model's prediction on a stat set
def score(pred, true, name, parameters, reverse, startTime, n_comp):
    if (reverse): print (pred[:100])
    num_samples = float(len(pred))
    squaredError = 0.0
    correct_outcomes = 0
    for i in range(len(pred)):
        if reverse and pred[i]*true[i] < 0: correct_outcomes += 1
        if pred[i]*true[i] > 0 and not reverse: correct_outcomes += 1
        squaredError += ((pred[i]-true[i])**2.0)/num_samples

    updateHighScore(float(correct_outcomes/num_samples*100.0), squaredError, name, parameters, pred, true, n_comp)
    saveScore(float(correct_outcomes/num_samples*100.0), squaredError, name, parameters, startTime, pred, true, n_comp)

    print("===== " + name + " =====")
    print("Mean squared error: " + str(squaredError))
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
    hyperparameters[1] = [x * 0.1 for x in range(1, 20)] #for alpha
    hyperparameters[2] = [ 500, 1000, 5000, 10000, 50000 ] #for max_iter

    #runDefaultModels(training_x, training_y, test_x, test_y)

    for component in hyperparameters[0]:#num_components for pca
        pca.set_params(n_components = component)
        training_x = pca.fit_transform(old_training_x)
        test_x = pca.transform(old_test_x)

        
        log = open("modelResults.txt", 'a')
        output = ""
        output += "-------------- | --------------\n"
        output += "------ NEW NUM COMPONENTS -----\n"
        output += str(component) + "\n"
        output += str(training_x.shape) + "\n"
        output += "-------------- | --------------\n"
        log.write(output)
        log.close()
        print(training_x.shape)
        print(test_x.shape)

        startTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        model = skm.LinearRegression(fit_intercept=True, normalize='deprecated', copy_X=True, n_jobs=-1, positive=False)
        model.fit(training_x, training_y)
        predictedScores = model.predict(test_x)
        score(predictedScores, test_y, "Linear", model.get_params(), False, startTime, component)

        for maxiter in hyperparameters[2]: #for alpha

            startTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            model = PLSRegression(n_components=component, scale=False, max_iter=maxiter, tol=1e-06, copy=True)
            model.fit(training_x, training_y)
            predictedScores = model.predict(test_x)
            score(predictedScores, test_y, "Partial Least Squares", model.get_params(), False, startTime, component)

            for alpha in hyperparameters[1]: #for alpha

                startTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                model = skm.Ridge(alpha=alpha, max_iter=maxiter, tol=0.00001, solver='auto')
                model.fit(training_x, training_y)
                predictedScores = model.predict(test_x)
                score(predictedScores, test_y, "Ridge", model.get_params(), False, startTime, component)

def runDefaultModels(trainx, trainy, testx, testy):
    startTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    model = skm.LinearRegression(n_jobs=-1)
    model.fit(trainx, trainy)
    predictedScores = model.predict(testx)
    score(predictedScores, testy, "Linear", model.get_params(), False, startTime, trainx.shape[1])

    startTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    model = skm.Ridge(alpha=1.0, max_iter=None, tol=0.0001, solver='auto')
    model.fit(trainx, trainy)
    predictedScores = model.predict(testx)
    score(predictedScores, testy, "Ridge", model.get_params(), False, startTime, trainx.shape[1])

    startTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    model = skm.Lasso(alpha=1.0, fit_intercept=True, normalize='deprecated', precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
    model.fit(np.asfortranarray(trainx), trainy)
    predictedScores = model.predict(testx)
    score(predictedScores, testy, "Lasso", model.get_params(), False, startTime, trainx.shape[1])

    startTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    model = PLSRegression(n_components=2, scale=True, max_iter=500, tol=1e-06, copy=True)
    model.fit(trainx, trainy)
    predictedScores = model.predict(testx)
    score(predictedScores, testy, "Partial Least Squares", model.get_params(), False, startTime, trainx.shape[1])

    startTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    model = SVR(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, max_iter=- 1, verbose=True)
    model.fit(trainx, trainy)
    predictedScores = model.predict(testx)
    score(predictedScores, testy, "Support Vector Machines", model.get_params(), False, startTime, trainx.shape[1])

    startTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    model = RandomForestRegressor(n_estimators=100, criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, warm_start=False, ccp_alpha=0.0, max_samples=None, verbose=True)
    model.fit(trainx, trainy)
    predictedScores = model.predict(testx)
    score(predictedScores, testy, "Random Forest", model.get_params(), False, startTime, trainx.shape[1])

start()