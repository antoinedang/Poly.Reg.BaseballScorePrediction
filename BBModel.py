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
import pickle

#display starting message with the current time
time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
print("Starting Baseball Stats Model > " + time)

#parameters
max_samples = -1
use_pca = True
use_scaling = True
shuffle_data = True
startYear = 1990
endYear = 2020
num_components = 500

#hyperparameters for hyperparameter search
hyperparameters = [ [0], [0], [0] ] #empty array to store arrays
hyperparameters[0] = [ 50, 75, 100, 200, 500, 750, 1000, 1500, 2500 ] #num_components for pca
hyperparameters[1] = [x * 0.1 for x in range(1, 20)] #for alpha
hyperparameters[2] = [ 500, 1000, 5000, 10000, 50000 ] #for max_iter

#update GenerateGameData's parameters
GenerateGameData.startYear = startYear
GenerateGameData.endYear = endYear
GenerateGameData.max_samples = max_samples

#function that saves the input score to a file along with information about the model that acheived the results
def saveScore(accuracy, mse, name, parameters, startTime, pred, true, n_comp):
    scores = open("modelResults.txt", 'a')
    #get time
    time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

    output = ""
    output += "-------------- " + startTime + " --------------\n" #show starting time of model training
    output += "\t" + name + "\n" #name of the model
    output += "\t\t" + "Mean Squared Error: " + str(mse) + "\n" #MSE of this model's predictions
    output += "\t\t" + "Accuracy: " + str(accuracy) + "%\n" #accuracy this model achieved
    output += "\t\t" + "Parameters: " + str(parameters) #all the models parameters
    output += "\n\t\t" + "Predicted: " + str(pred[:10]) #first 10 predicted outcomes
    output += "\n\t\t" + "Mean of predicted: " + str(np.mean(pred)) #mean of the model predictions
    output += "\n\t\t" + "True: " + str(true[:10]) #first 10 target outcomes
    output += "\n\t\t" + "Mean of true: " + str(np.mean(true)) #mean of the target outcomes
    output += "\n\t\t" + "Hyperparameters: num_components: " + str(n_comp) + ", max_samples: " + str(max_samples) + ", startYear: " + str(startYear) + ", endYear: " + str(endYear)
    output += "\n-------------- " + time + " --------------\n" #show end time of model training

    scores.write(output)
    scores.close()

def updateHighScore(accuracy, mse, name, parameters, pred, true, n_comp):
    try: #if high score file exists
        data = open("bestMSE.txt", 'r').readlines()
        lowestMSE = float(data[2][23:]) #get current best MSE
    except: #if highScore file doesnt exist set a high MSE so it will be beat
        data = [""]
        lowestMSE = 1000000000.0

    try: #if high score file exists
        data = open("bestAccuracy.txt", 'r').readlines()
        highestAccuracy = float(data[3][12:18]) #get current best accuracy
    except:
        data = [""]
        highestAccuracy = 0.0


    output = ""
     #if this model has a new best accuracy overwrite the bestAccuracy.txt file with this model's details
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
     #if this model has a new best MSE overwrite the bestMSE.txt file with this model's details
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

#function that prints the score of a model's prediction on a stat set, saves the result to the filesystem and updates the high score
def score(pred, true, name, parameters, startTime, n_comp):

    num_samples = float(len(pred))
    squaredError = 0.0
    correct_outcomes = 0
    
    #go through predictions
    for i in range(len(pred)):
        #if they are correct count it as so
        if pred[i]*true[i] == 0 and pred[i]+true[i] == 0: correct_outcomes += 1
        elif pred[i]*true[i] > 0: correct_outcomes += 1
        #add to overall mean squared error
        squaredError += ((pred[i]-true[i])**2.0)/num_samples

    #check if this result beats the high score
    updateHighScore(float(correct_outcomes/num_samples*100.0), squaredError, name, parameters, pred, true, n_comp)

    #save this result to the filesystem so we can see all our training
    saveScore(float(correct_outcomes/num_samples*100.0), squaredError, name, parameters, startTime, pred, true, n_comp)

    #print basic info about the model to console for status update
    print("===== " + name + " =====")
    print("Mean squared error: " + str(squaredError))
    print("Model predicted outcome correctly " + str(correct_outcomes) + " times out of " + str(num_samples) + " games. (%.2f" % float(correct_outcomes/num_samples*100.0) + "% Accuracy)")

################### TESTING CODE

def startHyperparameterSearch():
    #load training data from file
    training_x, training_y, status = GenerateGameData.loadFromFile("data/TrainingData" + str(startYear) + "-" + str(endYear) + ".txt", shuffle_data)
    #if it hasnt been generated yet, generate and load it
    if (status == 1):
        GenerateGameData.setup()
        training_x, training_y, status = GenerateGameData.loadFromFile("data/TrainingData" + str(startYear) + "-" + str(endYear) + ".txt", shuffle_data)
    #load test data from file
    test_x, test_y, status = GenerateGameData.loadFromFile("data/TestData" + str(startYear) + "-" + str(endYear) + ".txt", shuffle_data)

    if (use_scaling): #standardize the x and y arrays for training and test
        scalerx = preprocessing.StandardScaler().fit(training_x)
        training_x = scalerx.transform(training_x)
        test_x = scalerx.transform(test_x)

        scalery = preprocessing.StandardScaler().fit(training_y.reshape(-1, 1))
        training_y = scalery.transform(training_y.reshape(-1, 1))
        test_y = scalery.transform(test_y.reshape(-1, 1))

    if (use_pca): #pca the x and y arrays for training and test data
        pca = PCA(n_components = num_components)
        old_training_x = training_x
        old_test_x = test_x
        training_x = pca.fit_transform(training_x)
        test_x = pca.transform(test_x)

    #append a starting message to the log (indicating we started a hyperparameter search)
    log = open("modelResults.txt", 'a')
    output = ""
    output += "\n"
    output += "\n"
    output += "\n|"
    output += "\n|"
    output += "\n|"
    output += "----- " + datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + " -----\n"
    output += "|"
    output += "\n|"
    log.write(output)
    log.close()

    for component in hyperparameters[0]:#num_components for pca

        #pca the training and test data with the new num components
        pca.set_params(n_components = component)
        training_x = pca.fit_transform(old_training_x)
        test_x = pca.transform(old_test_x)

        #get the time this model started training
        startTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        #create and fit model to data
        model = skm.LinearRegression(fit_intercept=True, normalize='deprecated', copy_X=True, n_jobs=-1, positive=False)
        model.fit(training_x, training_y)
        #predict scores with test set
        predictedScores = model.predict(test_x)
        #score and log the model
        score(predictedScores, test_y, "Linear", model.get_params(), startTime, component)

        for maxiter in hyperparameters[2]: #for maxiter
            #do the same for each different maxiter value
            startTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            model = PLSRegression(n_components=component, scale=False, max_iter=maxiter, tol=1e-06, copy=True)
            model.fit(training_x, training_y)
            predictedScores = model.predict(test_x)
            score(predictedScores, test_y, "Partial Least Squares", model.get_params(), startTime, component)

            for alpha in hyperparameters[1]: #for alpha
                #do the same for each different alpha value
                startTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                model = skm.Ridge(alpha=alpha, max_iter=maxiter, tol=0.00001, solver='auto')
                model.fit(training_x, training_y)
                predictedScores = model.predict(test_x)
                score(predictedScores, test_y, "Ridge", model.get_params(), startTime, component)

def saveModel(alpha, maxiter, numc):
    
    #load datasets from the file
    training_x, training_y, result1 = GenerateGameData.loadFromFile("data/TrainingData" + str(startYear) + "-" + str(endYear) + ".txt")
    test_x, test_y, result2 = GenerateGameData.loadFromFile("data/TestData" + str(startYear) + "-" + str(endYear) + ".txt")
    valid_x, valid_y, result3 = GenerateGameData.loadFromFile("data/ValidationData" + str(startYear) + "-" + str(endYear) + ".txt")
    
    if result1+result2+result3 > 0: #if the data hasn't been generated yet
        GenerateGameData.setup() #create the datasets and writes them to file for later
        #load them from filesystem
        training_x, training_y, ignore = GenerateGameData.loadFromFile("data/TrainingData" + str(startYear) + "-" + str(endYear) + ".txt")
        test_x, test_y, ignore = GenerateGameData.loadFromFile("data/TestData" + str(startYear) + "-" + str(endYear) + ".txt")
        valid_x, valid_y, ignore = GenerateGameData.loadFromFile("data/ValidationData" + str(startYear) + "-" + str(endYear) + ".txt")
        
    #combine all data into one huge dataset
    training_x = np.concatenate((training_x, test_x, valid_x), axis = 0)
    training_y = np.concatenate((training_y, test_y, valid_y), axis = 0)

    #standardize the x and y sets
    scalerx = preprocessing.StandardScaler().fit(training_x)
    training_x = scalerx.transform(training_x)
    scalery = preprocessing.StandardScaler().fit(training_y.reshape(-1, 1))
    training_y = scalery.transform(training_y.reshape(-1, 1))

    #lower the number of components with PCA
    pca = PCA(n_components = numc)
    training_x = pca.fit_transform(training_x)

    #save the scaler and pca instance to a file (so we can use the exact same pca and scaling in our webapp)
    pickle.dump(scalerx, open('.scaler.sav', 'wb'))
    pickle.dump(pca, open('.pca.sav', 'wb'))
    
    #create our model, fit it to the data, and save it to a file (so we can predict outcomes in our webapp)
    model = skm.Ridge(alpha=alpha, max_iter=maxiter, tol=0.00001, solver='auto')
    model.fit(training_x, training_y)
    pickle.dump(model, open('.finalized_model.sav', 'wb'))

def createFinalModel(): #creates and saves the final model to the filesystem
    #saveModel(alpha, max_iter, pca num components)
    saveModel(0.11280000000000001, 5000, 500)

createFinalModel()

if __name__ == '__main__':
    startHyperparameterSearch() #allows us to run this script from terminal (just add an argument) but also avoids it running default functions when it is imported by the BBModel script