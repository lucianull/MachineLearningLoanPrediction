import numpy as np
import warnings

class LogisticRegression:


    def __init__(self, learningRate=0.01, nrIterations=1000) -> None:
        self.learningRate = learningRate
        self.nrIterations = nrIterations

        self.weights = None
        self.bias = None

        self.costImprovement = []
        self.costTrack = 100


    def setLearningRate(self, lr):
        self.learningRate = lr


    def setNrIterations(self, nr):
        self.nrIterations = nr


    def setCostTrack(self, nr):
        self.costTrack = nr


    def getCostImprovement(self):
        return self.costImprovement


    def SigmoidFunction(self, x):
        return 1 / (1 + np.exp(-x))


    def FitModel(self, dataset, datasetValue):
        nrSamples, nrFeatures = dataset.shape
        self.weights = np.zeros(nrFeatures)
        self.bias = 0
        for i in range(self.nrIterations):
            liniarPredictions = np.dot(dataset, self.weights) + self.bias
            logisticPredictions = self.SigmoidFunction(liniarPredictions)


            cost = -1/nrSamples * np.sum(datasetValue * np.log(logisticPredictions) + (1 - datasetValue) * np.log(1 - logisticPredictions))

            weightsDerivatives = (1 / nrSamples) * np.dot(dataset.T, (logisticPredictions - datasetValue))
            biasDerivative = (1 / nrSamples) * np.sum(logisticPredictions - datasetValue)

            self.weights -= self.learningRate * weightsDerivatives
            self.bias -= self.learningRate * biasDerivative
            if i % self.costTrack == 0:
                self.costImprovement.append(cost)

    
    def PredictData(self, dataset):
        liniarPredictions = np.dot(dataset, self.weights) + self.bias
        logisticPredictions = self.SigmoidFunction(liniarPredictions)
        return logisticPredictions

