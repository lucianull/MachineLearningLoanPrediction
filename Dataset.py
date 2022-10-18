import numpy as np
import pandas as pd
import warnings
import os


warnings.filterwarnings('ignore')

STRUCT_FILENAME = 'dataset_structure'
TRAIN_FILENAMES = os.listdir('Data/TrainData')
TEST_FILENAMES = os.listdir('Data/TestData')
if TEST_FILENAMES:
    TEST_FILENAMES = TEST_FILENAMES[0]
else:
    TEST_FILENAMES = ""
PREDICT_FILENAME = os.listdir('Data/PredictData')
if PREDICT_FILENAME:
    PREDICT_FILENAME = PREDICT_FILENAME[0]
else:
    PREDICT_FILENAME = ""

def convertNumber(string):
    if string == '?':
        return np.nan
    return float(string)


class Dataset:


    def __init__(self) -> None:
        self.fileNames = TRAIN_FILENAMES


    def setEncodingLabel(self, dict):

        self.encodingFeatures[self.nrFeatures - 1] = dict


    def ReadFile(self, fileName):

        dataset = pd.DataFrame(columns=self.structure)
        with open(fileName, 'r') as inFile:
            for line in inFile.readlines():
                line = line.rstrip()
                line = line.split(',')
                for i in range(1, len(line)):
                    if self.nrFeature[i] != -1:                                        #data preprocessing (turning limited options for a feature into numbers)
                        if line[i] in self.encodingFeatures[i]:
                            line[i] = self.encodingFeatures[i][line[i]]
                        else:
                            self.encodingValues[i] += 1
                            self.encodingFeatures[i][line[i]] = self.encodingValues[i]
                            line[i] = self.encodingFeatures[i][line[i]]
                    else:
                        line[i] = convertNumber(line[i])
                dataset.loc[dataset.shape[0]] = line
        for i in range(1, self.nrFeatures):
            if self.structureDataTypes[i] == 'numeric':
                dataset[self.structure[i]].fillna(dataset[self.structure[i]].mean(), inplace=True)
        return dataset


    def ReadDataset(self):
        self.dataset = None
        self.structure = None
        self.nrFeature = None                                                           
        self.structureDataTypes = None
        self.encodingFeatures = {}                                                      #dictionary needed for encoding a feature with a limited number of options, for example gender: male, female or unknown
        self.encodingValues = None
        self.testFilename = TEST_FILENAMES
        self.PredictDataset = None
        with open(STRUCT_FILENAME) as inFile:
            self.structure = inFile.readline()
            self.structure = self.structure.rstrip()
            self.structure = self.structure.split(',')
            for feature in range(len(self.structure)):
                self.encodingFeatures[feature] = {}
            self.structureDataTypes = inFile.readline().rstrip().split(', ')
            self.nrFeature = [int(x) for x in inFile.readline().rstrip().split(', ')]  #x is a number >= 0 if the number of options for that feature is limited, otherwise x is -1
            self.encodingValues = [-1 for x in self.nrFeature]                         #number from 0 to n, where n is the limited number of options for a feature
        self.dataset = pd.DataFrame(columns=self.structure)
        self.nrFeatures = len(self.nrFeature)                                          #number of features of the dataset
        for file in self.fileNames:                                                     #the class have the posibility to read the dataset from multiple files, and then combines them as a whole
            self.dataset = pd.concat([self.dataset, self.ReadFile('Data/TrainData/' + file)])
        self.dataset.set_index('Loan_ID', inplace=True)
    

    def SplitDataset(self, trainRatio, testRatio):
        n = int(self.dataset.shape[0] * trainRatio)
        m = int(self.dataset.shape[0] * testRatio)
        trainDataset = self.dataset.iloc[: n + 1]
        testDataset = self.dataset.iloc[n + 1 : n + m + 1]
        return trainDataset, testDataset


    def getDatasets(self, trainRatio=0.8, testRatio=0.2):
        self.trainDataset = None
        self.testDataset = None
        if self.testFilename == "":
            self.trainDataset, self.testDataset = self.SplitDataset(trainRatio, testRatio)
        else:
            self.trainDataset = self.dataset
            self.testDataset = self.ReadDataset('Data/TestData/' + self.testFilename)
        trainData = self.trainDataset[self.structure[1:-1]]
        trainDataResults = self.trainDataset[self.structure[-1]]
        testData = self.testDataset[self.structure[1:-1]]
        testDataResults = self.testDataset[self.structure[-1]]
        return (trainData.to_numpy().astype(float), testData.to_numpy().astype(float), trainDataResults.to_numpy().astype(float), testDataResults.to_numpy().astype(float))
    

    def getPredictDataset(self):
        dataset = pd.DataFrame(columns=self.structure[:-1])
        with open('Data/PredictData/' + PREDICT_FILENAME, 'r') as inFile:
            for line in inFile.readlines():
                line = line.rstrip()
                line = line.split(',')
                for i in range(1, len(line)):
                    if self.nrFeature[i] != -1:                                        #data preprocessing (turning limited options for a feature into numbers)
                        if line[i] in self.encodingFeatures[i]:
                            line[i] = self.encodingFeatures[i][line[i]]
                        else:
                            self.encodingValues[i] += 1
                            self.encodingFeatures[i][line[i]] = self.encodingValues[i]
                            line[i] = self.encodingFeatures[i][line[i]]
                    else:
                        line[i] = convertNumber(line[i])
                dataset.loc[dataset.shape[0]] = line
        for i in range(1, self.nrFeatures):
            if self.structureDataTypes[i] == 'numeric':
                dataset[self.structure[i]].fillna(dataset[self.structure[i]].mean(), inplace=True)
        dataset.set_index('Loan_ID', inplace=True)
        return dataset, dataset.to_numpy().astype(float)

    
    def PrintResults(self, dataset, results, option):
        results = results.tolist()
        r = [self.encodingFeatures[self.nrFeatures - 1][round(x)] for x in results]
        dataset[self.structure[-1]] = r
        csvColumns = [self.structure[-1]]
        if option == 0:
            columnName = 'Likelihood of the asnwer'
            r = [round(100 * (2 * abs(0.5 - x)), 2) for x in results]
            dataset[columnName] = r
            csvColumns.append(columnName)
        dataset.to_csv("Data/PredictDataResults/Results.csv", columns=csvColumns)