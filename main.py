from LogisticRegression import LogisticRegression
from sklearn.preprocessing import StandardScaler
from Dataset import Dataset
from GraphicalUserInterface import *
from tkinter import *
import os
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np


WIDTH = 1000
HEIGHT = 700
BACKGROUND_COLOR = '#535353'
FONT_COLOR = '#a9a9a9'
BOX_BACKGROUND = '#5c5c5c'
TITLE = "Loan Prediction - ML Project"                                                  #setting up the interface's settings

def CalculateAccuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)


def Run(model, dataModel, scaler):
    dataModel.ReadDataset()
    trainRatio = trainRatioInput.GetText()
    testRatio = testRatioInput.GetText()
    if trainRatio != "":
        trainRatio = float(trainRatio)
    if testRatio != "":
        testRatio = float(testRatio)
    if trainRatio != "":
        trainData, testData, trainDataResults, testDataResults = dataModel.getDatasets(trainRatio, testRatio)
    else:
        trainData, testData, trainDataResults, testDataResults = dataModel.getDatasets()

    learningRate = learningRateInput.GetText()
    nrIterations = iterationsInput.GetText()
    costTrack = costInput.GetText()
    if costTrack != "":
        costTrack = int(costTrack)
        model.setCostTrack(costTrack)
    if nrIterations != "":
        nrIterations = int(nrIterations)
        model.setNrIterations(nrIterations)
    if learningRate != "":
        learningRate = float(learningRate)
        model.setLearningRate(learningRate)
    
    nametags = nametagsInput.GetText().rstrip().split(', ')
    dict = {}
    dict[nametags[0]] = 1                                               #setting up the encoding for True and False label
    dict[nametags[1]] = 0
    dict[1] = nametags[0]
    dict[0] = nametags[1]
    dataModel.setEncodingLabel(nametags)

    trainData = scaler.fit_transform(trainData)                         
    testData = scaler.transform(testData)                               #we use different function for trainData and for testData because we want to standardize the testData with the mean calculated from the trainData

    model.FitModel(trainData, trainDataResults)
    y = model.getCostImprovement()
    logisticPredictions = model.PredictData(testData)
    predictions = [0 if x <= 0.5 else 1 for x in logisticPredictions]
    accuracy = CalculateAccuracy(predictions, testDataResults)
    truePositives = 0
    trueNegatives = 0
    falsePositives = 0
    falseNegatives = 0
    for i in range(len(testDataResults)):
        if predictions[i] == 1 and testDataResults[i] == 1:
            truePositives += 1
        if predictions[i] == 0 and testDataResults[i] == 1:
            falseNegatives += 1
        if predictions[i] == 1 and testDataResults[i] == 0:
            falsePositives += 1
        if predictions[i] == 0 and testDataResults[i] == 0:
            trueNegatives += 1
    accuracyLabel.SetText("Accuracy: " + str(round(accuracy * 100, 2)) + '%')
    trueNegativeLabel.SetText("True Negatives: " + str(trueNegatives))
    truePositiveLabel.SetText("True Positives: " + str(truePositives))
    falseNegativeLabel.SetText("False Negatives: " + str(falseNegatives))
    falsePositiveLabel.SetText("False Positives: " + str(falsePositives))
    costPlot.plot(y)
    costPlot.set_title("Gradient Descent")
    canvas.draw()


def StartPredicting(dataModel, scaler):
    predictLabel.SetText('')
    ds, predictData = dataModel.getPredictDataset()
    predictData = scaler.transform(predictData)
    logisticPredictions = model.PredictData(predictData)
    dataModel.PrintResults(ds, logisticPredictions, radiobuttonOption.get())
    predictLabel.SetText("Succesfuly printed the results.")



if __name__ == '__main__':
    Window = Tk()
    Window.geometry(str(WIDTH) + 'x' + str(HEIGHT))
    Window.title(TITLE)
    Window.configure(bg = BACKGROUND_COLOR)
    model = LogisticRegression()
    dataModel = Dataset()
    scaler = StandardScaler()                                           #using StandardScaler to standardize our data using this formula: x_scaled = (x-mean)/sd

    learningRateLabel = LabelBox(Window, 20, 30, 'Learning Rate:*', BACKGROUND_COLOR, FONT_COLOR)
    learningRateInput = InputBox(Window, 135, 30, 15, 1, BOX_BACKGROUND, FONT_COLOR, 1)
    iterationsLabel = LabelBox(Window, 300, 30, 'Nr. Iterations:*', BACKGROUND_COLOR, FONT_COLOR)
    iterationsInput = InputBox(Window, 410, 30, 15, 1, BOX_BACKGROUND, FONT_COLOR, 1)
    costImprovementLabel = LabelBox(Window, 570, 30, 'Nr. Iterations for Tracking Error:*', BACKGROUND_COLOR, FONT_COLOR)
    costInput = InputBox(Window, 800, 30, 15, 1, BOX_BACKGROUND, FONT_COLOR, 1)
    trainRatioLabel = LabelBox(Window, 20, 85, 'Train Ratio:*', BACKGROUND_COLOR, FONT_COLOR)
    trainRatioInput = InputBox(Window, 135, 85, 15, 1, BOX_BACKGROUND, FONT_COLOR, 1)
    testRatioLabel = LabelBox(Window, 300, 85, 'Test Ratio:*', BACKGROUND_COLOR, FONT_COLOR)
    testRatioInput = InputBox(Window, 410, 85, 15, 1, BOX_BACKGROUND, FONT_COLOR, 1)
    nametagsLabel = LabelBox(Window, 570, 85, 'True, Negative:', BACKGROUND_COLOR, FONT_COLOR)
    nametagsInput = InputBox(Window, 800, 85, 15, 1, BOX_BACKGROUND, FONT_COLOR, 1)
    radiobuttonOption = IntVar()
    likelihoodRadioButton = RadioBox(Window, 200, 125, "Return the likelihood of the event", 0, radiobuttonOption, BACKGROUND_COLOR, FONT_COLOR)
    boolRadioButton = RadioBox(Window, 600, 125, "Return the predicted value", 1, radiobuttonOption, BACKGROUND_COLOR, FONT_COLOR)
    startButton = ButtonBox(Window, 465, 165, "   Start   ", BOX_BACKGROUND, FONT_COLOR, BOX_BACKGROUND, FONT_COLOR, Run, model, dataModel, scaler)
    accuracyLabel = LabelBox(Window, 436, 245, "", BACKGROUND_COLOR, FONT_COLOR)
    trueNegativeLabel = LabelBox(Window, 130, 210, "", BACKGROUND_COLOR, FONT_COLOR)
    truePositiveLabel = LabelBox(Window, 320, 210, "", BACKGROUND_COLOR, FONT_COLOR)
    falseNegativeLabel = LabelBox(Window, 510, 210, "", BACKGROUND_COLOR, FONT_COLOR)
    falsePositiveLabel = LabelBox(Window, 700, 210, "", BACKGROUND_COLOR, FONT_COLOR)

    costFigure = Figure(figsize=(5,3), dpi = 100)
    costPlot = costFigure.add_subplot(111)
    
    canvas = FigureCanvasTkAgg(costFigure, master=Window)
    canvas.draw()
    canvas.get_tk_widget().place(x=250, y=280)

    predictButton = ButtonBox(Window, 450, 625, "Start Predicting", BOX_BACKGROUND, FONT_COLOR, BOX_BACKGROUND, FONT_COLOR, StartPredicting, dataModel, scaler)
    predictLabel = LabelBox(Window, 390, 660, '', BACKGROUND_COLOR, FONT_COLOR)


    Window.mainloop()
    
