# MachineLearningLoanPrediction
I've created a Supervised Machine Learning Algorithm which use Logistic Regression to predict an outcome. It can be feed with any type of dataset that has 2 outcomes possible.
I've tested it on a dataset in which you have to predict if a person gets the loan he applied for.
Link to the dataset: https://www.openml.org/search?type=data&status=active&id=43595

Setup:
1. In dataset_structure file, on the first line should be the columns from the dataset (the last column should always be the label, in this case Loan_Status), on the second line, for each column there should be it's type ('string' if it's a string and 'numeric' if it's a number) and on the third line the number of possibilities for that feature (-1 -> infinite, a number if there are finite, for example on a Gender column you can have 3 possibilities: Male, Female or Unknown).
2. In the Interface all fields marked with '*' are optional.
3. As default Train Ratio is set to 0.8 (80%) and Test Ratio to 0.2 (20%).
4. The dataset for which it should make a prediction should be placed in 'Data/PredictData' and the output will be provided in 'Data/PredictDataResults'
![Untitled](https://user-images.githubusercontent.com/66367023/196531615-137c0d31-ec5f-403c-919c-c72cdcc66040.png)
