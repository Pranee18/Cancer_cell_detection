import numpy as np
import pandas as pd
import sklearn.datasets
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from google.colab import files
uploaded = files.upload()

# loading the data from the dataset
breast_cancer_dataset = pd.read_csv('breast_cancer_data.csv')
print(breast_cancer_dataset)

# loading the data to a data frame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)

# print the first 5 rows of the dataframe
data_frame.head()

# adding the 'target' column to the data frame
data_frame['label'] = breast_cancer_dataset.target

# print last 5 rows of the dataframe
data_frame.tail()

# number of rows and columns in the dataset
data_frame.shape

# getting some information about the data
data_frame.info()

# checking for missing values
data_frame.isnull().sum()

# statistical measures about the data
data_frame.describe()

# checking the distribution of Target Varibale
data_frame['label'].value_counts()
data_frame.groupby('label').mean()

#Separating the features and target
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']
print(X)
print(Y)

#Splitting the data into training data & Testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

#MODEL TRAINING

#LOGISTIC REGRESSION
# Suppress warnings
warnings.filterwarnings("ignore")
# training the Logistic Regression model using Training data
model = LogisticRegression()
model.fit(X_train, Y_train)
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy on training data = ', training_data_accuracy)
# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy on test data = ', test_data_accuracy)

#SVM MODEL
#training the SVM model using Training data
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, Y_train)
# accuracy on training data
Y_train_pred = svm_model.predict(X_train)
training_data_accuracy_svm = accuracy_score(Y_train, Y_train_pred)
print('Accuracy on training data = ', training_data_accuracy_svm)
# accuracy on test data
Y_test_pred = svm_model.predict(X_test)
testing_data_accuracy_svm = accuracy_score(Y_test, Y_test_pred)
print('Accuracy on testing data = ', testing_data_accuracy_svm)

#KNN MODEL
#training the KNN model using Training data
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, Y_train)
# accuracy on training data
Y_train_pred_knn = knn_model.predict(X_train)
training_data_accuracy_knn = accuracy_score(Y_train, Y_train_pred_knn)
print('Accuracy on training data = ', training_data_accuracy_knn)
# accuracy on test data
Y_test_pred_knn = knn_model.predict(X_test)
testing_data_accuracy_knn = accuracy_score(Y_test, Y_test_pred_knn)
print('Accuracy on testing data = ', testing_data_accuracy_knn)

#BUILDING A PREDICTIVE SYSTEM 
input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)
# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)
# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)
if (prediction[0] == 0):
  print('The Breast cancer is Malignant')
else:
  print('The Breast Cancer is Benign')
