# IDS 575
# University of Illinois at Chicago
# Spring 2023
# quiz #03
#
# Do not rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages (not specified in
# the assignment) then you need prior approval from course staff.
# This part of the assignment will be graded automatically using Gradescope.
# =========================================================================================================

import numpy as np
import pandas as pd

# Data split
from sklearn.model_selection import StratifiedShuffleSplit

def splitTrainTest(df, size):
  split = StratifiedShuffleSplit(n_splits=1, test_size=size, random_state=0)

  # For each pair of train and test indices,
  X = df.drop('Class', axis=1)
  y = df.Class  
  for trainIndexes, testIndexes in split.split(X, y):
    X_train, y_train = X.iloc[trainIndexes], y.iloc[trainIndexes]
    X_test, y_test = X.iloc[testIndexes], y.iloc[testIndexes]

  return (X_train, y_train), (X_test, y_test)

# using built-in model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix 

def doLogisticRegression(X, y, normalize=False):
  # If normalize option is enabled,
  if normalize:
    # For each feature (indexed by j as usual)
    for j in X.columns:
      # Subtract its column mean and update the value.
      X[j] -= X[j].mean()

      # Divide by its standard deviation and update the value.
      X[j] /= X[j].std()

  # Instanciate an object from Logistic Regression class.
  lr = LogisticRegression()

  # Perform training and prediction.
  lr.fit(X, y)
  y_pred = lr.predict(X)
      
  # Return training accuracy and confusion matrix.
  return accuracy_score(y, y_pred), confusion_matrix(y, y_pred), lr

# Implement your own logistic regression model
class MyLogisticRegression:
  # Randomly initialize the parameter vector.
  theta = None

  def logistic(self, z):
    # Return the sigmoid function value.
    ##############################################################
    # TO-DO: Complete the evaluation of logistic function given z.
    logisticValue = 1 / (1 + np.exp(-z))
    ##############################################################
    return logisticValue

  def logLikelihood(self, X, y):
    # Compute the log-likelihood hood of all training examples.
    # X: (m x (n+1)) data matrix
    # y: (m x 1) output vector    

    # If theta parameter has not trained yet,
    if not isinstance(self.theta, np.ndarray):
      return 0.0

    # Compute the linear hypothesis given individual examples (as a whole).
    h_theta = self.logistic(np.dot(X, self.theta))

    # Evalaute the two terms in the log-likelihood.    
    #################################################################
    # TO-DO: Compute the two terms in the log-likelihood of the data.
    probability1 = y * np.log(h_theta)
    probability0 = (1 - y) * np.log(1 - h_theta)
    #################################################################

    # Return the average of the log-likelihood
    m = X.shape[0]
    return (1.0/m) * np.sum(probability1 + probability0) 

  def fit(self, X, y, alpha=0.01, epoch=50):
    # Extract the data matrix and output vector as a numpy array from the data frame.
    # Note that we append a column of 1 in the X for the intercept.
    X = np.concatenate((np.array(X), np.ones((X.shape[0], 1), dtype=np.float64)), axis=1)
    y = np.array(y)  

    # Run mini-batch gradient descent.
    self.miniBatchGradientDescent(X, y, alpha, epoch)

  def predict(self, X):
    # Extract the data matrix and output vector as a numpy array from the data frame.
    # Note that we append a column of 1 in the X for the intercept.
    X = np.concatenate((np.array(X), np.ones((X.shape[0], 1), dtype=np.float64)), axis=1)

    # Perfrom a prediction only after a training happens.
    if isinstance(self.theta, np.ndarray):
      y_pred = self.logistic(X.dot(self.theta))
      ####################################################################################
      # TO-DO: Given the predicted probability value, decide your class prediction 1 or 0.
      y_pred_class = (y_pred >= 0.5).astype(int)
      ####################################################################################
      return y_pred_class
    return None

  def miniBatchGradientDescent(self, X, y, alpha, epoch, batch_size=100):    
    (m, n) = X.shape
  
    # Randomly initialize our parameter vector. (DO NOT CHANGE THIS PART!)
    # Note that n here indicates (n+1) because X is already appended by the intercept term.
    np.random.seed(2) 
    self.theta = 0.1*(np.random.rand(n) - 0.5)
    print('L2-norm of the initial theta = %.4f' % np.linalg.norm(self.theta, 2))
    
    # Start iterations
    for iter in range(epoch):
      # Print out the progress report for every 1000 iteration.
      if (iter % 5) == 0:
        print('+ currently at %d epoch...' % iter)   
        print('  - log-likelihood = %.4f' % self.logLikelihood(X, y))

      # Create a list of shuffled indexes for iterating training examples.     
      indexes = np.arange(m)
      np.random.shuffle(indexes)

      # For each mini-batch,
      for i in range(0, m - batch_size + 1, batch_size):
        # Extract the current batch of indexes and corresponding data and outputs.
        indexSlice = indexes[i:i+batch_size]        
        X_batch = X[indexSlice, :]
        y_batch = y[indexSlice]

        # For each feature
        for j in np.arange(n):
          gradient = np.dot(X_batch.T, (self.logistic(np.dot(X_batch, self.theta)) - y_batch)) / batch_size
        self.theta -= alpha * gradient
          ####################################################################################
          # TO-DO: Perform like a batch gradient desceint within the current mini-batch.
          # Note that your algorithm must update self.theta[j].


          ####################################################################################
          
  
def doMyLogisticRegression(X, y, alpha, epoch, normalize=False):
  # If normalize option is enabled,
  if normalize:
    # For each feature (indexed by j as usual)
    for j in X.columns:
      # Subtract its column mean and update the value.
      X[j] -= X[j].mean()

      # Divide by its standard deviation and update the value.
      X[j] /= X[j].std()

  # Instanciate an object from Logistic Regression class.
  lr = MyLogisticRegression()

  # Perform training and prediction.
  lr.fit(X, y, alpha, epoch,)
  y_pred = lr.predict(X)
      
  # Return training accuracy and confusion matrix.
  return accuracy_score(y, y_pred), confusion_matrix(y, y_pred), lr

def testYourCode(X_train, y_train, X_test, y_test, alpha, epoch):
  # Test the code with scikit-learn.
  trainAcc, trainConf, lr = doLogisticRegression(X_train, y_train, normalize=True)
  y_test_pred = lr.predict(X_test)
  testAcc, testConf = accuracy_score(y_test, y_test_pred), confusion_matrix(y_test, y_test_pred)
  print("Scikit's training/test accuracies = %.4f / %.4f" % (trainAcc, testAcc))
  print("Scikit's training/test confusion matrix\n %s\n %s" % (trainConf, testConf))
  theta = np.append(lr.coef_[0], lr.intercept_)
  print(theta)

  # Test the code with your own version.
  myTrainAcc, myTrainConf, myLR = doMyLogisticRegression(X_train, y_train, alpha, epoch, normalize=True)
  my_y_test_pred = myLR.predict(X_test)
  myTestAcc, myTestConf = accuracy_score(y_test, my_y_test_pred), confusion_matrix(y_test, my_y_test_pred)
  print("My training/test accuracies = %.4f / %.4f" % (myTrainAcc, myTestAcc))
  print("My training/test confusion matrix\n %s\n %s" % (myTrainConf, myTestConf))
  print(myLR.theta)

# Use the main function to test your code when running it from a terminal
# output should be a list of floats
def main():
  # Load the data
  FraudDataset = pd.read_csv('fraud.csv')
  print(type(FraudDataset))
  print(FraudDataset.keys())

  (X_train, y_train), (X_test, y_test) = splitTrainTest(FraudDataset, 0.2)

  testYourCode(X_train, y_train, X_test, y_test, 0.05, 100)



################ Do not make any changes below this line ################
if __name__ == '__main__':
	main()