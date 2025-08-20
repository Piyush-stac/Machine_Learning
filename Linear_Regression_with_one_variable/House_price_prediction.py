import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Loading the dataset
dataset = pd.read_csv('Linear_Regression_with_one_variable/house_price_dataset.csv')

# Extracting features and target variable
x_train = dataset.loc[:,'House_Size_sqft'].values
y_train = dataset.loc[:,'Price_1000s'].values

# Scaling The Data
x_mean = np.mean(x_train)
x_std = np.std(x_train)
x_train = (x_train - x_mean) / x_std

y_mean = np.mean(y_train)
y_std = np.std(y_train)
y_train = (y_train - y_mean) / y_std

def compute_gradient(x , y , w , b):
  m=y.shape[0]
  w_i = 0
  b_i = 0
  # Calculating the gradients
  for i in range(0 ,m):
    f_i = w * x[i] + b
    w_i += (f_i - y[i])*x[i]
    b_i += (f_i - y[i])
  w_i/=m
  b_i/=m
  return w_i, b_i

def gradient_descent(x, y, w_i, b_i, alpha, num_iterations):
  m = y.shape[0]
  w = w_i
  b = b_i
  for i in range(num_iterations):
    w_i, b_i = compute_gradient(x, y, w, b)
    w = w - alpha * w_i
    b = b - alpha * b_i
  return w, b

def predicted_price(x, w, b):
  return w * x + b

# Initial parameters
w_initial = 0
b_initial = 0
alpha = 0.01
num_iterations = 10000

# Running gradient descent to find optimal parameters
w , b = gradient_descent(x_train, y_train, w_initial, b_initial, alpha, num_iterations)

# Rescaling the parameters and dataset back to original scale
x_train = x_train * x_std + x_mean
y_train = y_train * y_std + y_mean

# Rescaling the parameters
w = w * y_std / x_std
b = b * y_std + y_mean - w * x_mean

# Plotting the actual data points
plt.scatter(x_train, y_train, marker='x' ,color='red' , label='Data Points')
plt.xlabel('House Size (sqft)')
plt.ylabel('Price (in 1000s)')

# Plotting the results
plt.plot(x_train, predicted_price(x_train, w, b), color='blue', label='Regression Line')
plt.legend()
plt.title("Training Data Set (Housing Price vs Size)")
plt.show()

# Testing
testing_dataset = pd.read_csv('Linear_Regression_with_one_variable/house_price_testing_dataset.csv')
x_test = testing_dataset.loc[:,'House_Size_sqft'].values
y_test = testing_dataset.loc[:,'Price_1000s'].values

# Testing Data Points
plt.scatter(x_test, y_test, marker='x' ,color='red' , label='Data Points')
plt.xlabel('House Size (sqft)')
plt.ylabel('Price (in 1000s)')

# Behaviour of Hypothesis with testing dataset 
plt.plot(x_test, predicted_price(x_test, w, b), color='blue', label='Regression Line')
plt.legend()
plt.title("Testing Data Set (Housing Price vs Size)")
plt.show()