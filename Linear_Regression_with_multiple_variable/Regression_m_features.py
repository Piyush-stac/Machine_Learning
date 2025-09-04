import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("multifeature_house_prices_300.csv")
print(data.head(50))

m = data.shape[1] -1 # number of features
n = data.shape[0] # number of samples

# Prepare the feature matrix X and target vector y
x = np.zeros((m,n))
y = data.iloc[:,m].values

for i in range (0,m):
  x[i] = data.iloc[:,i]

# Feature Normalization
for i in range(0 , m):
  x[i][:]  = (x[i][:] - np.mean(x[i][:]))/np.std(x[i][:])
y[:]  = (y[:] - np.mean(y[:]))/np.std(y[:])

parameter = np.zeros(m)

# Hypothesis function for multiple features
def calculate_hypothesis(parameter , b , x , m , n , i):
  y=0
  for j in range(0 , m):
    y += parameter[j]*x[j][i]
  y += b
  return y

# Calculating the gradient
def calculate_gradient(parameter , b , x , y , m , n):
  d_w = np.zeros(m)
  d_b = 0
  for i in range(0 , n):
    f_i = calculate_hypothesis(parameter , b , x , m , n , i)
    for j in range(0 , m):
      d_w[j] += (f_i - y[i])*x[j][i]
    d_b += (f_i - y[i])
  
  d_w = d_w/n
  d_b = d_b/n
  return d_w , d_b

# applying gradient descent to find optimal parameters
def gradient_descent(x , y , parameter_in , b_in , alpha , iteration , m , n):
  parameter = parameter_in
  b = b_in
  for i in range (0 , iteration ):
    d_w , d_b = calculate_gradient(parameter , b , x , y , m , n)
    for j in range(0 , m):
      parameter[j] = parameter[j] - (alpha*d_w[j])
    b = b - (alpha*d_b)

  return parameter , b

# Running gradient descent
alpha = 0.001
iteration = 10000
b_in = 0
parameter_in = np.zeros(m)
parameter , b = gradient_descent(x , y , parameter_in , b_in , alpha , iteration , m , n)

# Rescaling the parameters to original scale
std_y = np.std(data.iloc[:,m])
mean_y = np.mean(data.iloc[:,m])
for j in range(0 , m):
  std_xj = np.std(data.iloc[:,j])
  mean_xj = np.mean(data.iloc[:,j])
  parameter[j] = parameter[j]*(std_y/std_xj)
  b = b*std_y + mean_y
  b = b - parameter[j]*mean_xj

# Rescaling the data to original scale
for i in range(0 , n):
  y[i] = y[i]*std_y + mean_y
  for j in range(0 , m):
    x[j][i] = x[j][i]*np.std(data.iloc[:,j]) + np.mean(data.iloc[:,j])


# Plotting the final hypothesis line
y_pred = np.zeros(n)
for i in range(0 , n):
  y_pred[i] = calculate_hypothesis(parameter , b , x , m , n , i)


residuals = y - y_pred
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residuals (Error)")
plt.title("Residual Plot")
plt.show()


# Final Optimal parameters
print("The optimal parameters are : ")
print(parameter)
print("The optimal bias is : ")
print(b)