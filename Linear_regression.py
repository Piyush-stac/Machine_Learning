import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0])   # Features / Inputs
y_train = np.array([300.0, 500.0]) # Output / Labels

# Plotting the expected output points
plt.scatter(x_train, y_train, color='red', marker='x', label='Expected data points')
plt.xlabel('Size (in 1000s of square feet)')
plt.ylabel('Price (in 1000s of dollars)') 


# Hypothesis function for linear regression
def hypothesis(x, w, b):
    m = x.shape[0]
    y=[]
    for i in range(m):
        y.append(w * x[i] + b)
    return y

# Calculating the cost function
def cost_function(x , y , w , b):
  m = x.shape[0]
  cost = 0; 
  
  for i in range(0 , m):
    f_i= w*x[i] + b
    cost+= (f_i - y[i])**2
  cost = cost / (2*m)
  return cost

# Calculating the gradient
def calculate_gradient(x,y,w,b):
  m = x.shape[0]
  d_w =0
  d_b = 0
  for i in range(0,m):
    f_i = w*x[i] + b
    d_w += (f_i - y[i])*x[i]
    d_b += (f_i - y[i])
  d_w = d_w/m
  d_b = d_b/m
  return d_w , d_b

# applying gradient descent to find optimal parameters
def gradient_descent(x , y , w_in , b_in , alpha , iteration):
  J_history = []
  parameter_history = []
  w = w_in 
  b = b_in
  for i in range(1 , iteration+1):
    d_w , d_b = calculate_gradient(x,y,w,b)
    w = w - (alpha*d_w)
    b = b - (alpha*d_b)

    J_history.append(cost_function(x,y,w,b))
    parameter_history.append([w,b])
  return w, b, J_history, parameter_history

w_in = 0
b_in = 0
alpha = 0.01
iteration = 10000
w, b, J_history, parameter_history = gradient_descent(x_train, y_train, w_in, b_in, alpha, iteration)

print("Optimal parameters: w =", w, ", b =", b)

plt.plot(x_train, hypothesis(x_train, w, b), 'blue', label='Hypothesis Line')
plt.legend()
plt.show()

