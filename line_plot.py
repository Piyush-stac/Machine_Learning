import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0 , 2.0])
y_train = np.array([300 , 500])

# Expected Output Line 
plt.scatter(x_train , y_train ,color = 'red', marker='x' , label='Expected data points')
plt.xlabel('Size(in 1000s of square feet)')
plt.ylabel('Price (in 1000s of dollars)')
plt.title('House Price vs Size')
# plt.show()

def hypothesis(x , w , b):
  length = x.shape[0]
  y = np.zeros(length)
  for i in range(length):
    y[i] = w * x[i] + b # y[i] is the output for the i-th input(x[i])
  return y

# Plotting the hypothesis line
plt.plot(x_train, hypothesis(x_train, 200, 100), 'blue' , label='Hypothesis Line')
plt.legend()
plt.show()
