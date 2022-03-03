import matplotlib.pyplot as plt
import numpy as np
import csv

def gradient_descent(x, y):
  m_curr = b_curr = 0  # starting values [ initial values of thetas ], Theta-1 = 0, Theta-0 = 0
  n = len(x)  # no. of datapoints
  learning_rate = 0.08

  for i in range(n):
    y_pred = m_curr * x + b_curr # hypothesis function - predicted value
    cost = (1 / n) * sum([
      val ** 2 for val in (y - y_pred)
    ]) # cost function

    bd = - (2 / n) * sum(y - y_pred) # theta0
    md = - (2 / n) * sum(x * (y - y_pred)) # theta1
    
    b_curr = b_curr - learning_rate * bd
    m_curr = m_curr - learning_rate * md

  print(md, bd)

  # plotting model
  plt.plot(x, y, 'o')
  m, b = np.polyfit(x, y, 1)
  plt.plot(x, m * x + b)
  plt.grid()
  plt.show()


x = []
y = []

with open('dataset.csv', 'r') as csv_file:  # read csv file
  csv_reader = csv.reader(csv_file)
  csv_reader = csv.DictReader(csv_file) # to make output like this {x: n, y: n}
  for i in csv_reader:
    x.append(int(i['x']))
    y.append(float(i['y']))

    print(i) # thetas

gradient_descent(np.array(x), np.array(y))
