import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sep import sep
import scipy.optimize as opt

df = pd.read_csv('healthcare-dataset-stroke-data.csv');

# Values of every column in this print()

# print("gender:", df.gender.unique())                      # ['Male' 'Female' 'Other'] [ 0 1 2 ] # Done
# print("status:", df.ever_married.unique())                # ['Yes' 'No'] [ 1 0 ] # Done
# print("residence_type:", df.Residence_type.unique())      # ['Urban' 'Rural'] [ 0 1 ] # Done
# print("work:", df.work_type.unique())                     # ['Private' 'Self-employed' 'Govt_job' 'children' 'Never_worked'] [ 0 1 2 3 4 ]
# print("smoking:", df.smoking_status.unique())             # ['formerly smoked' 'never smoked' 'smokes' 'Unknown'] [ 0 1 2 3 ]
# print("hypertension:", df.hypertension.unique())          # [0 1]
# print("heart disease:", df.heart_disease.unique())        # [1 0]
# print("stroke:", df.stroke.unique())                      # [1 0]

cats = ['Ones', 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
scl  = ['age', 'bmi', 'avg_glucose_level'] # for_rescaling_data

# Rescaling data
df[scl] = (df[scl] - df[scl].mean()) / df[scl].std()

# Remove unknown values in smoking_status
mask = (df['stroke'] == 1) & (df['smoking_status'] == 'Unknown')
df.loc[mask, 'smoking_status'] = 'formerly smoked'

mask1 = (df['stroke'] == 0) & (df['smoking_status'] == 'Unknown')
df.loc[mask1, 'smoking_status'] = 'never smoked'

# Convert all text values to numerical one
df['gender'].replace('Male', 0, inplace=True)
df['gender'].replace('Female', 1, inplace=True)
df['gender'].replace('Other', 1, inplace=True) # Only there's one other person we will replace with the majority (Females)

df['ever_married'].replace('Yes', 1, inplace=True)
df['ever_married'].replace('No', 0, inplace=True)

df['Residence_type'].replace('Urban', 0, inplace=True)
df['Residence_type'].replace('Rural', 1, inplace=True)

df['work_type'].replace('Private', 0, inplace=True)
df['work_type'].replace('Self-employed', 1, inplace=True)
df['work_type'].replace('Govt_job', 2, inplace=True)
df['work_type'].replace('children', 3, inplace=True)
df['work_type'].replace('Never_worked', 4, inplace=True)

df['smoking_status'].replace('formerly smoked', 0, inplace=True)
df['smoking_status'].replace('never smoked', 1, inplace=True)
df['smoking_status'].replace('smokes', 2, inplace=True)

# Fill NaN values in bmi
df['bmi'].fillna(df['bmi'].median(), inplace=True)

# Drop unnesscary column ID
df = df.drop(columns=['id'])
df.insert(0, 'Ones', 1)

X = df[cats].values
y = np.matrix(df['stroke'].values).T

theta = np.matrix(np.zeros(X.shape))

print("X values = ")
print(X)
print("X shape = ")
print(X.shape)

sep()

print("Y values = ")
print(y)
print("Y shape = ")
print(y.shape)

sep()

print("theta values = ")
print(theta)

print("theta shape = ")
print(theta.shape)

sep()

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

#################################################################

def hypothesis(x, theta):
  return sigmoid(x * theta.T)

#################################################################

def cost(theta, x, y):
  J = 0
  for i in range(len(x)):
    J += (-np.log(hypothesis(x[i], theta[i]))) * y[i]
    J += (-np.log(1 - hypothesis(x[i], theta[i]))) * (1 - y[i])
  return np.sum(J) / len(x)

print("Cost value = ")
print(cost(theta, X, y))

sep()

#################################################################

def gradient_descent(theta, x, y):
  m = len(x)

  params = int(theta.shape[1])

  grad = np.zeros(params)

  error = hypothesis(x, theta) - y
  for j in range(10):

    for i in range(params):
      grad[i] = grad[i] - (.01* (np.sum((hypothesis(x[i], theta[i]) - y[i]) * x[i])))
      # term = (hypothesis(x[i], theta[i]) - y[i]) * x[i]
      # grad[i] = grad[i] - ((0.0002 * np.sum(term)) / len(x))
  return np.matrix(grad).reshape(1, 11)

grad = gradient_descent(theta, X, y)

print(grad)

sep()

def predict(theta, x):
  prob = hypothesis(x, theta)
  for i in prob:
    print(i)

print(predict(grad, X))
