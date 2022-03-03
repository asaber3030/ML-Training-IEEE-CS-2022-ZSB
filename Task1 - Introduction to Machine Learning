import numpy as np
import sys

n = 2

a = np.zeros((n, n + 1))
x = np.zeros(n)

print("aX0+bX1=c, dX0+eX1=f")
print("Enter coefficients: ")

for i in range(n):

  for j in range(n + 1):
    if i == 0 and j == 0:
      a[i][j] = float(input("a="))
    elif i == 0 and j == 1:
      a[i][j] = float(input("b="))
    elif i == 0 and j == 2:
      a[i][j] = float(input("c="))
    elif i == 1 and j == 0:
      a[i][j] = float(input("d="))
    elif i == 1 and j == 1:
      a[i][j] = float(input("e="))
    elif i == 1 and j == 2:
      a[i][j] = float(input("f="))

for i in range(n):
  if a[i][i] == 0:
    sys.exit("Divide By Zero Detected")

  for j in range(i + 1, n):
    ratio = a[j][i] / a[i][i]
    for k in range(n + 1):
      a[j][k] = a[j][k] - ratio * a[i][k]

x[n - 1] = a[n - 1][n] / a[n - 1][n - 1]

for i in range(n - 2, -1, -1):
  x[i] = a[i][n]
  for j in range(i + 1, n):
    x[i] = x[i] - a[i][j] * x[j]

  x[i] = x[i] / a[i][i]

for i in range(n):
  print(f'X{i} = {x[i]}', end='\t')
