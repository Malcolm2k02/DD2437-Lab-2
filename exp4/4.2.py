import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data/cities.dat", delimiter=",")
cities = data.reshape(10, 2)

print(cities)
print(cities.shape)