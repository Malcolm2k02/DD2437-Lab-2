import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data/animals.dat", delimiter=",")
props = data.reshape(32, 84)
names = np.loadtxt(
    "data/animalnames.txt",
    delimiter=",",
    dtype=str
)
names = np.char.strip(names, '')

rng = np.random.default_rng(42)
weights = rng.uniform(0, 1, size=(100, 84))

epochs = 20
eta = 0.2

for i in range(epochs):
    neighbourhood = int(50 * (1 - i / (epochs - 1)))
    for j in range(0, 32):
        animal = props[j]
        distances = np.linalg.norm(weights - animal, axis=1)

        winner = np.argmin(distances)
        start  = max(0, winner - neighbourhood)
        end = min(100, winner + neighbourhood + 1)

        weights[start:end] += eta * (animal - weights[start:end])

pos = np.zeros(32, dtype=int)

for j in range(32):
    animal = props[j]

    distances = np.linalg.norm(weights - animal, axis=1)
    winner = np.argmin(distances)

    pos[j] = winner

order = np.argsort(pos)

for index in order:
    print(pos[index], names[index])


