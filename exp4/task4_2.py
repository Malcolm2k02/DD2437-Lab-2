import numpy as np
import matplotlib.pyplot as plt

with open("data/cities.dat", "r") as f:
    text = f.read()

text = text.replace(";", ",")

values = np.fromstring(text, sep=",")
cities = values.reshape(10, 2)

rng = np.random.default_rng(42)
weights = rng.uniform(0, 1, size=(10, 2))

epochs = 50
eta = 0.2

for i in range(epochs):
    current_eta = eta * (1 - i / epochs)
    if i < epochs / 3:
        neighbourhood = 2
    elif i < 2 * epochs / 3:
        neighbourhood = 1
    else:
        neighbourhood = 0
    
    for j in rng.permutation(10):
        city = cities[j]
        distances = np.linalg.norm(weights - city, axis=1)

        winner = np.argmin(distances)
        
        for offset in range(-neighbourhood, neighbourhood + 1):
            neighbour = (winner + offset) % 10

            weights[neighbour] += eta * (
                city - weights[neighbour]
            )


tour = np.vstack([weights, weights[0]])

plt.scatter(
    cities[:, 0], cities[:, 1],
    marker="x", s=70, label="Cities"
)

plt.plot(
    tour[:, 0], tour[:, 1],
    marker="o", label="SOM tour"
)

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("SOM Cyclic Tour")
plt.show()

# Find which SOM node each city belongs to
pos = np.zeros(10, dtype=int)

for j in range(10):
    city = cities[j]

    distances = np.linalg.norm(weights - city, axis=1)
    winner = np.argmin(distances)

    pos[j] = winner

# Sort cities according to their position on the SOM ring
order = np.argsort(pos)

# Actual city coordinates in learned order
tour = cities[order]

# Close the tour
tour = np.vstack([tour, tour[0]])

plt.scatter(
    cities[:, 0], cities[:, 1],
    marker="x", s=70, label="Cities"
)

plt.plot(
    tour[:, 0], tour[:, 1],
    marker="o", label="SOM tour"
)

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("SOM Cyclic Tour")
plt.axis("equal")
plt.show()




