import numpy as np
import matplotlib.pyplot as plt

from src.data_setup import x_train

from exp3_3.competitive_learning import (
    competitive_learning_dead_unit_test,
    competitive_learning_multi_winner
)


# ============================================================
# Settings
# ============================================================

n_units = 15
eta = 0.05
epochs = 50
seed = 42


# ============================================================
# Vanilla CL
# ============================================================

centers_vanilla, wins_vanilla = (
    competitive_learning_dead_unit_test(
        x_train,
        n_units=n_units,
        eta=eta,
        epochs=epochs,
        seed=seed
    )
)


# ============================================================
# Multi-winner CL
# ============================================================

centers_multi, wins_multi = (
    competitive_learning_multi_winner(
        x_train,
        n_units=n_units,
        eta=eta,
        epochs=epochs,
        n_winners=4,
        seed=seed
    )
)


centers_vanilla = centers_vanilla.flatten()
centers_multi = centers_multi.flatten()


# ============================================================
# Count dead units
# ============================================================

dead_vanilla = np.sum(
    wins_vanilla == 0
)

dead_multi = np.sum(
    wins_multi == 0
)


# ============================================================
# Results
# ============================================================

print("\n========== VANILLA CL ==========")

print("Centers:")
print(np.sort(centers_vanilla))

print("\nWins:")
print(wins_vanilla)

print(
    f"\nDead units: "
    f"{dead_vanilla}/{n_units}"
)


print("\n========== MULTI-WINNER CL ==========")

print("Centers:")
print(np.sort(centers_multi))

print("\nWins:")
print(wins_multi)

print(
    f"\nDead units: "
    f"{dead_multi}/{n_units}"
)


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(10, 4))


# Training samples
plt.scatter(
    x_train,
    np.zeros_like(x_train),
    marker=".",
    label="Training samples"
)


# Vanilla centers
plt.scatter(
    centers_vanilla,
    np.full_like(
        centers_vanilla,
        0.15
    ),
    marker="x",
    s=80,
    label="Vanilla CL centers"
)


# Multi-winner centers
plt.scatter(
    centers_multi,
    np.full_like(
        centers_multi,
        -0.15
    ),
    marker="o",
    s=60,
    label="Multi-winner CL centers"
)


plt.axvline(
    0,
    linestyle="--",
    alpha=0.5
)

plt.axvline(
    2 * np.pi,
    linestyle="--",
    alpha=0.5
)

plt.yticks([])

plt.xlabel("x")

plt.title(
    "Dead units: vanilla CL vs multi-winner CL"
)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()