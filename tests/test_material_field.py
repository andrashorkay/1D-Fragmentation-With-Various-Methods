
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Load material field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
field_file = PROJECT_ROOT / "src" / "common" / "material_field.dat"

print(f"Reading material field from: {field_file}")

data = np.loadtxt(field_file)

assert data.ndim == 2 and data.shape[1] == 3, (
    "Expected three columns: x, sigma_c, G_c"
)

x = data[:, 0]
sigma_c = data[:, 1]
G_c = data[:, 2]

# ============================================================================
# Basic checks

assert len(x) > 1
assert len(x) == len(sigma_c) == len(G_c)

assert np.all(np.isfinite(x))
assert np.all(np.isfinite(sigma_c))
assert np.all(np.isfinite(G_c))

assert np.all(sigma_c > 0.0)
assert np.all(G_c > 0.0)

assert np.all(np.diff(x) > 0.0)

# G_c should be spatially constant.
assert np.allclose(G_c, G_c[0])

print("Material field checks passed.")
print()
print(f"Number of points:  {len(x)}")
print(f"x range:           [{x.min():.6e}, {x.max():.6e}] m")
print(f"Mean sigma_c:      {sigma_c.mean():.6e} Pa")
print(f"Std sigma_c:       {sigma_c.std():.6e} Pa")
print(f"Relative scatter:  {sigma_c.std() / sigma_c.mean():.6f}")
print(f"Min sigma_c:       {sigma_c.min():.6e} Pa")
print(f"Max sigma_c:       {sigma_c.max():.6e} Pa")
print(f"G_c:               {G_c[0]:.6e} J/m^2")

# ============================================================================
# Plot spatial strength field

plt.figure()
plt.plot(x * 1e3, sigma_c / 1e6)

plt.xlabel("Position x [mm]")
plt.ylabel(r"$\sigma_c$ [MPa]")
plt.title("Cohesive strength field")
plt.grid()

plt.tight_layout()
plt.show()

# ============================================================================
# Plot strength distribution

plt.figure()
plt.hist(sigma_c / 1e6, bins=30)

plt.xlabel(r"$\sigma_c$ [MPa]")
plt.ylabel("Count")
plt.title("Cohesive strength distribution")
plt.grid()

plt.tight_layout()
plt.show()
