from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

RUN_PATH = Path("data/PF/mesh_convergence/N_10000")

L = 0.005

DAMAGE_LIMITS = [
    0.90,
    0.92,
    0.94,
    0.95,
    0.96,
    0.98,
    0.99,
]


def step_number(path):
    match = re.search(r"nodal_(\d+)\.out", path.name)

    if match is None:
        return -1

    return int(match.group(1))


def read_damage_profile(file_path):
    data = np.loadtxt(file_path)

    x = data[:, 0]
    damage = data[:, 4]

    x_unique, inverse = np.unique(x, return_inverse=True)

    damage_1d = np.full(len(x_unique), np.inf)
    np.minimum.at(damage_1d, inverse, damage)

    return x_unique, damage_1d


def count_fragments(damage, damage_limit):
    broken = damage >= damage_limit

    num_cracks = np.count_nonzero(broken & np.concatenate(([True], ~broken[:-1])))

    return num_cracks + 1


field_files = sorted(
    (RUN_PATH / "fields").glob("nodal_*.out"),
    key=step_number,
)

energies = np.loadtxt(RUN_PATH / "energies_pf.out")

energy_steps = energies[:, 0].astype(int)
energy_times = energies[:, 1]

step_to_time = dict(zip(energy_steps, energy_times))


results = {
    damage_limit: {
        "steps": [],
        "times": [],
        "num_fragments": [],
    }
    for damage_limit in DAMAGE_LIMITS
}


for field_file in field_files:
    step = step_number(field_file)

    if step not in step_to_time:
        continue

    time = step_to_time[step]

    _, damage = read_damage_profile(field_file)

    for damage_limit in DAMAGE_LIMITS:
        num_fragments = count_fragments(
            damage,
            damage_limit,
        )

        results[damage_limit]["steps"].append(step)
        results[damage_limit]["times"].append(time)
        results[damage_limit]["num_fragments"].append(num_fragments)


for damage_limit in DAMAGE_LIMITS:
    results[damage_limit]["steps"] = np.array(results[damage_limit]["steps"])

    results[damage_limit]["times"] = np.array(results[damage_limit]["times"])

    results[damage_limit]["num_fragments"] = np.array(
        results[damage_limit]["num_fragments"]
    )


print("Damage-threshold sensitivity:")
print()
print("damage_limit  final_fragments  mean_size_mm  " "first_fragmentation_us")

for damage_limit in DAMAGE_LIMITS:
    data = results[damage_limit]

    final_fragments = data["num_fragments"][-1]
    mean_size_mm = L / final_fragments * 1e3

    fragmented = data["num_fragments"] > 1

    if np.any(fragmented):
        first_time = data["times"][np.argmax(fragmented)] * 1e6
    else:
        first_time = np.nan

    print(
        f"{damage_limit:12.2f}"
        f"{final_fragments:17d}"
        f"{mean_size_mm:14.6f}"
        f"{first_time:24.6f}"
    )


plt.figure()

for damage_limit in DAMAGE_LIMITS:
    data = results[damage_limit]

    plt.step(
        data["times"] * 1e6,
        data["num_fragments"],
        where="post",
        label=f"{damage_limit:.2f}",
    )

plt.xlabel(r"Time [$\mu$s]")
plt.ylabel("Number of fragments")
plt.legend(title=r"$d_{\mathrm{lim}}$")
plt.tight_layout()


final_fragments = [results[limit]["num_fragments"][-1] for limit in DAMAGE_LIMITS]

mean_sizes = [L / num_fragments * 1e3 for num_fragments in final_fragments]


plt.figure()

plt.plot(
    DAMAGE_LIMITS,
    final_fragments,
    marker="o",
)

plt.xlabel(r"Damage threshold $d_{\mathrm{lim}}$")
plt.ylabel("Final number of fragments")
plt.tight_layout()


plt.figure()

plt.plot(DAMAGE_LIMITS, mean_sizes, marker="o")

plt.xlabel(r"Damage threshold $d_{\mathrm{lim}}$")
plt.ylabel("Mean fragment size [mm]")
plt.tight_layout()

final_file = field_files[-1]

x, damage = read_damage_profile(final_file)

plt.figure()

plt.plot(x * 1e3, damage)

for damage_limit in DAMAGE_LIMITS:
    plt.axhline(
        damage_limit,
        linestyle="--",
        label=f"{damage_limit:.2f}",
    )

plt.xlabel("x [mm]")
plt.ylabel("Damage")
plt.ylim(0.85, 1.005)
plt.legend(title=r"$d_{\mathrm{lim}}$")
plt.tight_layout()

plt.show()
