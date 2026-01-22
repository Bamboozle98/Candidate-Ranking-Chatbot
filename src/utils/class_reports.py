import numpy as np


def class_report(y, name="y"):
    classes, counts = np.unique(y, return_counts=True)
    fracs = counts / counts.sum()
    print(f"{name} distribution:")
    for c, n, f in zip(classes, counts, fracs):
        print(f"  class {c}: n={n} ({f:.2%})")