"""Day 01 — ML helper functions written in plain Python (no NumPy).

Writing these by hand once makes it clear what libraries do for you later.
They are tested in test_day01.py.
"""

import math
import random
from collections import Counter


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean() needs at least one value")
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("median() needs at least one value")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def mode(values: list) -> list:
    """All most-frequent values, sorted (there can be more than one)."""
    if not values:
        raise ValueError("mode() needs at least one value")
    counts = Counter(values)
    top = max(counts.values())
    return sorted(v for v, c in counts.items() if c == top)


def variance(values: list[float], sample: bool = True) -> float:
    """Sample variance divides by n - 1 (Bessel's correction); population by n."""
    n = len(values)
    if n < 2 and sample:
        raise ValueError("sample variance needs at least two values")
    m = mean(values)
    squared_diffs = sum((x - m) ** 2 for x in values)
    return squared_diffs / (n - 1 if sample else n)


def std(values: list[float], sample: bool = True) -> float:
    return math.sqrt(variance(values, sample=sample))


def min_max_scale(values: list[float]) -> list[float]:
    """Rescale to [0, 1]. A constant column becomes all zeros."""
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0 for _ in values]
    return [(x - lo) / (hi - lo) for x in values]


def standardize(values: list[float]) -> list[float]:
    """z = (x - mean) / std, using the population std like scikit-learn's StandardScaler."""
    m = mean(values)
    s = std(values, sample=False)
    if s == 0:
        return [0.0 for _ in values]
    return [(x - m) / s for x in values]


def one_hot_encode(labels: list[str]) -> tuple[list[str], list[list[int]]]:
    """Return (categories, rows), with categories sorted for a stable column order."""
    categories = sorted(set(labels))
    index = {cat: i for i, cat in enumerate(categories)}
    rows = []
    for label in labels:
        row = [0] * len(categories)
        row[index[label]] = 1
        rows.append(row)
    return categories, rows


def train_test_split(rows: list, test_size: float = 0.2, seed: int = 42) -> tuple[list, list]:
    """Shuffle a copy with a fixed seed, then cut. The input list is not changed."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    shuffled = rows.copy()
    random.Random(seed).shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * test_size))
    return shuffled[n_test:], shuffled[:n_test]


def accuracy(y_true: list, y_pred: list) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("accuracy() needs at least one prediction")
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true)


def word_counts(text: str, top_k: int | None = None) -> list[tuple[str, int]]:
    """Lower-case, strip punctuation, count words; most common first."""
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    counts = Counter(cleaned.split())
    return counts.most_common(top_k)


if __name__ == "__main__":
    petal_lengths = [1.4, 1.4, 1.3, 4.7, 4.5, 6.0, 5.1, 1.5]
    species = ["setosa", "setosa", "setosa", "versicolor",
               "versicolor", "virginica", "virginica", "setosa"]

    print("mean:   ", round(mean(petal_lengths), 3))
    print("median: ", median(petal_lengths))
    print("mode:   ", mode(petal_lengths))
    print("std:    ", round(std(petal_lengths), 3))
    print("scaled: ", [round(x, 2) for x in min_max_scale(petal_lengths)])
    print("z:      ", [round(x, 2) for x in standardize(petal_lengths)])

    categories, encoded = one_hot_encode(species)
    print("one-hot columns:", categories)
    print("first row:      ", encoded[0])

    rows = list(zip(petal_lengths, species))
    train, test = train_test_split(rows, test_size=0.25)
    print(f"train={len(train)} rows, test={len(test)} rows")

    # A one-rule "model": petal length below 2.5 means setosa
    y_true = [label == "setosa" for _, label in test]
    y_pred = [length < 2.5 for length, _ in test]
    print("rule accuracy on test:", accuracy(y_true, y_pred))

    print("word counts:", word_counts("The cat sat. The cat ran! A dog sat.", top_k=3))
