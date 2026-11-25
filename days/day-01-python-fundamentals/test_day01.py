"""Tests for Day 01. Run: pytest days/day-01-python-fundamentals"""

import math
import statistics

import pytest

from pure_python_stats import (
    accuracy,
    mean,
    median,
    min_max_scale,
    mode,
    one_hot_encode,
    standardize,
    std,
    train_test_split,
    variance,
    word_counts,
)

DATA = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]


def test_mean_and_median_match_standard_library():
    assert mean(DATA) == pytest.approx(statistics.mean(DATA))
    assert median(DATA) == pytest.approx(statistics.median(DATA))
    assert median([3, 1, 2]) == 2


def test_mode_returns_every_tie():
    assert mode([1, 1, 2, 2, 3]) == [1, 2]
    assert mode(["a", "b", "a"]) == ["a"]


def test_variance_sample_vs_population():
    assert variance(DATA, sample=True) == pytest.approx(statistics.variance(DATA))
    assert variance(DATA, sample=False) == pytest.approx(statistics.pvariance(DATA))
    assert std(DATA, sample=False) == pytest.approx(2.0)


@pytest.mark.parametrize("func", [mean, median, mode])
def test_empty_input_raises(func):
    with pytest.raises(ValueError):
        func([])


def test_min_max_scale_range_and_constant_column():
    scaled = min_max_scale(DATA)
    assert min(scaled) == 0.0 and max(scaled) == 1.0
    assert min_max_scale([3.0, 3.0]) == [0.0, 0.0]


def test_standardize_has_zero_mean_unit_std():
    z = standardize(DATA)
    assert mean(z) == pytest.approx(0.0, abs=1e-12)
    assert std(z, sample=False) == pytest.approx(1.0)


def test_one_hot_encode():
    categories, rows = one_hot_encode(["dog", "cat", "dog"])
    assert categories == ["cat", "dog"]
    assert rows == [[0, 1], [1, 0], [0, 1]]
    assert all(sum(r) == 1 for r in rows)


def test_train_test_split_is_reproducible_and_complete():
    rows = list(range(100))
    train_a, test_a = train_test_split(rows, test_size=0.2, seed=7)
    train_b, test_b = train_test_split(rows, test_size=0.2, seed=7)

    assert (train_a, test_a) == (train_b, test_b)       # same seed, same split
    assert len(test_a) == 20 and len(train_a) == 80
    assert set(train_a).isdisjoint(test_a)              # no row in both
    assert sorted(train_a + test_a) == rows             # nothing lost
    assert rows == list(range(100))                     # input untouched


def test_train_test_split_rejects_bad_size():
    with pytest.raises(ValueError):
        train_test_split([1, 2, 3], test_size=1.5)


def test_accuracy():
    assert accuracy([1, 0, 1, 1], [1, 0, 0, 1]) == 0.75
    with pytest.raises(ValueError):
        accuracy([1, 0], [1])


def test_word_counts_ignores_case_and_punctuation():
    counts = word_counts("The cat. THE dog, the end!")
    assert counts[0] == ("the", 3)
    assert dict(counts)["cat"] == 1
    assert math.isclose(sum(c for _, c in counts), 6)
