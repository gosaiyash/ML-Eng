"""Day 02 — OOP by building a tiny scikit-learn-style library in plain Python.

Run:  python days/day-02-oop-dataclasses/mini_estimators.py

What it shows:
  * classes, __init__, instance vs class attributes, properties
  * abstract base classes (an interface every model must follow)
  * inheritance: shared behaviour (score, repr, fitted checks) lives in the base class
  * the template-method pattern: fit_transform() is written once and reused
  * composition: a Pipeline *has* transformers and a model, it is not one of them
  * custom exceptions and dunder methods (__repr__, __len__, __getitem__)

Data is a list of rows (list[list[float]]) so nothing is hidden behind NumPy.
The functions are tested against scikit-learn in test_day02.py.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Sequence

Matrix = list[list[float]]
Vector = list[float]


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class NotFittedError(RuntimeError):
    """Raised when predict/transform is called before fit.

    Subclassing a built-in exception means callers can catch it either
    specifically (NotFittedError) or broadly (RuntimeError).
    """


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------
class BaseEstimator(ABC):
    """Behaviour shared by every estimator.

    Convention borrowed from scikit-learn: constructor arguments are
    hyperparameters, and anything learned in fit() ends with an underscore
    (mean_, classes_). That convention lets the base class find both
    without each subclass repeating code.
    """

    # Class attribute: one value shared by all instances of all subclasses.
    _instances_created: int = 0

    def __init__(self) -> None:
        BaseEstimator._instances_created += 1

    def get_params(self) -> dict[str, Any]:
        """Hyperparameters = public attributes that do not end with '_'."""
        return {k: v for k, v in vars(self).items()
                if not k.startswith("_") and not k.endswith("_")}

    @property
    def is_fitted(self) -> bool:
        """A property is computed on access but read like an attribute."""
        return any(k.endswith("_") and not k.startswith("_") for k in vars(self))

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise NotFittedError(
                f"{type(self).__name__} is not fitted yet; call fit() first")

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({params})"

    @abstractmethod
    def fit(self, X: Matrix, y: Sequence | None = None) -> BaseEstimator:
        """Learn from data and return self (so calls can be chained)."""


class Transformer(BaseEstimator):
    """Something that learns a mapping in fit() and applies it in transform()."""

    @abstractmethod
    def transform(self, X: Matrix) -> Matrix:
        ...

    def fit_transform(self, X: Matrix, y: Sequence | None = None) -> Matrix:
        # Template method: written once here, works for every subclass
        # because it only relies on the abstract fit() and transform().
        return self.fit(X, y).transform(X)


class Predictor(BaseEstimator):
    @abstractmethod
    def predict(self, X: Matrix) -> list:
        ...

    @abstractmethod
    def score(self, X: Matrix, y: Sequence) -> float:
        ...


class Classifier(Predictor):
    def score(self, X: Matrix, y: Sequence) -> float:
        """Accuracy, the default classification score (like scikit-learn)."""
        preds = self.predict(X)
        return sum(p == t for p, t in zip(preds, y)) / len(y)


class Regressor(Predictor):
    def score(self, X: Matrix, y: Sequence[float]) -> float:
        """R^2 = 1 - SS_res / SS_tot, the default regression score."""
        preds = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((t - p) ** 2 for p, t in zip(preds, y))
        ss_tot = sum((t - y_mean) ** 2 for t in y)
        return 1.0 - ss_res / ss_tot if ss_tot else 0.0


def _columns(X: Matrix) -> list[list[float]]:
    if not X:
        raise ValueError("X must contain at least one row")
    return [list(col) for col in zip(*X)]


# ---------------------------------------------------------------------------
# Transformers
# ---------------------------------------------------------------------------
class StandardScaler(Transformer):
    """z = (x - mean) / std per column, population std (like scikit-learn)."""

    def fit(self, X: Matrix, y: Sequence | None = None) -> StandardScaler:
        cols = _columns(X)
        self.mean_ = [sum(c) / len(c) for c in cols]
        stds = [math.sqrt(sum((v - m) ** 2 for v in c) / len(c))
                for c, m in zip(cols, self.mean_)]
        # A constant column has std 0; dividing by 1 leaves it at 0 instead of NaN.
        self.scale_ = [s if s > 0 else 1.0 for s in stds]
        return self

    def transform(self, X: Matrix) -> Matrix:
        self._check_fitted()
        return [[(v - m) / s for v, m, s in zip(row, self.mean_, self.scale_)]
                for row in X]


class MinMaxScaler(Transformer):
    """Rescale each column to [0, 1] using the training min and max."""

    def fit(self, X: Matrix, y: Sequence | None = None) -> MinMaxScaler:
        cols = _columns(X)
        self.min_ = [min(c) for c in cols]
        ranges = [max(c) - lo for c, lo in zip(cols, self.min_)]
        self.range_ = [r if r > 0 else 1.0 for r in ranges]
        return self

    def transform(self, X: Matrix) -> Matrix:
        self._check_fitted()
        return [[(v - lo) / r for v, lo, r in zip(row, self.min_, self.range_)]
                for row in X]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class MajorityClassifier(Classifier):
    """Baseline: always predict the most common training label."""

    def fit(self, X: Matrix, y: Sequence | None = None) -> MajorityClassifier:
        if not y:
            raise ValueError("y must not be empty")
        counts = Counter(y)
        top = max(counts.values())
        # Ties go to the smallest label, matching scikit-learn's DummyClassifier.
        self.majority_ = min(label for label, c in counts.items() if c == top)
        return self

    def predict(self, X: Matrix) -> list:
        self._check_fitted()
        return [self.majority_ for _ in X]


class MeanRegressor(Regressor):
    """Baseline: always predict the training mean. Its R^2 on train is 0."""

    def fit(self, X: Matrix, y: Sequence[float] | None = None) -> MeanRegressor:
        if not y:
            raise ValueError("y must not be empty")
        self.mean_ = sum(y) / len(y)
        return self

    def predict(self, X: Matrix) -> list[float]:
        self._check_fitted()
        return [self.mean_ for _ in X]


class KNNClassifier(Classifier):
    """k-nearest neighbours with Euclidean distance and a majority vote."""

    def __init__(self, k: int = 5) -> None:
        super().__init__()          # keep the base-class bookkeeping
        self.k = k                  # goes through the property setter below

    @property
    def k(self) -> int:
        return self._k

    @k.setter
    def k(self, value: int) -> None:
        # Validation in one place: the constructor and later assignments both use it.
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"k must be a positive int, got {value!r}")
        self._k = value

    def get_params(self) -> dict[str, Any]:
        return {"k": self.k}        # _k is private, so expose it explicitly

    def fit(self, X: Matrix, y: Sequence | None = None) -> KNNClassifier:
        if y is None or len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if self.k > len(X):
            raise ValueError(f"k={self.k} is larger than the {len(X)} training rows")
        # KNN is "lazy": fitting just stores the data.
        self.X_ = [list(row) for row in X]
        self.y_ = list(y)
        return self

    def _predict_one(self, row: Vector) -> Any:
        dists = sorted(
            (math.dist(row, train_row), i) for i, train_row in enumerate(self.X_))
        votes = Counter(self.y_[i] for _, i in dists[: self.k])
        top = max(votes.values())
        return min(label for label, c in votes.items() if c == top)

    def predict(self, X: Matrix) -> list:
        self._check_fitted()
        return [self._predict_one(row) for row in X]


# ---------------------------------------------------------------------------
# Composition: a Pipeline has steps; it is not a scaler or a model itself
# ---------------------------------------------------------------------------
class Pipeline:
    """Chain transformers and end with a predictor.

    fit() fits each transformer on the output of the previous one, so the
    test data later goes through transformations learned only on train data.
    That is exactly what prevents leakage from scaling.
    """

    def __init__(self, steps: list[tuple[str, BaseEstimator]]) -> None:
        if not steps:
            raise ValueError("a Pipeline needs at least one step")
        names = [name for name, _ in steps]
        if len(set(names)) != len(names):
            raise ValueError(f"step names must be unique: {names}")
        *transformers, (_, final) = steps
        for name, step in transformers:
            if not isinstance(step, Transformer):
                raise TypeError(f"step {name!r} must be a Transformer")
        if not isinstance(final, Predictor):
            raise TypeError("the last step must be a Predictor")
        self.steps = steps

    def _transform(self, X: Matrix, fit: bool) -> Matrix:
        for _, step in self.steps[:-1]:
            X = step.fit_transform(X) if fit else step.transform(X)
        return X

    def fit(self, X: Matrix, y: Sequence) -> Pipeline:
        self.steps[-1][1].fit(self._transform(X, fit=True), y)
        return self

    def predict(self, X: Matrix) -> list:
        return self.steps[-1][1].predict(self._transform(X, fit=False))

    def score(self, X: Matrix, y: Sequence) -> float:
        return self.steps[-1][1].score(self._transform(X, fit=False), y)

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, name: str) -> BaseEstimator:
        for step_name, step in self.steps:
            if step_name == name:
                return step
        raise KeyError(name)

    def __repr__(self) -> str:
        inner = ", ".join(f"({n!r}, {s!r})" for n, s in self.steps)
        return f"Pipeline([{inner}])"


def split_rows(X: Matrix, y: list, test_size: float = 0.3,
               seed: int = 0) -> tuple[Matrix, Matrix, list, list]:
    """Shuffle indices with a fixed seed and split X and y the same way."""
    idx = list(range(len(X)))
    random.Random(seed).shuffle(idx)
    n_test = max(1, int(len(X) * test_size))
    test, train = idx[:n_test], idx[n_test:]
    return ([X[i] for i in train], [X[i] for i in test],
            [y[i] for i in train], [y[i] for i in test])


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def main() -> None:
    from sklearn.datasets import load_wine

    data = load_wine()
    X, y = data.data.tolist(), data.target.tolist()
    X_tr, X_te, y_tr, y_te = split_rows(X, y, test_size=0.3, seed=0)
    print(f"wine: {len(X)} rows, {len(X[0])} features, classes={sorted(set(y))}")
    print(f"train={len(X_tr)}  test={len(X_te)}")

    section("1. Inheritance and the class hierarchy")
    print("KNNClassifier MRO:", " -> ".join(c.__name__ for c in KNNClassifier.__mro__))
    knn = KNNClassifier(k=5)
    print("repr:", knn)
    print("isinstance(knn, Classifier):", isinstance(knn, Classifier))
    print("isinstance(knn, Transformer):", isinstance(knn, Transformer))
    try:
        BaseEstimator()  # type: ignore[abstract]
    except TypeError as err:
        print("Cannot instantiate an abstract class ->", err)

    section("2. Properties, validation and custom errors")
    print("is_fitted before fit:", knn.is_fitted)
    try:
        knn.predict(X_te[:1])
    except NotFittedError as err:
        print("NotFittedError ->", err)
    try:
        knn.k = 0
    except ValueError as err:
        print("ValueError from the k setter ->", err)

    section("3. Baseline vs raw KNN vs scaled KNN (composition)")
    baseline = MajorityClassifier().fit(X_tr, y_tr)
    raw_knn = KNNClassifier(k=5).fit(X_tr, y_tr)
    pipe = Pipeline([("scale", StandardScaler()), ("knn", KNNClassifier(k=5))])
    pipe.fit(X_tr, y_tr)
    print(f"majority baseline accuracy: {baseline.score(X_te, y_te):.3f}")
    print(f"KNN on raw features:        {raw_knn.score(X_te, y_te):.3f}")
    print(f"scaler + KNN pipeline:      {pipe.score(X_te, y_te):.3f}")
    print("Why the jump: 'proline' is in the hundreds and dominates raw distances.")
    print(pipe)
    scaler = pipe["scale"]
    assert isinstance(scaler, StandardScaler)
    print("scaler learned means (first 3):", [round(m, 2) for m in scaler.mean_[:3]])

    section("4. Class attribute shared by every instance")
    print("estimators created so far:", BaseEstimator._instances_created)

    section("5. Regression baseline")
    from sklearn.datasets import load_diabetes
    d = load_diabetes()
    Xd, yd = d.data.tolist(), d.target.tolist()
    Xd_tr, Xd_te, yd_tr, yd_te = split_rows(Xd, yd, seed=0)
    mean_model = MeanRegressor().fit(Xd_tr, yd_tr)
    print(f"MeanRegressor R^2 train={mean_model.score(Xd_tr, yd_tr):.3f}  "
          f"test={mean_model.score(Xd_te, yd_te):.3f}  (any real model must beat this)")


if __name__ == "__main__":
    main()
