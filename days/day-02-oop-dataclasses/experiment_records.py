"""Day 02 — dataclasses, enums, protocols and type hints for ML experiment code.

Run:  python days/day-02-oop-dataclasses/experiment_records.py

What it shows:
  * @dataclass to remove __init__/__repr__/__eq__ boilerplate
  * frozen dataclasses for configs that must not change mid-run
  * __post_init__ validation, field(default_factory=...), dataclasses.replace
  * order=True with a hidden sort key, to rank experiment results
  * Enum for a closed set of choices instead of magic strings
  * typing.Protocol: "duck typing with a type checker" (no inheritance needed)
  * class vs instance attributes, @classmethod constructors, @staticmethod helpers
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from enum import Enum
from typing import Iterator, Protocol, Sequence, runtime_checkable


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


class Task(str, Enum):
    """A closed set of options. Inheriting from str makes it JSON-friendly."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for one run. Frozen: assigning to a field raises an error.

    Immutability matters because a config that changes during a run makes
    logged results impossible to reproduce.
    """

    model: str
    task: Task = Task.CLASSIFICATION
    learning_rate: float = 0.01
    epochs: int = 10
    batch_size: int = 32
    seed: int = 42
    tags: tuple[str, ...] = ()       # tuple, not list: frozen should mean fully immutable

    def __post_init__(self) -> None:
        # Runs after the generated __init__; the place for validation.
        if not 0 < self.learning_rate <= 1:
            raise ValueError(f"learning_rate must be in (0, 1], got {self.learning_rate}")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be >= 1")
        if not isinstance(self.task, Task):
            # Accept "regression" from a JSON/YAML file and convert it.
            # object.__setattr__ is the documented way to set a field on a frozen instance.
            object.__setattr__(self, "task", Task(self.task))

    @classmethod
    def from_dict(cls, raw: dict) -> TrainingConfig:
        """Alternative constructor. Unknown keys are an error, not silently ignored:
        a typo like 'learning_rte' would otherwise run with the default value."""
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise KeyError(f"unknown config keys: {sorted(unknown)}")
        data = dict(raw)
        if "tags" in data:
            data["tags"] = tuple(data["tags"])
        return cls(**data)


@dataclass(order=True)
class ExperimentResult:
    """One finished run. order=True compares fields in order, so the first
    field is a hidden sort key: sorting results ranks them by score."""

    sort_index: float = field(init=False, repr=False, compare=True)
    run_id: str = field(compare=False)
    score: float = field(compare=False)
    metrics: dict[str, float] = field(default_factory=dict, compare=False)
    higher_is_better: bool = field(default=True, compare=False)

    def __post_init__(self) -> None:
        # Negate when lower is better (e.g. RMSE) so sorting still puts the best last.
        self.sort_index = self.score if self.higher_is_better else -self.score


@runtime_checkable
class SupportsPredict(Protocol):
    """Anything with predict(X) -> list. No base class required."""

    def predict(self, X: Sequence[Sequence[float]]) -> list: ...


class ThresholdRule:
    """Does not inherit from anything, yet satisfies SupportsPredict."""

    def __init__(self, feature: int, threshold: float) -> None:
        self.feature = feature
        self.threshold = threshold

    def predict(self, X: Sequence[Sequence[float]]) -> list[int]:
        return [int(row[self.feature] > self.threshold) for row in X]


def evaluate(model: SupportsPredict, X: Sequence[Sequence[float]], y: Sequence) -> float:
    """Accuracy of any object that has a predict method."""
    preds = model.predict(X)
    return sum(p == t for p, t in zip(preds, y)) / len(y)


class Leaderboard:
    """Composition: a Leaderboard *has* a list of results and adds behaviour on top."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._results: list[ExperimentResult] = []   # instance attribute: one list per object

    def add(self, result: ExperimentResult) -> None:
        if result.run_id in self:
            raise ValueError(f"duplicate run_id {result.run_id!r}")
        self._results.append(result)

    def best(self) -> ExperimentResult:
        if not self._results:
            raise ValueError("leaderboard is empty")
        return max(self._results)

    def ranked(self) -> list[ExperimentResult]:
        return sorted(self._results, reverse=True)

    @staticmethod
    def format_row(rank: int, r: ExperimentResult) -> str:
        """A helper that needs neither self nor cls: a staticmethod."""
        return f"{rank:>2}. {r.run_id:<14} score={r.score:.3f}"

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self) -> Iterator[ExperimentResult]:
        return iter(self._results)

    def __contains__(self, run_id: object) -> bool:
        return any(r.run_id == run_id for r in self._results)


class SharedListPitfall:
    """Class attribute that is mutable: every instance shares the same list."""

    history: list[str] = []          # deliberately wrong, for the demo

    def log(self, msg: str) -> None:
        self.history.append(msg)


def main() -> None:
    section("1. Frozen config with validation")
    cfg = TrainingConfig(model="logreg", learning_rate=0.1, tags=("baseline",))
    print(cfg)
    try:
        cfg.epochs = 100  # type: ignore[misc]
    except AttributeError as err:            # FrozenInstanceError subclasses AttributeError
        print(f"{type(err).__name__} -> {err}")
    try:
        TrainingConfig(model="logreg", learning_rate=5)
    except ValueError as err:
        print("ValueError ->", err)
    longer = replace(cfg, epochs=50)          # a modified copy; the original is untouched
    print("replace(cfg, epochs=50) ->", longer.epochs, "| original:", cfg.epochs)
    print("asdict:", asdict(cfg))
    print("equal by value:", cfg == TrainingConfig(model="logreg", learning_rate=0.1,
                                                     tags=("baseline",)))
    print("hashable, so usable as a dict key:", {cfg: "run-1"}[cfg])

    raw = {"model": "ridge", "task": "regression", "epochs": 5}
    print("from_dict:", TrainingConfig.from_dict(raw))
    try:
        TrainingConfig.from_dict({"model": "ridge", "learning_rte": 0.1})
    except KeyError as err:
        print("KeyError ->", err)

    section("2. Ranking results with order=True")
    board = Leaderboard("wine")
    # Illustrative scores, not real runs: the point is the ranking behaviour.
    for run_id, score in [("knn-k3", 0.944), ("majority", 0.407),
                          ("knn-k7", 0.963), ("logreg", 0.981)]:
        board.add(ExperimentResult(run_id=run_id, score=score, metrics={"acc": score}))
    for rank, r in enumerate(board.ranked(), start=1):
        print(Leaderboard.format_row(rank, r))
    print("best:", board.best().run_id, "| 'knn-k3' in board:", "knn-k3" in board,
          "| len:", len(board))

    rmse_runs = [ExperimentResult("a", 3.2, higher_is_better=False),
                 ExperimentResult("b", 2.1, higher_is_better=False)]
    print("lower-is-better best:", max(rmse_runs).run_id, "(RMSE 2.1)")

    section("3. Protocols: structural typing")
    rule = ThresholdRule(feature=0, threshold=0.5)
    X = [[0.1], [0.7], [0.9], [0.2]]
    y = [0, 1, 1, 1]
    print("ThresholdRule is a SupportsPredict:", isinstance(rule, SupportsPredict))
    print("a plain dict is a SupportsPredict:", isinstance({}, SupportsPredict))
    print("accuracy of the rule:", evaluate(rule, X, y))

    section("4. Pitfall: mutable class attribute")
    a, b = SharedListPitfall(), SharedListPitfall()
    a.log("from a")
    print("b.history (never logged to b!):", b.history)
    print("Fix: create the list in __init__ (self.history = []), as Leaderboard does.")


if __name__ == "__main__":
    main()
