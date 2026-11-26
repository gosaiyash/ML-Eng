# Day 02 — OOP: classes, inheritance, composition, dataclasses, type hints

**Handbook:** §01 Python Programming and Software Engineering · **Phase 1 — Foundations**

## Goal

Learn object-oriented Python by building something you will use every day as an ML engineer:
a tiny scikit-learn-style library. Abstract base classes define the `fit` / `predict` /
`transform` interface, inheritance shares `score`, `repr` and "is it fitted?" checks,
and a `Pipeline` combines a scaler and a model through composition. Then use dataclasses,
enums and protocols to write experiment configs and results that are typed, validated
and hard to misuse.

## Files

| File | What it shows |
|------|---------------|
| [`mini_estimators.py`](mini_estimators.py) | `BaseEstimator` → `Transformer` / `Classifier` / `Regressor` hierarchy, `StandardScaler`, `MinMaxScaler`, `MajorityClassifier`, `MeanRegressor`, `KNNClassifier` (property with validation), `Pipeline` (composition, `__len__`, `__getitem__`), custom `NotFittedError`. The demo compares a baseline, raw KNN and scaled KNN on the wine dataset |
| [`experiment_records.py`](experiment_records.py) | Frozen `TrainingConfig` with `__post_init__` validation and a `from_dict` classmethod, `ExperimentResult` ranked with `order=True`, a `Task` enum, a `SupportsPredict` protocol, a `Leaderboard` class, and the shared-mutable-class-attribute pitfall |
| [`test_day02.py`](test_day02.py) | 22 pytest tests (28 cases): scalers, KNN and the majority baseline checked against scikit-learn, R² against `r2_score`, plus errors, pipeline behaviour and dataclass rules |

## Run it

```bash
python days/day-02-oop-dataclasses/mini_estimators.py
python days/day-02-oop-dataclasses/experiment_records.py
pytest days/day-02-oop-dataclasses
```

## Notes

**The four ideas, in ML terms**

| Idea | Meaning | Example here |
|------|---------|--------------|
| Encapsulation | an object keeps its data and the rules for it together | `KNNClassifier.k` is validated by a property setter |
| Abstraction | callers use an interface, not the details | every model has `fit` / `predict` / `score` |
| Inheritance ("is a") | a subclass reuses and extends a parent | `KNNClassifier` is a `Classifier`, so it gets accuracy-based `score` for free |
| Polymorphism | the same call works on different types | `Pipeline` calls `fit_transform` on any `Transformer` |

**Inheritance vs composition.** Use inheritance only for a real "is a" relationship with
shared behaviour, and keep hierarchies shallow. Use composition ("has a") for everything else:
a `Pipeline` *has* a scaler and a model; making it inherit from both would be a mess.
When unsure, prefer composition — it is easier to change and to test.

**Things that trip people up**

- Forgetting `super().__init__()` in a subclass skips the parent's setup.
- A mutable class attribute (`history = []` in the class body) is shared by all instances. Create it in `__init__`.
- A dataclass field default of `[]` or `{}` is rejected; use `field(default_factory=list)`.
- `frozen=True` makes a dataclass immutable and hashable, but a `list` inside it can still be mutated — use tuples.
- Abstract methods only block instantiation; they do not check argument types. Type hints are not enforced at runtime either — validate inputs that come from users or files.
- Learned state ends with `_` (`mean_`, `classes_`) by scikit-learn convention, so you can tell hyperparameters from what `fit` learned.

**When to use what**

- `@dataclass` — plain records: configs, results, metadata. `frozen=True` for anything that should not change during a run.
- `Enum` — a fixed set of choices (`Task.REGRESSION`) instead of strings that can be misspelled.
- `ABC` + `@abstractmethod` — you own the hierarchy and want to force subclasses to implement methods.
- `Protocol` — you only care that an object *has* certain methods (duck typing that a type checker understands). Third-party models can satisfy it without inheriting anything.
- `@property` — computed or validated attributes; `@classmethod` — alternative constructors (`from_dict`); `@staticmethod` — a helper that needs neither `self` nor `cls`.

## Test yourself

<details>
<summary>1. When would you choose composition over inheritance?</summary>

When the relationship is "has a" rather than "is a", or when you would need to inherit from several unrelated classes. A training pipeline has a preprocessor and a model; it is neither. Composition keeps classes small and lets you swap parts (a different scaler) without touching a hierarchy.
</details>

<details>
<summary>2. What does an abstract base class give you that a normal base class does not?</summary>

A class with unimplemented `@abstractmethod`s cannot be instantiated, and neither can any subclass that forgets to implement them. The error appears when the object is created, not later when a missing method is finally called during training.
</details>

<details>
<summary>3. Why make a config dataclass frozen?</summary>

So no code can change hyperparameters in the middle of a run, which would make the logged config differ from what actually trained the model. Frozen dataclasses are also hashable, so they can be dictionary keys or cache keys; use `dataclasses.replace` to make a modified copy.
</details>

<details>
<summary>4. Why does <code>@dataclass</code> refuse <code>metrics: dict = {}</code>?</summary>

The default would be one dict object shared by every instance, the same bug as a mutable default argument. `field(default_factory=dict)` calls `dict()` for each new instance, so each gets its own.
</details>

<details>
<summary>5. What is the difference between an ABC and a Protocol?</summary>

An ABC uses nominal typing: a class must explicitly inherit from it. A Protocol uses structural typing: any class with matching methods counts, which a type checker verifies (and `isinstance` can check method names with `@runtime_checkable`). Protocols suit code that must accept objects you did not write, such as models from other libraries.
</details>

<details>
<summary>6. What is the difference between <code>@classmethod</code> and <code>@staticmethod</code>?</summary>

A classmethod receives the class as `cls`, so it can construct instances and works correctly for subclasses — the usual pattern for alternative constructors like `from_dict` or `from_yaml`. A staticmethod receives nothing implicit; it is just a function kept in the class namespace because it belongs there logically.
</details>

## Next

Day 03 — exceptions, custom errors, CSV/JSON files, and context managers.
