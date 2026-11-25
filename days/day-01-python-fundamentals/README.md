# Day 01 — Python fundamentals

**Handbook:** §01 Python Programming and Software Engineering · **Phase 1 — Foundations**

## Goal

Get comfortable with the core language before any ML library: data types, the four built-in
collections, control flow, functions, and comprehensions. Then use only plain Python to write
the small helpers that NumPy and scikit-learn will do for you later (mean, std, scaling,
one-hot encoding, a train/test split, accuracy).

## Files

| File | What it shows |
|------|---------------|
| [`fundamentals.py`](fundamentals.py) | Six short sections: types and operators, collections, control flow, functions and scope, comprehensions, list-vs-set performance |
| [`pure_python_stats.py`](pure_python_stats.py) | ML helpers in plain Python: `mean`, `median`, `mode`, `variance`, `std`, `min_max_scale`, `standardize`, `one_hot_encode`, `train_test_split`, `accuracy`, `word_counts` |
| [`test_day01.py`](test_day01.py) | 13 pytest tests, several checked against Python's `statistics` module |

## Run it

```bash
python days/day-01-python-fundamentals/fundamentals.py
python days/day-01-python-fundamentals/pure_python_stats.py
pytest days/day-01-python-fundamentals
```

## Notes

**Choosing a collection**

| Type | Ordered | Mutable | Duplicates | Use it for |
|------|:-------:|:-------:|:----------:|------------|
| `list` | yes | yes | yes | sequences you build up: losses per epoch, rows |
| `tuple` | yes | no | yes | fixed records: a shape `(150, 4)`, dict keys |
| `set` | no | yes | no | unique values, fast `in` checks, overlaps |
| `dict` | insertion order | yes | keys unique | lookups, configs, counts |

**Things that trip people up**

- `0.1 + 0.2 != 0.3`. Floats are approximate, so compare with a tolerance (`math.isclose`, `pytest.approx`).
- `b = a` does not copy a list. Both names point to the same object; use `a.copy()`.
- A mutable default argument (`def f(x, bucket=[])`) is created once and shared between calls. Use `None` and create the list inside.
- `x in some_list` scans the whole list (O(n)); `x in some_set` is a hash lookup (O(1) on average). The demo shows the difference.
- Sample variance divides by `n - 1`, population variance by `n`. scikit-learn's `StandardScaler` uses the population version.
- A train/test split should be shuffled with a fixed seed so results are reproducible, and it must not change the original data.

**Comprehensions** replace a loop that builds a collection:
`[f(x) for x in xs if cond]`, `{k: v for ...}`, `{x for ...}`. Use a generator expression
(`sum(x * x for x in xs)`) when you only need to consume the values once — it does not build a list in memory.

## Test yourself

<details>
<summary>1. When would you use a tuple instead of a list?</summary>

When the data is a fixed record that should not change (a shape, coordinates, a database row), or when you need it as a dictionary key or set member — only immutable (hashable) objects can be keys.
</details>

<details>
<summary>2. Why is <code>x in my_set</code> faster than <code>x in my_list</code>?</summary>

A set stores items in a hash table, so membership is one hash lookup (O(1) on average). A list has to compare against each item in turn (O(n)).
</details>

<details>
<summary>3. What is wrong with <code>def add(item, items=[])</code>?</summary>

The default list is created once, when the function is defined, so every call that uses the default appends to the same list. Use `items=None` and create a new list inside.
</details>

<details>
<summary>4. What is the difference between sample and population variance?</summary>

Population variance divides by `n`. Sample variance divides by `n - 1` (Bessel's correction) because the sample mean sits closer to the sample's own points than the true mean does, which would otherwise make the variance estimate too small.
</details>

<details>
<summary>5. Why fix a random seed when splitting data?</summary>

So the split is the same every run. Without it, model scores change between runs for reasons unrelated to the model, and nobody can reproduce your results.
</details>

<details>
<summary>6. Why min-max scaling or standardizing at all?</summary>

Many models (KNN, SVM, linear models trained with gradient descent, neural networks) are sensitive to feature scale: a feature measured in thousands would dominate one measured in fractions. Scaling puts features on comparable ranges.
</details>

## Next

Day 02 — object-oriented programming: classes, inheritance, composition, dataclasses, and type hints.
