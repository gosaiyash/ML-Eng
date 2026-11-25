"""Day 01 — Python fundamentals, with an ML flavour.

Run:  python days/day-01-python-fundamentals/fundamentals.py

Each section prints what it demonstrates. Read the code alongside the output.
"""

import timeit


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# ---------------------------------------------------------------------------
# 1. Data types and operators
# ---------------------------------------------------------------------------
def data_types_and_operators() -> None:
    section("1. Data types and operators")

    n_samples = 150            # int
    learning_rate = 0.01       # float
    model_name = "baseline"    # str
    is_trained = False         # bool
    best_score = None          # NoneType — "no value yet"

    for value in (n_samples, learning_rate, model_name, is_trained, best_score):
        print(f"{value!r:>12} -> {type(value).__name__}")

    # Division: / always gives a float, // floors, % gives the remainder
    print("\n7 / 2  =", 7 / 2)
    print("7 // 2 =", 7 // 2)
    print("7 % 2  =", 7 % 2)
    print("2 ** 10 =", 2 ** 10)

    # Floats are not exact — compare with a tolerance, never with ==
    print("\n0.1 + 0.2 == 0.3 ?", 0.1 + 0.2 == 0.3)
    print("abs((0.1 + 0.2) - 0.3) < 1e-9 ?", abs((0.1 + 0.2) - 0.3) < 1e-9)

    # Type conversion: data from files arrives as strings
    raw_row = "5.1,3.5,1.4,0.2,setosa"
    *features_str, label = raw_row.split(",")
    features = [float(x) for x in features_str]
    print(f"\nParsed row -> features={features}, label={label!r}")

    # Truthiness: empty containers, 0 and None are falsy
    for value in ([], {}, "", 0, None, [0], "0"):
        print(f"bool({value!r:>4}) = {bool(value)}")


# ---------------------------------------------------------------------------
# 2. Collections: list, tuple, set, dict
# ---------------------------------------------------------------------------
def collections_demo() -> None:
    section("2. Collections: list, tuple, set, dict")

    # list — ordered, mutable. Use for sequences you append to.
    losses = [0.92, 0.71, 0.55, 0.48]
    losses.append(0.44)
    print("losses:", losses, "| last two:", losses[-2:], "| reversed:", losses[::-1])

    # tuple — ordered, immutable. Use for fixed records, and as dict keys.
    shape = (150, 4)
    n_rows, n_cols = shape  # unpacking
    print(f"shape={shape} -> rows={n_rows}, cols={n_cols}")

    # set — unique items, fast membership checks.
    labels = ["cat", "dog", "cat", "bird", "dog", "cat"]
    classes = set(labels)
    print("unique classes:", sorted(classes))
    print("train/test overlap:", {"a", "b", "c"} & {"c", "d"})

    # dict — key -> value. Use for lookups, configs, counts.
    config = {"model": "logistic_regression", "C": 1.0, "max_iter": 200}
    config["C"] = 0.5
    print("config:", config)
    print("missing key with default:", config.get("penalty", "l2"))
    for key, value in config.items():
        print(f"  {key:>10}: {value}")

    # Gotcha: assignment does not copy a list — both names point to one object
    a = [1, 2, 3]
    b = a
    c = a.copy()
    b.append(4)
    print(f"\na={a}, b={b} (same object), c={c} (a copy)")


# ---------------------------------------------------------------------------
# 3. Control flow
# ---------------------------------------------------------------------------
def control_flow() -> None:
    section("3. Control flow")

    scores = {"logreg": 0.81, "tree": 0.74, "forest": 0.88, "dummy": 0.50}

    for name, score in scores.items():
        if score >= 0.85:
            verdict = "strong"
        elif score >= 0.75:
            verdict = "ok"
        else:
            verdict = "weak"
        print(f"{name:>7}: {score:.2f} ({verdict})")

    # enumerate gives the index, zip walks lists in step
    epochs_loss = [0.9, 0.6, 0.45, 0.44, 0.44, 0.45, 0.43]
    for epoch, loss in enumerate(epochs_loss, start=1):
        print(f"epoch {epoch}: loss={loss}")

    names = ["sepal_len", "sepal_wid"]
    values = [5.1, 3.5]
    print("zipped:", dict(zip(names, values)))

    # break: a tiny early-stopping loop
    best, patience, waited = float("inf"), 2, 0
    for epoch, loss in enumerate(epochs_loss, start=1):
        if loss < best - 1e-3:
            best, waited = loss, 0
        else:
            waited += 1
            if waited >= patience:
                print(f"early stop at epoch {epoch}, best loss {best}")
                break


# ---------------------------------------------------------------------------
# 4. Functions, lambdas, and scope
# ---------------------------------------------------------------------------
def describe(values, *, precision=2):
    """Keyword-only argument after * — callers must write precision=..."""
    lo, hi = min(values), max(values)
    return round(lo, precision), round(hi, precision)   # returns a tuple


def log_metrics(step, **metrics):
    """**kwargs collects any named arguments into a dict."""
    parts = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
    return f"step {step}: {parts}"


def append_bad(item, bucket=[]):      # noqa: B006 — deliberately wrong
    bucket.append(item)
    return bucket


def append_good(item, bucket=None):
    if bucket is None:
        bucket = []
    bucket.append(item)
    return bucket


THRESHOLD = 0.5  # global


def classify(prob):
    # Python looks names up in order: Local, Enclosing, Global, Built-in (LEGB)
    return 1 if prob >= THRESHOLD else 0


def functions_demo() -> None:
    section("4. Functions, lambdas, and scope")

    print("describe:", describe([3.14159, 2.71828, 1.41421], precision=3))
    print(log_metrics(10, loss=0.4321, acc=0.9))

    # Lambda: a small anonymous function, mostly used as a sort key
    models = [("tree", 0.74), ("forest", 0.88), ("logreg", 0.81)]
    ranked = sorted(models, key=lambda pair: pair[1], reverse=True)
    print("ranked by score:", ranked)

    # The mutable-default-argument trap
    print("\nappend_bad:", append_bad(1), append_bad(2))   # shares one list!
    print("append_good:", append_good(1), append_good(2))

    print("\npredictions:", [classify(p) for p in (0.1, 0.5, 0.93)])


# ---------------------------------------------------------------------------
# 5. Comprehensions
# ---------------------------------------------------------------------------
def comprehensions_demo() -> None:
    section("5. Comprehensions")

    raw = ["  Cat", "dog ", "CAT", "Bird", "dog"]

    cleaned = [w.strip().lower() for w in raw]                     # list
    vocab = {w for w in cleaned}                                    # set
    word_to_id = {w: i for i, w in enumerate(sorted(vocab))}        # dict
    encoded = [word_to_id[w] for w in cleaned]
    long_words = [w for w in cleaned if len(w) > 3]                 # with filter

    print("cleaned:   ", cleaned)
    print("vocab:     ", sorted(vocab))
    print("word_to_id:", word_to_id)
    print("encoded:   ", encoded)
    print("long words:", long_words)

    # Nested comprehension: a 3x3 identity matrix as a list of lists
    identity = [[1 if r == c else 0 for c in range(3)] for r in range(3)]
    print("identity:  ", identity)

    # Generator expression: computes lazily, no list kept in memory
    total = sum(x * x for x in range(1_000_000))
    print("sum of squares below 1e6:", total)


# ---------------------------------------------------------------------------
# 6. Performance: membership in list vs set
# ---------------------------------------------------------------------------
def performance_demo() -> None:
    section("6. Performance: list vs set membership")

    ids_list = list(range(100_000))
    ids_set = set(ids_list)
    target = 99_999

    t_list = timeit.timeit(lambda: target in ids_list, number=200)
    t_set = timeit.timeit(lambda: target in ids_set, number=200)
    print(f"list lookup: {t_list * 1e3:.2f} ms  (O(n) — scans every item)")
    print(f"set lookup:  {t_set * 1e3:.4f} ms  (O(1) average — hash lookup)")


def main() -> None:
    data_types_and_operators()
    collections_demo()
    control_flow()
    functions_demo()
    comprehensions_demo()
    performance_demo()


if __name__ == "__main__":
    main()
