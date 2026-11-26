"""Tests for Day 02. Run: pytest days/day-02-oop-dataclasses"""

import dataclasses

import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_wine, make_classification
from sklearn.dummy import DummyClassifier
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler as SkMinMax
from sklearn.preprocessing import StandardScaler as SkStandard

from experiment_records import (
    ExperimentResult,
    Leaderboard,
    SupportsPredict,
    Task,
    ThresholdRule,
    TrainingConfig,
    evaluate,
)
from mini_estimators import (
    BaseEstimator,
    Classifier,
    KNNClassifier,
    MajorityClassifier,
    MeanRegressor,
    MinMaxScaler,
    NotFittedError,
    Pipeline,
    StandardScaler,
    Transformer,
    split_rows,
)


@pytest.fixture(scope="module")
def wine():
    d = load_wine()
    return d.data.tolist(), d.target.tolist()


# ----------------------------- transformers ---------------------------------
def test_standard_scaler_matches_sklearn(wine):
    X, _ = wine
    ours = StandardScaler().fit_transform(X)
    np.testing.assert_allclose(ours, SkStandard().fit_transform(np.array(X)), atol=1e-9)


def test_minmax_scaler_matches_sklearn(wine):
    X, _ = wine
    ours = MinMaxScaler().fit_transform(X)
    np.testing.assert_allclose(ours, SkMinMax().fit_transform(np.array(X)), atol=1e-12)


def test_constant_column_does_not_produce_nan():
    X = [[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]]
    for scaler in (StandardScaler(), MinMaxScaler()):
        out = scaler.fit_transform(X)
        assert [row[1] for row in out] == [0.0, 0.0, 0.0]


def test_scaler_uses_training_statistics_only():
    scaler = StandardScaler().fit([[0.0], [2.0]])        # mean 1, std 1
    assert scaler.transform([[10.0]]) == [[9.0]]


# ----------------------------- models ---------------------------------------
def test_knn_matches_sklearn_predictions():
    X, y = make_classification(n_samples=200, n_features=5, n_informative=3,
                               n_classes=3, random_state=0)
    X_tr, X_te, y_tr, y_te = X[:150], X[150:], y[:150], y[150:]
    ours = KNNClassifier(k=5).fit(X_tr.tolist(), y_tr.tolist()).predict(X_te.tolist())
    ref = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr).predict(X_te)
    assert ours == ref.tolist()


def test_majority_matches_dummy_classifier_including_ties():
    X = [[0.0]] * 6
    y = [2, 1, 2, 1, 0, 0]                          # three-way tie -> smallest label
    ours = MajorityClassifier().fit(X, y)
    ref = DummyClassifier(strategy="most_frequent").fit(X, y)
    assert ours.predict(X) == ref.predict(X).tolist() == [0] * 6


def test_mean_regressor_r2_matches_sklearn():
    d = load_diabetes()
    X, y = d.data.tolist(), d.target.tolist()
    X_tr, X_te, y_tr, y_te = split_rows(X, y, seed=1)
    model = MeanRegressor().fit(X_tr, y_tr)
    assert model.score(X_tr, y_tr) == pytest.approx(0.0, abs=1e-12)
    assert model.score(X_te, y_te) == pytest.approx(r2_score(y_te, model.predict(X_te)))


def test_predict_before_fit_raises_not_fitted():
    with pytest.raises(NotFittedError):
        KNNClassifier().predict([[1.0]])
    # The custom error is also a RuntimeError, so broad handlers still catch it.
    with pytest.raises(RuntimeError):
        StandardScaler().transform([[1.0]])


@pytest.mark.parametrize("bad_k", [0, -1, 2.5, "3"])
def test_knn_rejects_invalid_k(bad_k):
    with pytest.raises(ValueError):
        KNNClassifier(k=bad_k)


def test_knn_k_larger_than_data_raises():
    with pytest.raises(ValueError):
        KNNClassifier(k=10).fit([[0.0], [1.0]], [0, 1])


def test_abstract_classes_cannot_be_instantiated():
    for cls in (BaseEstimator, Transformer, Classifier):
        with pytest.raises(TypeError):
            cls()


def test_params_repr_and_fitted_state():
    knn = KNNClassifier(k=3)
    assert knn.get_params() == {"k": 3}
    assert repr(knn) == "KNNClassifier(k=3)"
    assert not knn.is_fitted
    knn.fit([[0.0], [1.0], [2.0]], [0, 0, 1])
    assert knn.is_fitted


# ----------------------------- pipeline -------------------------------------
def test_pipeline_equals_manual_steps(wine):
    X, y = wine
    X_tr, X_te, y_tr, y_te = split_rows(X, y, seed=0)
    pipe = Pipeline([("scale", StandardScaler()), ("knn", KNNClassifier(k=5))]).fit(X_tr, y_tr)
    scaler = StandardScaler().fit(X_tr)
    manual = KNNClassifier(k=5).fit(scaler.transform(X_tr), y_tr)
    assert pipe.predict(X_te) == manual.predict(scaler.transform(X_te))
    assert pipe.score(X_te, y_te) > 0.9           # scaling makes KNN work on wine
    assert len(pipe) == 2 and isinstance(pipe["scale"], StandardScaler)
    with pytest.raises(KeyError):
        pipe["missing"]


def test_pipeline_validates_steps():
    with pytest.raises(TypeError):
        Pipeline([("knn", KNNClassifier()), ("scale", StandardScaler())])
    with pytest.raises(ValueError):
        Pipeline([("a", StandardScaler()), ("a", KNNClassifier())])


def test_split_rows_is_reproducible_and_disjoint():
    X = [[float(i)] for i in range(20)]
    y = list(range(20))
    a = split_rows(X, y, test_size=0.25, seed=3)
    b = split_rows(X, y, test_size=0.25, seed=3)
    assert a == b
    _, _, y_tr, y_te = a
    assert len(y_te) == 5 and not set(y_tr) & set(y_te)
    assert sorted(y_tr + y_te) == y


# ----------------------------- dataclasses ----------------------------------
def test_config_is_frozen_hashable_and_replaceable():
    cfg = TrainingConfig(model="logreg")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.epochs = 3
    changed = dataclasses.replace(cfg, epochs=3)
    assert changed.epochs == 3 and cfg.epochs == 10
    assert hash(cfg) == hash(TrainingConfig(model="logreg"))


@pytest.mark.parametrize("kwargs", [{"learning_rate": 0}, {"learning_rate": 2},
                                    {"epochs": 0}, {"batch_size": -1}])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(model="m", **kwargs)


def test_config_from_dict_converts_and_rejects_typos():
    cfg = TrainingConfig.from_dict({"model": "ridge", "task": "regression", "tags": ["a"]})
    assert cfg.task is Task.REGRESSION and cfg.tags == ("a",)
    with pytest.raises(KeyError):
        TrainingConfig.from_dict({"model": "ridge", "epoch": 5})
    with pytest.raises(ValueError):
        TrainingConfig.from_dict({"model": "ridge", "task": "clustering"})


def test_results_sort_by_score_and_direction():
    runs = [ExperimentResult("a", 0.7), ExperimentResult("b", 0.9), ExperimentResult("c", 0.8)]
    assert [r.run_id for r in sorted(runs)] == ["a", "c", "b"]
    rmse = [ExperimentResult("x", 3.0, higher_is_better=False),
            ExperimentResult("y", 1.5, higher_is_better=False)]
    assert max(rmse).run_id == "y"


def test_default_factory_gives_each_result_its_own_dict():
    r1, r2 = ExperimentResult("a", 0.5), ExperimentResult("b", 0.6)
    r1.metrics["f1"] = 0.4
    assert r2.metrics == {}


def test_leaderboard_behaviour():
    board = Leaderboard("demo")
    with pytest.raises(ValueError):
        board.best()
    board.add(ExperimentResult("a", 0.6))
    board.add(ExperimentResult("b", 0.8))
    with pytest.raises(ValueError):
        board.add(ExperimentResult("a", 0.9))
    assert len(board) == 2 and "b" in board and "z" not in board
    assert board.best().run_id == "b"
    assert [r.run_id for r in board.ranked()] == ["b", "a"]
    assert [r.run_id for r in board] == ["a", "b"]          # insertion order


def test_protocol_is_structural():
    rule = ThresholdRule(feature=0, threshold=0.5)
    assert isinstance(rule, SupportsPredict)
    assert isinstance(KNNClassifier(), SupportsPredict)   # unrelated class, same shape
    assert not isinstance(object(), SupportsPredict)
    assert evaluate(rule, [[0.1], [0.9]], [0, 1]) == 1.0
