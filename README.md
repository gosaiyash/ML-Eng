# ML Engineer — 70 Days of Code

Day-by-day practice code that follows the
[ML Engineer Preparation Handbook](ML_Engineer_Preparation_Handbook.pdf)
([Word version](ML_Engineer_Preparation_Handbook.docx)).
One day of material is added at a time, in the handbook's recommended learning order:
foundations → core ML → project engineering → deployment and MLOps → deep learning and GenAI → job preparation.

Each day folder has:

- `README.md` — what the day covers, short notes, and interview questions to test yourself
- Python files you can run directly
- `test_dayNN.py` — tests for that day's code (where it makes sense)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run a day's code and its tests:

```bash
python days/day-01-python-fundamentals/fundamentals.py
pytest days/day-01-python-fundamentals
```

Run every test in the repo with `pytest`.

## Progress

✅ done · ⬜ coming up

### Phase 1 — Foundations

| Day | Topic | Handbook | Status |
|----:|-------|----------|:------:|
| 01 | [Python fundamentals: data types, collections, functions, comprehensions](days/day-01-python-fundamentals) | §01 | ✅ |
| 02 | OOP: classes, inheritance, composition, dataclasses, type hints | §01 | ⬜ |
| 03 | Exceptions, custom errors, CSV/JSON files, context managers | §01 | ⬜ |
| 04 | Iterators, generators, decorators, logging | §01 | ⬜ |
| 05 | Modules, packages, a small CLI, and pytest | §01 | ⬜ |
| 06 | NumPy: arrays, shapes, indexing, broadcasting, vectorization | §03A | ⬜ |
| 07 | Pandas: Series, DataFrames, filtering, groupby, merge, pivot | §03A | ⬜ |
| 08 | Pandas cleaning: missing values, duplicates, datetimes | §03A | ⬜ |
| 09 | SQL basics with SQLite: SELECT, WHERE, joins, GROUP BY, CASE WHEN | §03D | ⬜ |
| 10 | Advanced SQL: CTEs, window functions, indexes, transactions | §03D | ⬜ |
| 11 | Linear algebra: vectors, matrices, norms, inverse, rank, projections | §02A | ⬜ |
| 12 | Eigenvalues, SVD, and PCA from scratch | §02A | ⬜ |
| 13 | Calculus: derivatives, gradients, chain rule, gradient descent from scratch | §02B | ⬜ |
| 14 | Probability: distributions, Bayes' theorem, LLN, CLT by simulation | §02C | ⬜ |
| 15 | Statistics: outliers, confidence intervals, hypothesis tests, A/B tests | §02D | ⬜ |
| 16 | Information theory: entropy, cross-entropy, KL divergence, information gain | §02E | ⬜ |
| 17 | EDA project — Phase 1 exit check: clean and analyse a dataset with a tested script | §03B | ⬜ |

### Phase 2 — Core ML

| Day | Topic | Handbook | Status |
|----:|-------|----------|:------:|
| 18 | Core concepts: splits, baselines, under/overfitting, bias-variance | §04 | ⬜ |
| 19 | Preprocessing: imputation, scaling, encoding, Pipeline, ColumnTransformer, leakage | §03C | ⬜ |
| 20 | Feature engineering: date/cyclical features, feature creation and selection | §03C | ⬜ |
| 21 | Linear regression from scratch vs scikit-learn | §05A | ⬜ |
| 22 | Polynomial, Ridge, Lasso, and Elastic Net regression | §05A | ⬜ |
| 23 | Logistic regression from scratch and decision thresholds | §05B | ⬜ |
| 24 | K-Nearest Neighbors and Naive Bayes | §05B | ⬜ |
| 25 | Decision trees: Gini/entropy splits from scratch and scikit-learn | §05A, §05B | ⬜ |
| 26 | Bagging, random forests, and feature importance | §05B | ⬜ |
| 27 | Gradient boosting from scratch, plus XGBoost/LightGBM basics | §05B | ⬜ |
| 28 | Support Vector Machines and SVR | §05A, §05B | ⬜ |
| 29 | Ensembles: voting, stacking, blending | §05B | ⬜ |
| 30 | Classification metrics: confusion matrix, precision/recall, F1, ROC-AUC, PR, log loss | §06A | ⬜ |
| 31 | Regression metrics: MAE, MSE, RMSE, R², adjusted R², MAPE, residuals | §06B | ⬜ |
| 32 | Cross-validation: k-fold, stratified, group, time-series splits | §06C | ⬜ |
| 33 | Hyperparameter tuning, learning curves, validation curves | §06C | ⬜ |
| 34 | Imbalanced data: class weights, resampling, SMOTE, thresholds, calibration | §03C, §06C | ⬜ |
| 35 | Interpretability: permutation importance, SHAP ideas, error analysis | §06C | ⬜ |
| 36 | Clustering: K-Means, hierarchical, DBSCAN, GMM | §05C | ⬜ |
| 37 | Dimensionality reduction and anomaly detection: PCA, Isolation Forest, LOF | §05C | ⬜ |
| 38 | Model comparison project — Phase 2 exit check: compare 3+ models fairly | §06 | ⬜ |

### Phase 3 — Project engineering

| Day | Topic | Handbook | Status |
|----:|-------|----------|:------:|
| 39 | From notebook to package: src layout, config, logging, CLI | §01, §11B | ⬜ |
| 40 | Testing ML code: schema, feature, metric-threshold, reproducibility tests | §11E | ⬜ |
| 41 | Data validation pipeline: schemas, quality checks, dataset hashing, lineage | §09 | ⬜ |
| 42 | Reproducible project — Phase 3 exit check: someone else can rerun your results | §16 | ⬜ |

### Phase 4 — Deployment and MLOps

| Day | Topic | Handbook | Status |
|----:|-------|----------|:------:|
| 43 | Saving, loading, and versioning models with metadata | §10A | ⬜ |
| 44 | Prediction API with FastAPI and Pydantic | §10A | ⬜ |
| 45 | API tests, error handling, health checks, logging | §10A, §11E | ⬜ |
| 46 | Docker: Dockerfile and configuration with environment variables | §10B | ⬜ |
| 47 | Experiment tracking with MLflow | §11A | ⬜ |
| 48 | CI with GitHub Actions | §11C | ⬜ |
| 49 | Drift monitoring: PSI, KS test, prediction distributions | §11D | ⬜ |
| 50 | Model acceptance rules, registry, and rollback | §11B | ⬜ |

### Phase 5 — Deep learning, GenAI, and specializations

| Day | Topic | Handbook | Status |
|----:|-------|----------|:------:|
| 51 | Neural network from scratch in NumPy: forward pass and backprop | §08A | ⬜ |
| 52 | PyTorch tensors, devices, and autograd | §08B | ⬜ |
| 53 | PyTorch Dataset, DataLoader, training loop, checkpoints | §08B | ⬜ |
| 54 | Training techniques: optimizers, dropout, batch norm, LR schedules, early stopping | §08A | ⬜ |
| 55 | Convolutional neural networks | §08C | ⬜ |
| 56 | RNNs and LSTMs for sequences | §08C | ⬜ |
| 57 | Attention and a mini Transformer block | §08C | ⬜ |
| 58 | NLP basics: tokenization, n-grams, bag of words, TF-IDF, text classification | §12A | ⬜ |
| 59 | Embeddings, cosine similarity, and semantic search | §12A, §12C | ⬜ |
| 60 | RAG pipeline: chunking, retrieval, reranking, retrieval evaluation | §12C | ⬜ |
| 61 | Next-token prediction and sampling: temperature, top-k, top-p | §12B | ⬜ |
| 62 | Structured output validation, tool schemas, prompt-injection checks | §12B, §12E | ⬜ |
| 63 | Time series basics: trend, seasonality, stationarity, lag features | §07 | ⬜ |
| 64 | Forecasting: naive baselines, ARIMA, walk-forward backtesting | §07 | ⬜ |
| 65 | Recommender systems: collaborative filtering and matrix factorization | §15D | ⬜ |
| 66 | Responsible ML: subgroup errors, fairness metrics, model card | §13 | ⬜ |

### Phase 6 — Job preparation

| Day | Topic | Handbook | Status |
|----:|-------|----------|:------:|
| 67 | Python interview practice with tests | §18 | ⬜ |
| 68 | SQL interview practice | §18 | ⬜ |
| 69 | ML algorithms from scratch, interview style | §18 | ⬜ |
| 70 | Capstone review: readiness checklist and project stories | §20 | ⬜ |
