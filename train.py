from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parents[1]
official = ROOT / "data" / "ai4i2020.csv"
demo = ROOT / "data" / "ai4i_demo_sample.csv"
DATA = official if official.exists() else demo
OUT = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
# AI4I uses 'Machine failure' as target. Drop identifiers and failure-mode indicators
# (TWF, HDF, PWF, OSF, RNF) to avoid target leakage.
target = "Machine failure"
if target not in df.columns:
    raise ValueError(f"Target column '{target}' was not found in {DATA.name}")
drop_cols = [c for c in ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"] if c in df.columns]
y = df[target].astype(int)
X = df.drop(columns=[target] + drop_cols)

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]
prep = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
models = {
    "logistic_regression": (LogisticRegression(max_iter=2000, class_weight="balanced"),
                            {"model__C": [0.1, 1, 10]}),
    "decision_tree": (DecisionTreeClassifier(class_weight="balanced", random_state=42),
                      {"model__max_depth": [3, 5, None], "model__min_samples_split": [2, 10]}),
    "random_forest": (RandomForestClassifier(class_weight="balanced", random_state=42),
                      {"model__n_estimators": [100, 200], "model__max_depth": [None, 10]})
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
best_name, best_score, best_pipe = None, -1, None

for name, (estimator, params) in models.items():
    pipe = Pipeline([("prep", prep), ("model", estimator)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
    search = GridSearchCV(pipe, params, cv=cv, scoring="f1", n_jobs=-1)
    search.fit(X_train, y_train)
    pred = search.best_estimator_.predict(X_test)
    results[name] = {
        "cross_validation_f1_mean": float(scores.mean()),
        "cross_validation_f1_scores": [float(s) for s in scores],
        "best_parameters": search.best_params_,
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_precision": float(precision_score(y_test, pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0)
    }
    if search.best_score_ > best_score:
        best_name, best_score, best_pipe = name, search.best_score_, search.best_estimator_

joblib.dump(best_pipe, OUT / "best_machine_failure_model.joblib")
summary = {"dataset_file": DATA.name, "rows": int(len(df)), "selected_model": best_name, "models": results}
(OUT / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"Selected model: {best_name}")
for name, r in results.items():
    print(name, "CV F1:", round(r["cross_validation_f1_mean"], 4),
          "Test F1:", round(r["test_f1"], 4), "Recall:", round(r["test_recall"], 4))
