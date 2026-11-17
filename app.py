"""AutoML-lite: upload a CSV, pick a task, train & compare models.
Run with:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             mean_absolute_error, mean_squared_error, r2_score)

SEED = 42
MAX_MB = 20
MIN_ROWS = 30

# ---- Supported algorithms (the "defined set") -------------------------------
CLASSIFIERS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=SEED),
    "Decision Tree": lambda: DecisionTreeClassifier(random_state=SEED),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=SEED),
    "KNN": lambda: KNeighborsClassifier(),
}
REGRESSORS = {
    "Linear Regression": lambda: LinearRegression(),
    "Ridge": lambda: Ridge(random_state=SEED),
    "Decision Tree": lambda: DecisionTreeRegressor(random_state=SEED),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=100, random_state=SEED),
    "KNN": lambda: KNeighborsRegressor(),
}


# ---- 1. Upload + validation -------------------------------------------------
def load_and_validate(file):
    """Return (dataframe, None) or (None, error message)."""
    if file.size > MAX_MB * 1024 * 1024:
        return None, f"File is larger than {MAX_MB} MB."
    try:
        df = pd.read_csv(file)
    except pd.errors.EmptyDataError:
        return None, "The file is empty."
    except Exception as e:
        return None, f"Could not read this as a CSV: {e}"
    if df.shape[1] < 2:
        return None, "Need at least 2 columns (features + a target). Check the delimiter."
    if len(df) < MIN_ROWS:
        return None, f"Need at least {MIN_ROWS} rows; found {len(df)}."
    if df.columns.duplicated().any():
        return None, "Duplicate column names found. Please rename them."
    df.columns = [str(c).strip() for c in df.columns]
    return df, None


# ---- 2. Reproducible preprocessing pipeline --------------------------------
def build_pipeline(X, model):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    return Pipeline([("pre", pre), ("model", model)])


# ---- 3. Evaluation ----------------------------------------------------------
def eval_classification(pipe, X_te, y_te):
    pred = pipe.predict(X_te)
    m = {
        "Accuracy": accuracy_score(y_te, pred),
        "Precision": precision_score(y_te, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_te, pred, average="weighted", zero_division=0),
        "F1": f1_score(y_te, pred, average="weighted", zero_division=0),
    }
    return m, pred


def eval_regression(pipe, X_te, y_te):
    pred = pipe.predict(X_te)
    m = {
        "MAE": mean_absolute_error(y_te, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
        "R2": r2_score(y_te, pred),
    }
    return m, pred


# ---- 4. Plots ---------------------------------------------------------------
def plot_confusion(y_te, pred, classes, title):
    cm = confusion_matrix(y_te, pred, labels=classes)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45)
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_roc(pipe, X_te, y_te):
    classes = list(pipe.classes_)
    if len(classes) != 2 or not hasattr(pipe, "predict_proba"):
        return None
    proba = pipe.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, proba, pos_label=classes[1])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve"); ax.legend()
    fig.tight_layout()
    return fig


def plot_actual_vs_pred(y_te, pred):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.scatter(y_te, pred, alpha=0.6)
    lo, hi = min(y_te.min(), pred.min()), max(y_te.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title("Actual vs predicted")
    fig.tight_layout()
    return fig


def plot_residuals(y_te, pred):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.scatter(pred, y_te - pred, alpha=0.6)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Residual"); ax.set_title("Residuals")
    fig.tight_layout()
    return fig


# ---- 5. UI ------------------------------------------------------------------
st.set_page_config(page_title="ML Trainer", layout="wide")
st.title("Tabular ML Trainer")
st.caption("Upload a CSV, choose a task, train models and view the results.")

file = st.sidebar.file_uploader("1. Upload CSV", type=["csv"])
if file is None:
    st.info("Upload a CSV file in the sidebar to begin.")
    st.stop()

df, err = load_and_validate(file)
if err:
    st.error(err)
    st.stop()

st.subheader("Data preview")
st.dataframe(df.head(20), use_container_width=True)
c1, c2, c3 = st.columns(3)
c1.metric("Rows", len(df)); c2.metric("Columns", df.shape[1])
c3.metric("Missing cells", int(df.isna().sum().sum()))

# Target + task selection
target = st.sidebar.selectbox("2. Target column", df.columns, index=len(df.columns) - 1)
task = st.sidebar.radio("3. Problem type", ["Classification", "Regression"])

y_raw = df[target]
n_unique = y_raw.nunique()
if task == "Regression" and not pd.api.types.is_numeric_dtype(y_raw):
    st.error("Regression needs a numeric target. Pick another column or use Classification.")
    st.stop()
if task == "Classification":
    if n_unique > 20:
        st.warning(f"Target has {n_unique} unique values - this looks more like regression.")
    if n_unique < 2:
        st.error("Target has only one class."); st.stop()
    if y_raw.value_counts().min() < 2:
        st.error("Every class needs at least 2 rows (for the stratified split)."); st.stop()

features = st.sidebar.multiselect(
    "Feature columns", [c for c in df.columns if c != target],
    default=[c for c in df.columns if c != target])
if not features:
    st.error("Select at least one feature."); st.stop()

registry = CLASSIFIERS if task == "Classification" else REGRESSORS
chosen = st.sidebar.multiselect("4. Algorithms (pick several to compare)",
                                list(registry), default=[list(registry)[0]])
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)

if st.sidebar.button("Train & evaluate", type="primary"):
    if not chosen:
        st.error("Pick at least one algorithm."); st.stop()

    data = df.dropna(subset=[target])
    X = data[features]
    y = data[target].astype(str) if task == "Classification" else data[target]

    # ONE split shared by every model -> fair comparison
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=SEED,
        stratify=y if task == "Classification" else None)
    st.success(f"Train rows: {len(X_tr)} | Test rows: {len(X_te)} | Seed: {SEED}")

    rows, details = [], {}
    for name in chosen:
        pipe = build_pipeline(X_tr, registry[name]())
        pipe.fit(X_tr, y_tr)
        if task == "Classification":
            m, pred = eval_classification(pipe, X_te, y_te)
        else:
            m, pred = eval_regression(pipe, X_te, y_te)
        rows.append({"Model": name, **m})
        details[name] = (pipe, pred)

    res = pd.DataFrame(rows).set_index("Model")
    st.subheader("Results")
    st.dataframe(res.style.format("{:.4f}"), use_container_width=True)

    if len(chosen) > 1:
        st.subheader("Model comparison")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        res.plot.bar(ax=ax); ax.set_ylabel("Score"); plt.xticks(rotation=0)
        fig.tight_layout(); st.pyplot(fig)

    st.subheader("Per-model graphs")
    for name, (pipe, pred) in details.items():
        with st.expander(name, expanded=len(chosen) == 1):
            cols = st.columns(2)
            if task == "Classification":
                classes = sorted(y.unique())
                cols[0].pyplot(plot_confusion(y_te, pred, classes, "Confusion matrix"))
                roc = plot_roc(pipe, X_te, y_te)
                if roc:
                    cols[1].pyplot(roc)
                else:
                    cols[1].caption("ROC curve is shown for binary targets only.")
            else:
                cols[0].pyplot(plot_actual_vs_pred(y_te, pred))
                cols[1].pyplot(plot_residuals(y_te, pred))

    st.download_button("Download results (CSV)", res.to_csv().encode(),
                       "results.csv", "text/csv")
else:
    st.info("Set your options in the sidebar, then click **Train & evaluate**.")
