import json
import numpy as np
import pandas as pd
from joblib import dump, load

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

TEXT_COL = "job_title_clean"


def build_pipeline():
    # TF-IDF for job title + numeric features
    text = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("title", text, TEXT_COL),
            ("num", StandardScaler(), ["connections_num", "has_aspiring", "has_seeking", "has_manager", "has_engineer"]),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )

    # Baseline probabilistic model (will be weak at first; improves with feedback)
    clf = LogisticRegression(penalty="l2", C=0.5, max_iter=2000)

    return Pipeline([("pre", pre), ("clf", clf)])


def keyword_relevance_scores(df: pd.DataFrame, keywords: str) -> np.ndarray:
    """Return cosine similarity between each title and the keywords."""
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vec.fit_transform(df[TEXT_COL].fillna(""))
    q = vec.transform([keywords.lower().strip()])
    sims = cosine_similarity(X, q).ravel()
    return sims


def initial_filter(df: pd.DataFrame, keywords: str, tau: float = 0.08) -> pd.DataFrame:
    """Hard gate: remove candidates whose title is not relevant to the role query."""
    sims = keyword_relevance_scores(df, keywords)
    out = df.copy()
    out["kw_sim"] = sims
    return out[out["kw_sim"] >= tau].sort_values("kw_sim", ascending=False)


def fit_baseline_model(df: pd.DataFrame, y_col: str, model_path: str):
    """Train baseline logistic regression when you have labels (even a few)."""
    pipe = build_pipeline()
    X = df.copy()
    y = df[y_col].astype(int)
    pipe.fit(X, y)
    dump(pipe, model_path)


def score_candidates(df: pd.DataFrame, keywords: str, model_path: str | None = None) -> pd.DataFrame:
    out = df.copy()
    out["kw_sim"] = keyword_relevance_scores(out, keywords)

    # If no trained model yet, use a simple proxy for base fit.
    if model_path is None:
        # conservative base: mostly keyword similarity, slight boost for connections
        out["base_fit"] = np.clip(
            0.85 * out["kw_sim"] + 0.15 * (np.log1p(out["connection_num"]) / np.log1p(500)),
            0, 1
        )
    else:
        model = load(model_path)
        out["base_fit"] = model.predict_proba(out)[:, 1]

    return out
