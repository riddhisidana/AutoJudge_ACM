#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
# Load saved models
# -----------------------------
tfidf = joblib.load("tfidf.pkl")
scaler = joblib.load("scaler.pkl")
classifier = joblib.load("final_classifier.pkl")
regressor = joblib.load("final_regressor.pkl")

# -----------------------------
# Numeric feature extractor
def extract_numeric_features(text):
    text_lower = text.lower()

    algo_groups = {
        "dp": [
            "dp", "dynamic programming", "knapsack",
            "bitmask dp", "state", "transition",
            "memoization", "tabulation"
        ],
        "graph": [
            "graph", "bfs", "dfs", "dijkstra",
            "bellman ford", "floyd warshall",
            "topological", "shortest path",
            "flow", "max flow", "min cut",
            "matching", "strongly connected",
            "bridges", "articulation points",
            "tree", "lca"
        ],
        "ds": [
            "segment tree", "fenwick",
            "binary indexed tree", "heap",
            "priority queue", "stack",
            "queue", "deque",
            "union find", "disjoint set",
            "sparse table"
        ],
        "math": [
            "modulo", "prime", "gcd", "lcm",
            "combinatorics", "permutations",
            "probability", "matrix exponentiation",
            "fft", "fast fourier transform",
            "number theory"
        ],
        "geometry": [
            "geometry", "convex hull",
            "sweep line", "cross product",
            "dot product", "orientation"
        ],
        "string": [
            "string", "substring",
            "palindrome", "kmp",
            "z algorithm", "suffix array",
            "trie", "rolling hash"
        ],
        "greedy": [
            "greedy", "two pointers",
            "sliding window", "interval",
            "activity selection"
        ]
    }

    group_counts = {}
    for group, keywords in algo_groups.items():
        count = sum(text_lower.count(k) for k in keywords)
        group_counts[f"{group}_count"] = np.log1p(count)

    math_symbols = "+-*/^=<>(){}[]|&!%"
    math_symbol_count = sum(text.count(sym) for sym in math_symbols)

    text_len = len(text)

    has_constraints = int(
        "≤" in text or "<=" in text_lower or "constraints" in text_lower
    )
    has_big_n = int(
        "10^5" in text or "10^6" in text or
        "10^7" in text or "10^" in text
    )
    has_time_limit = int(
        "time limit" in text_lower or "seconds" in text_lower
    )

    return {
        "text_length": np.log1p(text_len),
        "math_symbol_count": np.log1p(math_symbol_count),
        "has_constraints": has_constraints,
        "has_big_n": has_big_n,
        "has_time_limit": has_time_limit,
        **group_counts
    }

# -----------------------------
# Build full feature vector
# -----------------------------
def build_features(text):
    text = text.lower().strip()

    # TF-IDF
    X_tfidf = tfidf.transform([text])

    # Numeric features
    num_feats = extract_numeric_features(text)
    num_df = pd.DataFrame([num_feats])
    X_num_scaled = scaler.transform(num_df)

    # Combine
    X_final = hstack([X_tfidf, X_num_scaled])
    return X_final

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AutoJudge", layout="centered")

st.title("🧠 AutoJudge: Programming Problem Difficulty Predictor")

st.markdown(
    """
    Paste the **problem description**, **input format**, and **output format** below.
    The system predicts:
    - **Difficulty Class** (Easy / Medium / Hard)
    - **Difficulty Score** (numerical)
    """
)

problem_desc = st.text_area("Problem Description", height=150)
input_desc = st.text_area("Input Description", height=100)
output_desc = st.text_area("Output Description", height=100)

if st.button("Predict Difficulty"):
    full_text = problem_desc + " " + input_desc + " " + output_desc

    if full_text.strip() == "":
        st.warning("Please enter problem details before predicting.")
    else:
        X_input = build_features(full_text)

        class_pred = classifier.predict(X_input)[0]
        score_pred = regressor.predict(X_input)[0]

        label_map = {0: "Easy", 1: "Medium", 2: "Hard"}

        st.success(f"### Predicted Difficulty Class: **{label_map[class_pred]}**")
        st.info(f"### Predicted Difficulty Score: **{score_pred:.2f}**")


# In[ ]:




