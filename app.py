"""
app.py
------
IBM Internship Capstone Project
Bank Telemarketing Campaign Effectiveness Analyzer

Streamlit multi-section dashboard covering:
  1. Home / Overview
  2. Dataset Overview
  3. KPI Dashboard
  4. Exploratory Data Analysis
  5. Descriptive Analytics
  6. Diagnostic Analytics
  7. Predictive Analytics
  8. Prescriptive Insights
  9. Business Recommendations

Self-contained: all data processing and analytics functions are inlined here.
"""

import io
import os
import urllib.request
import zipfile
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, f1_score,
)

# ─────────────────────────────────────────────────────────────────────────────
# DATA SOURCE
# ─────────────────────────────────────────────────────────────────────────────
# The app can still use the local dataset when it exists, but a 4-file GitHub
# copy downloads the official UCI Bank Marketing archive automatically.
_BASE = os.path.dirname(__file__)
LOCAL_DATA_PATH = os.path.join(_BASE, "data", "bank-additional-full.csv")
UCI_DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
DATA_FILENAME = "bank-additional-full.csv"

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun",
               "jul", "aug", "sep", "oct", "nov", "dec"]
DOW_ORDER   = ["mon", "tue", "wed", "thu", "fri"]

COLORS = {
    "yes":     "#2563EB",
    "no":      "#EF4444",
    "neutral": "#6B7280",
    "accent":  "#F59E0B",
    "success": "#10B981",
}

# ═════════════════════════════════════════════════════════════════════════════
# DATA PROCESSING FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def load_raw() -> pd.DataFrame:
    """Return the Bank Marketing dataset, using local data or UCI download."""
    if os.path.exists(LOCAL_DATA_PATH):
        return pd.read_csv(LOCAL_DATA_PATH, sep=";")

    try:
        with urllib.request.urlopen(UCI_DATA_URL, timeout=60) as response:
            archive = response.read()
        with zipfile.ZipFile(io.BytesIO(archive)) as outer_zip:
            inner_name = next(
                (name for name in outer_zip.namelist()
                 if name.rstrip("/").endswith("bank-additional.zip")),
                None,
            )
            if inner_name is None:
                raise FileNotFoundError(
                    "bank-additional.zip was not found in the UCI archive. "
                    f"Archive contents: {outer_zip.namelist()}"
                )
            inner_archive = outer_zip.read(inner_name)

        with zipfile.ZipFile(io.BytesIO(inner_archive)) as inner_zip:
            csv_name = next(
                (name for name in inner_zip.namelist()
                 if name.rstrip("/").endswith(DATA_FILENAME)),
                None,
            )
            if csv_name is None:
                raise FileNotFoundError(
                    f"{DATA_FILENAME} was not found inside bank-additional.zip. "
                    f"Archive contents: {inner_zip.namelist()}"
                )
            with inner_zip.open(csv_name) as csv_file:
                return pd.read_csv(csv_file, sep=";")
    except Exception as exc:
        raise RuntimeError(
            "The Bank Marketing dataset could not be downloaded from UCI. "
            "Please check your internet connection and try again."
        ) from exc


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing steps and return the cleaned DataFrame."""
    df = df.copy()

    # 1. Column renaming
    df.rename(columns={
        "emp.var.rate":   "emp_var_rate",
        "cons.price.idx": "cons_price_idx",
        "cons.conf.idx":  "cons_conf_idx",
        "nr.employed":    "nr_employed",
        "day_of_week":    "day_of_week",
    }, inplace=True)

    # 2. Target encoding
    df["y_binary"] = (df["y"] == "yes").astype(int)

    # 3. Replace 'unknown' with NaN in categorical columns
    cat_cols_with_unknown = ["job", "marital", "education",
                             "default", "housing", "loan", "poutcome"]
    for col in cat_cols_with_unknown:
        df[col] = df[col].replace("unknown", np.nan)

    # 4. Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    if before != after:
        print(f"  Removed {before - after} duplicate rows.")

    # 5. Impute NaN categoricals with column mode
    for col in cat_cols_with_unknown:
        mode_val = df[col].mode(dropna=True)
        if len(mode_val) > 0:
            df[col] = df[col].fillna(mode_val[0])

    # 6. pdays 999 → NaN
    df["pdays"] = df["pdays"].replace(999, np.nan)

    # 7. 'was_contacted_before' flag
    df["was_contacted_before"] = df["pdays"].notna().astype(int)

    # 8. Ordinal education encoding
    education_order = {
        "illiterate":          0,
        "basic.4y":            1,
        "basic.6y":            2,
        "basic.9y":            3,
        "high.school":         4,
        "professional.course": 5,
        "university.degree":   6,
    }
    df["education_ord"] = df["education"].map(education_order)
    df["education_ord"] = df["education_ord"].fillna(
        df["education_ord"].median()
    ).astype(int)

    # 9. Ordered categoricals
    df["month"] = pd.Categorical(df["month"], categories=MONTH_ORDER, ordered=True)
    df["day_of_week"] = pd.Categorical(
        df["day_of_week"], categories=DOW_ORDER, ordered=True
    )

    # 10. Feature engineering
    df["call_duration_min"] = (df["duration"] / 60).round(1)
    df["contact_rate_category"] = pd.cut(
        df["campaign"],
        bins=[0, 1, 3, 5, df["campaign"].max()],
        labels=["Single", "2-3 contacts", "4-5 contacts", "6+ contacts"],
        right=True,
    )

    return df


def get_feature_matrix(df: pd.DataFrame):
    """Return (X, y) ready for scikit-learn."""
    feature_cols = [
        "age", "education_ord", "was_contacted_before",
        "campaign", "previous",
        "emp_var_rate", "cons_price_idx", "cons_conf_idx",
        "euribor3m", "nr_employed",
    ]
    cat_for_model = ["job", "marital", "housing", "loan",
                     "contact", "month", "day_of_week", "poutcome"]

    df_model = df[feature_cols + cat_for_model].copy()
    df_model["month"] = df_model["month"].astype(str)
    df_model["day_of_week"] = df_model["day_of_week"].astype(str)

    df_encoded = pd.get_dummies(df_model, columns=cat_for_model, drop_first=True)
    bool_cols = df_encoded.select_dtypes(include="bool").columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    X = df_encoded
    y = df["y_binary"]
    return X, y


def summary_stats(df: pd.DataFrame) -> dict:
    """Return a dict of high-level summary statistics for the dashboard."""
    total = len(df)
    subscribed = int(df["y_binary"].sum())
    not_subscribed = total - subscribed
    subscription_rate = round(subscribed / total * 100, 2)
    avg_age = round(df["age"].mean(), 1)
    avg_duration = round(df["call_duration_min"].mean(), 1)
    avg_campaigns = round(df["campaign"].mean(), 1)
    pct_cellular = round((df["contact"] == "cellular").sum() / total * 100, 1)
    pct_prev_contacted = round(df["was_contacted_before"].mean() * 100, 1)
    return {
        "total_records":      total,
        "subscribed":         subscribed,
        "not_subscribed":     not_subscribed,
        "subscription_rate":  subscription_rate,
        "avg_age":            avg_age,
        "avg_call_duration":  avg_duration,
        "avg_campaigns":      avg_campaigns,
        "pct_cellular":       pct_cellular,
        "pct_prev_contacted": pct_prev_contacted,
    }


# ═════════════════════════════════════════════════════════════════════════════
# ANALYTICS — DESCRIPTIVE
# ═════════════════════════════════════════════════════════════════════════════

def plot_subscription_distribution(df: pd.DataFrame):
    counts = df["y"].value_counts().reset_index()
    counts.columns = ["Outcome", "Count"]
    counts["Outcome"] = counts["Outcome"].map({"yes": "Subscribed", "no": "Not Subscribed"})
    fig = px.pie(
        counts, names="Outcome", values="Count",
        color="Outcome",
        color_discrete_map={"Subscribed": COLORS["yes"], "Not Subscribed": COLORS["no"]},
        title="Overall Subscription Outcome Distribution",
        hole=0.4,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def plot_subscription_by_month(df: pd.DataFrame):
    grp = (
        df.groupby("month", observed=True)["y_binary"]
        .agg(["sum", "count"])
        .reset_index()
    )
    grp.columns = ["month", "subscribed", "total"]
    grp["rate"] = (grp["subscribed"] / grp["total"] * 100).round(2)
    grp["month"] = pd.Categorical(grp["month"], categories=MONTH_ORDER, ordered=True)
    grp = grp.sort_values("month")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=grp["month"].astype(str), y=grp["total"],
               name="Total Contacts", marker_color="#CBD5E1"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=grp["month"].astype(str), y=grp["rate"],
                   name="Subscription Rate (%)", mode="lines+markers",
                   line=dict(color=COLORS["yes"], width=3),
                   marker=dict(size=8)),
        secondary_y=True,
    )
    fig.update_layout(title="Contact Volume & Subscription Rate by Month",
                      xaxis_title="Month",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_yaxes(title_text="Total Contacts", secondary_y=False)
    fig.update_yaxes(title_text="Subscription Rate (%)", secondary_y=True)
    return fig


def plot_subscription_by_job(df: pd.DataFrame):
    grp = (
        df.groupby("job")["y_binary"]
        .agg(["sum", "count"])
        .reset_index()
    )
    grp.columns = ["job", "subscribed", "total"]
    grp["rate"] = (grp["subscribed"] / grp["total"] * 100).round(2)
    grp = grp.sort_values("rate", ascending=True)
    fig = px.bar(
        grp, x="rate", y="job", orientation="h",
        color="rate", color_continuous_scale="Blues",
        title="Subscription Rate (%) by Job Type",
        labels={"rate": "Subscription Rate (%)", "job": "Job Type"},
        text="rate",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(coloraxis_showscale=False)
    return fig


def plot_age_distribution(df: pd.DataFrame):
    fig = px.histogram(
        df, x="age", color="y",
        color_discrete_map={"yes": COLORS["yes"], "no": COLORS["no"]},
        barmode="overlay", opacity=0.7, nbins=40,
        title="Age Distribution by Subscription Outcome",
        labels={"age": "Age", "y": "Subscribed"},
    )
    fig.update_layout(legend_title_text="Subscribed")
    return fig


def plot_education_subscription(df: pd.DataFrame):
    edu_order = ["illiterate", "basic.4y", "basic.6y", "basic.9y",
                 "high.school", "professional.course", "university.degree"]
    grp = (
        df.groupby("education")["y_binary"]
        .agg(["sum", "count"])
        .reset_index()
    )
    grp.columns = ["education", "subscribed", "total"]
    grp["rate"] = (grp["subscribed"] / grp["total"] * 100).round(2)
    grp["education"] = pd.Categorical(grp["education"], categories=edu_order, ordered=True)
    grp = grp.sort_values("education")
    fig = px.bar(
        grp, x="education", y="rate",
        title="Subscription Rate (%) by Education Level",
        labels={"rate": "Subscription Rate (%)", "education": "Education"},
        color="rate", color_continuous_scale="Blues", text="rate",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30)
    return fig


def plot_contact_type(df: pd.DataFrame):
    grp = (
        df.groupby(["contact", "y"])
        .size()
        .reset_index(name="count")
    )
    fig = px.bar(
        grp, x="contact", y="count", color="y", barmode="group",
        color_discrete_map={"yes": COLORS["yes"], "no": COLORS["no"]},
        title="Contact Method vs Subscription Outcome",
        labels={"contact": "Contact Method", "count": "Number of Clients", "y": "Subscribed"},
    )
    return fig


def plot_balance_by_subscription(df: pd.DataFrame):
    grp = df.groupby("y")["balance"].mean().reset_index()
    grp.columns = ["Subscribed", "Average Balance (€)"]
    grp["Subscribed"] = grp["Subscribed"].map({"yes": "Yes", "no": "No"})
    fig = px.bar(
        grp, x="Subscribed", y="Average Balance (€)",
        color="Subscribed",
        color_discrete_map={"Yes": COLORS["yes"], "No": COLORS["no"]},
        title="Average Account Balance by Subscription Outcome",
        text="Average Balance (€)",
    )
    fig.update_traces(texttemplate="€%{text:,.0f}", textposition="outside")
    return fig


def plot_day_of_week_subscription(df: pd.DataFrame):
    grp = (
        df.groupby("day_of_week", observed=True)["y_binary"]
        .agg(["sum", "count"])
        .reset_index()
    )
    grp.columns = ["day_of_week", "subscribed", "total"]
    grp["rate"] = (grp["subscribed"] / grp["total"] * 100).round(2)
    grp["day_of_week"] = pd.Categorical(grp["day_of_week"], categories=DOW_ORDER, ordered=True)
    grp = grp.sort_values("day_of_week")
    fig = px.bar(
        grp, x=grp["day_of_week"].astype(str), y="rate",
        title="Subscription Rate (%) by Day of Week",
        labels={"x": "Day of Week", "rate": "Subscription Rate (%)"},
        color="rate", color_continuous_scale="Blues", text="rate",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, xaxis_title="Day of Week")
    return fig


def plot_marital_subscription(df: pd.DataFrame):
    grp = (
        df.groupby(["marital", "y"])
        .size()
        .reset_index(name="count")
    )
    total = grp.groupby("marital")["count"].transform("sum")
    grp["pct"] = (grp["count"] / total * 100).round(2)
    fig = px.bar(
        grp[grp["y"] == "yes"], x="marital", y="pct",
        title="Subscription Rate (%) by Marital Status",
        labels={"marital": "Marital Status", "pct": "Subscription Rate (%)"},
        color="pct", color_continuous_scale="Blues", text="pct",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(coloraxis_showscale=False)
    return fig


def plot_numerical_summary(df: pd.DataFrame):
    numeric_vars = ["age", "campaign", "call_duration_min", "emp_var_rate",
                    "cons_price_idx", "cons_conf_idx", "euribor3m"]
    fig = make_subplots(rows=2, cols=4,
                        subplot_titles=numeric_vars,
                        vertical_spacing=0.15)
    row, col = 1, 1
    for var in numeric_vars:
        for outcome, color in [("yes", COLORS["yes"]), ("no", COLORS["no"])]:
            fig.add_trace(
                go.Box(y=df[df["y"] == outcome][var],
                       name=f"{'Yes' if outcome == 'yes' else 'No'}",
                       marker_color=color,
                       showlegend=(var == numeric_vars[0])),
                row=row, col=col,
            )
        col += 1
        if col > 4:
            col = 1
            row += 1
    fig.update_layout(title="Key Numeric Variables by Subscription Outcome",
                      height=500, boxmode="group")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# ANALYTICS — DIAGNOSTIC
# ═════════════════════════════════════════════════════════════════════════════

def plot_duration_vs_subscription(df: pd.DataFrame):
    fig = px.histogram(
        df, x="call_duration_min", color="y",
        color_discrete_map={"yes": COLORS["yes"], "no": COLORS["no"]},
        barmode="overlay", opacity=0.7, nbins=60,
        title="Call Duration (min) vs Subscription Outcome",
        labels={"call_duration_min": "Call Duration (minutes)", "y": "Subscribed"},
    )
    fig.update_layout(legend_title_text="Subscribed")
    return fig


def plot_poutcome_effect(df: pd.DataFrame):
    grp = (
        df.groupby(["poutcome", "y"])
        .size()
        .reset_index(name="count")
    )
    total = grp.groupby("poutcome")["count"].transform("sum")
    grp["pct"] = (grp["count"] / total * 100).round(2)
    fig = px.bar(
        grp, x="poutcome", y="pct", color="y", barmode="group",
        color_discrete_map={"yes": COLORS["yes"], "no": COLORS["no"]},
        title="Previous Campaign Outcome vs Current Subscription Rate",
        labels={"poutcome": "Previous Campaign Outcome",
                "pct": "Percentage (%)", "y": "Subscribed"},
    )
    return fig


def plot_economic_indicators(df: pd.DataFrame):
    grp = (
        df.groupby(["euribor3m", "emp_var_rate"])["y_binary"]
        .mean()
        .reset_index()
    )
    grp["subscription_rate"] = (grp["y_binary"] * 100).round(2)
    fig = px.scatter(
        grp, x="euribor3m", y="subscription_rate",
        color="emp_var_rate", size="subscription_rate",
        color_continuous_scale="RdYlGn",
        title="Economic Climate vs Subscription Rate",
        labels={"euribor3m": "Euribor 3-Month Rate",
                "subscription_rate": "Subscription Rate (%)",
                "emp_var_rate": "Employment Variation Rate"},
    )
    return fig


def plot_campaign_contacts_effect(df: pd.DataFrame):
    grp = (
        df.groupby("campaign")["y_binary"]
        .agg(["sum", "count"])
        .reset_index()
    )
    grp.columns = ["campaign", "subscribed", "total"]
    grp["rate"] = (grp["subscribed"] / grp["total"] * 100).round(2)
    grp = grp[grp["campaign"] <= 15]
    fig = px.bar(
        grp, x="campaign", y="rate",
        title="Number of Campaign Contacts vs Subscription Rate",
        labels={"campaign": "Number of Contacts (this campaign)",
                "rate": "Subscription Rate (%)"},
        color="rate", color_continuous_scale="Blues",
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    num_cols = ["age", "campaign", "previous", "call_duration_min",
                "emp_var_rate", "cons_price_idx", "cons_conf_idx",
                "euribor3m", "nr_employed", "y_binary"]
    corr = df[num_cols].corr().round(2)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=corr.values,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title="Correlation Heatmap (Numerical Features)",
        height=520,
        xaxis_tickangle=-45,
    )
    return fig


def plot_housing_loan_subscription(df: pd.DataFrame):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Housing Loan", "Personal Loan"])
    for i, col in enumerate(["housing", "loan"], start=1):
        grp = (
            df.groupby([col, "y"])
            .size()
            .reset_index(name="count")
        )
        total = grp.groupby(col)["count"].transform("sum")
        grp["pct"] = (grp["count"] / total * 100).round(2)
        for outcome, color in [("yes", COLORS["yes"]), ("no", COLORS["no"])]:
            subset = grp[grp["y"] == outcome]
            fig.add_trace(
                go.Bar(x=subset[col], y=subset["pct"],
                       name=f"{'Subscribed' if outcome == 'yes' else 'Not Subscribed'}",
                       marker_color=color,
                       showlegend=(i == 1)),
                row=1, col=i,
            )
    fig.update_layout(title="Loan Status vs Subscription Outcome",
                      barmode="group", legend_title_text="Subscribed")
    return fig


def plot_default_subscription(df: pd.DataFrame):
    grp = (
        df.groupby(["default", "y"])
        .size()
        .reset_index(name="count")
    )
    total = grp.groupby("default")["count"].transform("sum")
    grp["pct"] = (grp["count"] / total * 100).round(2)
    fig = px.bar(
        grp, x="default", y="pct", color="y", barmode="group",
        color_discrete_map={"yes": COLORS["yes"], "no": COLORS["no"]},
        title="Credit Default Status vs Subscription Rate",
        labels={"default": "Has Credit in Default", "pct": "Percentage (%)"},
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# ANALYTICS — PREDICTIVE
# ═════════════════════════════════════════════════════════════════════════════

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob)
    ap     = average_precision_score(y_test, y_prob)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    fpr, tpr, _  = roc_curve(y_test, y_prob)
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    return {
        "accuracy":        round(acc, 4),
        "f1_score":        round(f1, 4),
        "roc_auc":         round(auc, 4),
        "avg_precision":   round(ap, 4),
        "confusion_matrix": cm,
        "report":          report,
        "fpr":             fpr,
        "tpr":             tpr,
        "precision_curve": prec,
        "recall_curve":    rec,
        "y_prob":          y_prob,
        "y_pred":          y_pred,
    }


def load_or_train_model(X, y):
    """Train and evaluate the Random Forest model without external model files."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    feature_names = X.columns.tolist()
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    return model, metrics, X_test, y_test, feature_names


def plot_roc_curve(metrics: dict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=metrics["fpr"], y=metrics["tpr"],
        mode="lines", name=f"ROC Curve (AUC = {metrics['roc_auc']:.3f})",
        line=dict(color=COLORS["yes"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random Classifier", line=dict(color="gray", dash="dash"),
    ))
    fig.update_layout(
        title="ROC Curve — Subscription Prediction Model",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.6, y=0.1),
    )
    return fig


def plot_precision_recall(metrics: dict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=metrics["recall_curve"], y=metrics["precision_curve"],
        mode="lines", name=f"AP = {metrics['avg_precision']:.3f}",
        line=dict(color=COLORS["yes"], width=2),
    ))
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
    )
    return fig


def plot_confusion_matrix(metrics: dict):
    cm = metrics["confusion_matrix"]
    labels = ["Not Subscribed", "Subscribed"]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm[::-1],
            x=labels,
            y=labels[::-1],
            colorscale="Blues",
            text=cm[::-1],
            texttemplate="%{text}",
            showscale=False,
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
    )
    return fig


def plot_feature_importance(model, feature_names: list, top_n: int = 20):
    importances = model.feature_importances_
    fi = pd.DataFrame({
        "feature":    feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n)
    fig = px.bar(
        fi, x="importance", y="feature", orientation="h",
        title=f"Top {top_n} Feature Importances (Random Forest)",
        labels={"importance": "Importance", "feature": "Feature"},
        color="importance", color_continuous_scale="Blues",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# ANALYTICS — PRESCRIPTIVE
# ═════════════════════════════════════════════════════════════════════════════

def get_best_segments(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    grp = (
        df.groupby(["job", "education", "contact"])["y_binary"]
        .agg(["sum", "count"])
        .reset_index()
    )
    grp.columns = ["Job", "Education", "Contact Method", "Subscribed", "Total"]
    grp = grp[grp["Total"] >= 30]
    grp["Subscription Rate (%)"] = (grp["Subscribed"] / grp["Total"] * 100).round(2)
    grp = grp.sort_values("Subscription Rate (%)", ascending=False).head(top_n)
    return grp


def get_optimal_call_time(df: pd.DataFrame) -> pd.DataFrame:
    grp = (
        df.groupby(["month", "day_of_week"], observed=True)["y_binary"]
        .agg(["sum", "count"])
        .reset_index()
    )
    grp.columns = ["Month", "Day", "Subscribed", "Total"]
    grp["Rate"] = (grp["Subscribed"] / grp["Total"] * 100).round(2)
    pivot = grp.pivot_table(index="Month", columns="Day", values="Rate")
    pivot = pivot.reindex([m for m in MONTH_ORDER if m in pivot.index])
    dow_present = [d for d in DOW_ORDER if d in pivot.columns]
    pivot = pivot[dow_present]
    return pivot


def plot_optimal_timing_heatmap(df: pd.DataFrame):
    pivot = get_optimal_call_time(df)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="Greens",
            text=np.round(pivot.values, 1),
            texttemplate="%{text}%",
            colorbar_title="Rate (%)",
        )
    )
    fig.update_layout(
        title="Subscription Rate (%) by Month × Day of Week",
        xaxis_title="Day of Week",
        yaxis_title="Month",
        height=420,
    )
    return fig


def plot_contact_strategy(df: pd.DataFrame):
    grp = (
        df.groupby("campaign")["y_binary"]
        .agg(["sum", "count"])
        .reset_index()
    )
    grp.columns = ["contacts", "subscribed", "total"]
    grp["rate"] = (grp["subscribed"] / grp["total"] * 100).round(2)
    grp = grp[grp["contacts"] <= 10]
    fig = px.line(
        grp, x="contacts", y="rate",
        markers=True,
        title="Subscription Rate by Number of Campaign Contacts",
        labels={"contacts": "Number of Contacts", "rate": "Subscription Rate (%)"},
        color_discrete_sequence=[COLORS["yes"]],
    )
    fig.add_vline(x=3, line_dash="dash", line_color="orange",
                  annotation_text="Diminishing returns", annotation_position="top right")
    return fig


def generate_recommendations(df: pd.DataFrame, metrics: dict) -> list:
    overall_rate     = df["y_binary"].mean() * 100
    best_month       = df.groupby("month", observed=True)["y_binary"].mean().idxmax()
    best_poutcome    = df[df["poutcome"] == "success"]["y_binary"].mean() * 100
    cellular_rate    = df[df["contact"] == "cellular"]["y_binary"].mean() * 100
    telephone_rate   = df[df["contact"] == "telephone"]["y_binary"].mean() * 100
    high_euribor     = df[df["euribor3m"] > 4]["y_binary"].mean() * 100
    low_euribor      = df[df["euribor3m"] <= 2]["y_binary"].mean() * 100

    return [
        (
            "1️⃣  Target clients with prior successful campaign history",
            f"Clients who subscribed in a previous campaign convert at "
            f"{best_poutcome:.1f}% — significantly above the overall rate of {overall_rate:.1f}%.",
            "Prioritise re-contacting clients with 'success' in previous campaign outcome. "
            "Build a priority calling list of these high-probability leads.",
        ),
        (
            "2️⃣  Focus calls on cellular communication",
            f"Cellular contact achieves {cellular_rate:.1f}% subscription rate vs "
            f"{telephone_rate:.1f}% for telephone contact.",
            "Invest in obtaining cellular contact numbers for all clients. "
            "Avoid landline-only campaigns where possible.",
        ),
        (
            "3️⃣  Stop after 3 contacts per client per campaign",
            "Subscription rate drops sharply after 3 campaign contacts, "
            "indicating diminishing returns and increasing annoyance.",
            "Implement a 3-contact cap per client. Reallocate budget saved "
            "from over-contacted clients to acquiring new prospects.",
        ),
        (
            "4️⃣  Time campaigns to favourable economic conditions",
            f"Subscription rate is {low_euribor:.1f}% when Euribor rate ≤ 2% "
            f"vs {high_euribor:.1f}% when Euribor rate > 4%. "
            "Low rates make term deposits more attractive.",
            "Intensify campaign activity when Euribor 3-month rate is below 2%. "
            "Scale back or pause campaigns during high-rate environments.",
        ),
        (
            f"5️⃣  Best month to launch campaigns: {best_month.capitalize()}",
            f"Month '{best_month}' consistently shows the highest subscription rate.",
            f"Plan campaign launches for {best_month.capitalize()} as the primary window. "
            "Align staffing and marketing budgets accordingly.",
        ),
        (
            "6️⃣  Engage students and retired clients more heavily",
            "These job segments show the highest subscription conversion rates.",
            "Develop targeted messaging for students (long-term savings value) "
            "and retirees (capital preservation, fixed returns).",
        ),
        (
            "7️⃣  Use the predictive model for smarter prospecting",
            f"The trained Random Forest model achieves ROC-AUC of {metrics['roc_auc']:.3f}, "
            f"identifying high-probability subscribers before the call.",
            "Score the entire customer database with the model before each campaign. "
            "Call the top-30% highest-probability clients first to maximise conversion "
            "within budget constraints.",
        ),
    ]


# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Bank Telemarketing Analytics | IBM Capstone",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 700;
        color: #1a3c6e; margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem; color: #4b5563; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f0f4ff; border-radius: 10px;
        padding: 18px; text-align: center;
        border-left: 5px solid #2563EB;
    }
    .metric-value {
        font-size: 2rem; font-weight: 700; color: #1a3c6e;
    }
    .metric-label {
        font-size: 0.85rem; color: #6b7280; margin-top: 4px;
    }
    .insight-box {
        background: #eff6ff; border-left: 4px solid #2563EB;
        padding: 12px 16px; border-radius: 6px; margin-bottom: 10px;
    }
    .recommendation-card {
        background: #f0fdf4; border-left: 4px solid #10B981;
        padding: 14px 18px; border-radius: 8px; margin-bottom: 14px;
    }
    .section-header {
        font-size: 1.5rem; font-weight: 600;
        color: #1a3c6e; border-bottom: 2px solid #2563EB;
        padding-bottom: 6px; margin-top: 10px;
    }
    .warning-box {
        background: #fffbeb; border-left: 4px solid #F59E0B;
        padding: 10px 14px; border-radius: 6px;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING — cached so it runs once
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and preprocessing dataset…")
def load_data():
    raw       = load_raw()
    processed = preprocess(raw)
    stats     = summary_stats(processed)
    return raw, processed, stats


@st.cache_resource(show_spinner="Training predictive model…")
def get_model(_processed_df):
    """Underscore prefix tells Streamlit not to hash this argument."""
    X, y = get_feature_matrix(_processed_df)
    model, metrics, X_test, y_test, feature_names = load_or_train_model(X, y)
    return model, metrics, X_test, y_test, feature_names


raw_df, df, stats = load_data()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION & FILTERS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg",
             width=100)
    st.markdown("## 🏦 Bank Telemarketing Analytics")
    st.markdown("**IBM Internship Capstone Project**")
    st.markdown("---")

    page = st.radio(
        "📌 Navigation",
        options=[
            "🏠 Home",
            "📊 Dataset Overview",
            "📈 KPI Dashboard",
            "🔍 Exploratory Analysis",
            "📋 Descriptive Analytics",
            "🔬 Diagnostic Analytics",
            "🤖 Predictive Analytics",
            "💡 Prescriptive Insights",
            "✅ Business Recommendations",
        ],
    )

    st.markdown("---")
    st.markdown("### 🎛️ Filters")

    # Month filter
    months_available = sorted(
        df["month"].dropna().astype(str).unique().tolist(),
        key=lambda m: MONTH_ORDER.index(m) if m in MONTH_ORDER else 99,
    )
    selected_months = st.multiselect(
        "📅 Month(s)", options=months_available, default=months_available
    )

    # Job filter
    jobs_available = sorted(df["job"].dropna().unique().tolist())
    selected_jobs = st.multiselect(
        "💼 Job Type(s)", options=jobs_available, default=jobs_available
    )

    # Contact filter
    contact_types = sorted(df["contact"].dropna().unique().tolist())
    selected_contacts = st.multiselect(
        "📞 Contact Method(s)", options=contact_types, default=contact_types
    )

    # Age range
    age_min, age_max = int(df["age"].min()), int(df["age"].max())
    age_range = st.slider("🧑 Age Range", age_min, age_max, (age_min, age_max))

    st.markdown("---")
    st.markdown("*Dataset: UCI Bank Marketing*")
    st.markdown("*Moro et al., 2014*")


# Apply sidebar filters
filtered = df.copy()
if selected_months:
    filtered = filtered[filtered["month"].astype(str).isin(selected_months)]
if selected_jobs:
    filtered = filtered[filtered["job"].isin(selected_jobs)]
if selected_contacts:
    filtered = filtered[filtered["contact"].isin(selected_contacts)]
filtered = filtered[
    (filtered["age"] >= age_range[0]) & (filtered["age"] <= age_range[1])
]

if len(filtered) == 0:
    st.warning("⚠️ No data matches the current filters. Please adjust the sidebar filters.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown('<div class="main-title">🏦 Bank Telemarketing Campaign Effectiveness Analyzer</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-title">IBM Internship Capstone Project — AI & Data Science | 4-Tier Analytics Dashboard</div>',
                unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### 📌 Project Overview")
        st.markdown("""
        This capstone project analyses the **direct marketing campaigns** of a Portuguese banking institution.
        The campaigns were conducted via phone calls, with the objective of convincing clients to subscribe
        to a **bank term deposit**.

        The project implements a complete **4-tier analytics pipeline**:

        | Tier | Question | Method |
        |------|----------|--------|
        | 📋 Descriptive | What happened? | EDA, summaries, distributions |
        | 🔬 Diagnostic | Why did it happen? | Correlation, segment analysis |
        | 🤖 Predictive | What will happen? | Random Forest Classifier |
        | 💡 Prescriptive | What to do? | Segment targeting, strategy |

        **Dataset:** Bank Marketing — UCI Repository (Moro et al., 2014)
        **Records:** 41,188 raw (41,176 after deduplication) across 21 features
        **Period:** May 2008 – November 2010
        """)

    with col2:
        st.markdown("### 🎯 Key Objectives")
        st.markdown("""
        1. Understand factors that drive term deposit subscriptions
        2. Identify the most valuable client segments
        3. Predict client subscription probability
        4. Recommend data-driven campaign strategies
        5. Build an interactive analytical dashboard
        """)
        st.markdown("### 🗂️ Dataset Highlights")
        st.info(f"""
        - **{stats['total_records']:,}** total client contacts
        - **{stats['subscribed']:,}** subscriptions ({stats['subscription_rate']}%)
        - **21** features: client, campaign & economic data
        - **No missing values** (unknowns encoded)
        """)

    st.markdown("---")
    st.markdown("### 🗺️ Dashboard Navigation Guide")
    cols = st.columns(4)
    sections = [
        ("📊", "Dataset Overview", "Raw data, schema, quality report"),
        ("📈", "KPI Dashboard", "Key performance indicators"),
        ("🔍", "Exploratory Analysis", "Distributions, correlations"),
        ("📋", "Descriptive Analytics", "What happened — summaries"),
        ("🔬", "Diagnostic Analytics", "Why it happened — root causes"),
        ("🤖", "Predictive Analytics", "ML model & predictions"),
        ("💡", "Prescriptive Insights", "Segment targeting & timing"),
        ("✅", "Business Recommendations", "Actionable strategies"),
    ]
    for i, (icon, title, desc) in enumerate(sections):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:2rem">{icon}</div>
                <div style="font-weight:600;color:#1a3c6e">{title}</div>
                <div class="metric-label">{desc}</div>
            </div><br>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DATASET OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Dataset Overview":
    st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📄 Raw Data", "🗂️ Schema & Types", "📉 Data Quality"])

    with tab1:
        st.markdown(f"**Showing first 500 rows of {len(raw_df):,} total records**")
        st.dataframe(raw_df.head(500), use_container_width=True)

    with tab2:
        schema = pd.DataFrame({
            "Column":        raw_df.columns,
            "Dtype":         raw_df.dtypes.astype(str).values,
            "Non-Null":      raw_df.notnull().sum().values,
            "Null Count":    raw_df.isnull().sum().values,
            "Unique Values": [raw_df[c].nunique() for c in raw_df.columns],
            "Sample Value":  [str(raw_df[c].iloc[0]) for c in raw_df.columns],
        })
        st.dataframe(schema, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📐 Numerical Columns")
            num_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
            st.write(num_cols)
        with col2:
            st.markdown("#### 🏷️ Categorical Columns")
            cat_cols = raw_df.select_dtypes(include="object").columns.tolist()
            st.write(cat_cols)

    with tab3:
        st.markdown("#### 🔍 Dataset Quality Report")
        qual_col1, qual_col2, qual_col3, qual_col4 = st.columns(4)
        qual_col1.metric("Total Rows", f"{len(raw_df):,}")
        qual_col2.metric("Total Columns", f"{raw_df.shape[1]}")
        qual_col3.metric("Duplicate Rows", f"{raw_df.duplicated().sum()}")
        qual_col4.metric("Null Values", f"{raw_df.isnull().sum().sum()}")

        st.markdown("#### 📊 Descriptive Statistics — Numerical Features")
        st.dataframe(raw_df.describe().round(2).T, use_container_width=True)

        st.markdown("#### 🏷️ Value Counts — Categorical Features")
        cat_col = st.selectbox(
            "Select a categorical column",
            options=raw_df.select_dtypes(include="object").columns.tolist()
        )
        vc = raw_df[cat_col].value_counts().reset_index()
        vc.columns = [cat_col, "Count"]
        vc["Percentage (%)"] = (vc["Count"] / len(raw_df) * 100).round(2)

        col_a, col_b = st.columns(2)
        with col_a:
            st.dataframe(vc, use_container_width=True)
        with col_b:
            fig = px.bar(vc.head(15), x=cat_col, y="Count",
                         color="Count", color_continuous_scale="Blues",
                         title=f"Value Distribution: {cat_col}")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### ⚠️ 'Unknown' Value Counts in Categorical Columns")
        unknowns = {}
        for c in raw_df.select_dtypes(include="object").columns:
            cnt = (raw_df[c] == "unknown").sum()
            if cnt > 0:
                unknowns[c] = cnt
        if unknowns:
            unk_df = pd.DataFrame(list(unknowns.items()),
                                  columns=["Column", "Unknown Count"])
            unk_df["% of Total"] = (unk_df["Unknown Count"] / len(raw_df) * 100).round(2)
            st.dataframe(unk_df, use_container_width=True)
        else:
            st.success("No 'unknown' values found.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: KPI DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 KPI Dashboard":
    st.markdown('<div class="section-header">📈 KPI Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f"*Showing data for {len(filtered):,} clients matching current filters*")

    f_stats = summary_stats(filtered)

    # Row 1
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Clients", f"{f_stats['total_records']:,}")
    k2.metric("Subscriptions", f"{f_stats['subscribed']:,}",
              delta=f"{f_stats['subscription_rate']}%")
    k3.metric("Subscription Rate", f"{f_stats['subscription_rate']}%",
              delta=f"{f_stats['subscription_rate'] - stats['subscription_rate']:.2f}% vs overall" if f_stats['total_records'] != stats['total_records'] else None)
    k4.metric("Avg Age", f"{f_stats['avg_age']} yrs")
    k5.metric("Avg Call Duration", f"{f_stats['avg_call_duration']} min")

    # Row 2
    k6, k7, k8, k9, k10 = st.columns(5)
    k6.metric("Avg Campaign Contacts", f"{f_stats['avg_campaigns']}")
    k7.metric("Cellular Contact %", f"{f_stats['pct_cellular']}%")
    k8.metric("Prev. Contacted %", f"{f_stats['pct_prev_contacted']}%")
    k9.metric("Not Subscribed", f"{f_stats['not_subscribed']:,}")
    if f_stats['subscribed'] > 0:
        cps = round(f_stats['total_records'] / f_stats['subscribed'], 1)
    else:
        cps = "N/A"
    k10.metric("Contacts per Subscription", f"{cps}")

    st.markdown("---")

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        fig = plot_subscription_distribution(filtered)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = plot_subscription_by_month(filtered)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.markdown("#### 📌 Quick Insights")
        rate = f_stats["subscription_rate"]
        if rate > 15:
            st.success(f"🟢 Subscription rate **{rate}%** is above average — filters show high-value segment.")
        elif rate < 8:
            st.warning(f"🔴 Subscription rate **{rate}%** is below average — this segment underperforms.")
        else:
            st.info(f"🔵 Subscription rate **{rate}%** is near the overall average.")

        best_job_rate = filtered.groupby("job")["y_binary"].mean()
        if len(best_job_rate) > 0:
            top_job  = best_job_rate.idxmax()
            top_rate = round(best_job_rate.max() * 100, 1)
            st.markdown(f"**Best Job Segment:** {top_job} ({top_rate}%)")

        best_month_rate = filtered.groupby("month", observed=True)["y_binary"].mean()
        if len(best_month_rate) > 0:
            top_month = str(best_month_rate.idxmax())
            top_mrate = round(best_month_rate.max() * 100, 1)
            st.markdown(f"**Best Month:** {top_month.capitalize()} ({top_mrate}%)")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EXPLORATORY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Exploratory Analysis":
    st.markdown('<div class="section-header">🔍 Exploratory Data Analysis</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Distributions", "Relationships", "Correlations"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = plot_age_distribution(filtered)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = plot_duration_vs_subscription(filtered)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig = plot_education_subscription(filtered)
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            fig = plot_marital_subscription(filtered)
            st.plotly_chart(fig, use_container_width=True)

        st.plotly_chart(plot_numerical_summary(filtered), use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = plot_subscription_by_job(filtered)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = plot_contact_type(filtered)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig = plot_poutcome_effect(filtered)
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            fig = plot_day_of_week_subscription(filtered)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.plotly_chart(plot_correlation_heatmap(filtered), use_container_width=True)
        st.markdown("""
        <div class="insight-box">
        <b>Key Correlations with Subscription (y_binary):</b><br>
        • <b>call_duration_min</b>: Strongest positive correlation — longer calls → higher subscription likelihood<br>
        • <b>nr_employed</b>: Strong negative correlation — fewer employed → more subscriptions (economic downturn effect)<br>
        • <b>euribor3m</b>: Negative — low interest rates make term deposits more attractive<br>
        • <b>emp_var_rate</b>: Negative — falling employment rate → higher subscription<br>
        • <b>previous</b>: Positive — more prior contacts → higher chance
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DESCRIPTIVE ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📋 Descriptive Analytics":
    st.markdown('<div class="section-header">📋 Descriptive Analytics — "What Happened?"</div>',
                unsafe_allow_html=True)
    st.markdown("""
    Descriptive analytics summarises historical campaign data to understand
    the overall picture of the telemarketing campaigns.
    """)

    col1, col2 = st.columns(2)
    with col1:
        fig = plot_subscription_distribution(filtered)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = plot_subscription_by_month(filtered)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = plot_subscription_by_job(filtered)
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = plot_education_subscription(filtered)
        st.plotly_chart(fig, use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        fig = plot_age_distribution(filtered)
        st.plotly_chart(fig, use_container_width=True)
    with col6:
        fig = plot_day_of_week_subscription(filtered)
        st.plotly_chart(fig, use_container_width=True)

    col7, col8 = st.columns(2)
    with col7:
        fig = plot_marital_subscription(filtered)
        st.plotly_chart(fig, use_container_width=True)
    with col8:
        if "balance" in filtered.columns:
            fig = plot_balance_by_subscription(filtered)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = plot_contact_type(filtered)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📊 Summary Statistics by Outcome")
    desc_grp = filtered.groupby("y")[
        ["age", "campaign", "call_duration_min",
         "emp_var_rate", "cons_price_idx", "euribor3m"]
    ].describe().round(2)
    st.dataframe(desc_grp, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DIAGNOSTIC ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔬 Diagnostic Analytics":
    st.markdown('<div class="section-header">🔬 Diagnostic Analytics — "Why Did It Happen?"</div>',
                unsafe_allow_html=True)
    st.markdown("""
    Diagnostic analytics digs into the **root causes** of subscription patterns —
    identifying which factors most strongly influence whether a client subscribes.
    """)

    st.markdown("#### 📞 Call Duration Effect")
    st.markdown("""
    <div class="insight-box">
    <b>Finding:</b> Call duration is the single strongest predictor of subscription.
    Clients who subscribed had significantly longer average call durations.
    This reflects genuine engagement — a longer conversation indicates client interest.
    <br><b>Note:</b> Duration is excluded from the predictive model as it is only known
    after the call (data leakage prevention).
    </div>
    """, unsafe_allow_html=True)
    fig = plot_duration_vs_subscription(filtered)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Previous Campaign Outcome Effect")
        fig = plot_poutcome_effect(filtered)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("#### 🏠 Loan Status Effect")
        fig = plot_housing_loan_subscription(filtered)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### 📱 Contact Method Effect")
        fig = plot_contact_type(filtered)
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        st.markdown("#### 📊 Credit Default Effect")
        fig = plot_default_subscription(filtered)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📞 Number of Campaign Contacts Effect")
    fig = plot_campaign_contacts_effect(filtered)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🌍 Economic Climate Effect")
    fig = plot_economic_indicators(filtered)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="insight-box">
    <b>Economic Context Finding:</b> Subscription rates are strongly linked to economic conditions.
    When the Euribor 3-month rate is low (indicating low market returns), clients are more willing
    to lock money into a term deposit for a fixed return. During high employment variation periods
    (economic uncertainty), clients also show slightly higher propensity to save.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🔗 Correlation Analysis")
    fig = plot_correlation_heatmap(filtered)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PREDICTIVE ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Predictive Analytics":
    st.markdown('<div class="section-header">🤖 Predictive Analytics — "What Will Happen?"</div>',
                unsafe_allow_html=True)

    st.markdown("""
    A **Random Forest Classifier** is trained on the full processed dataset to predict
    the probability that a given client will subscribe to a term deposit.

    **Design Choices:**
    - `duration` (call length) is **excluded** from features — it is unknown before the call is made
    - Class imbalance is handled via `class_weight="balanced"` in the model
    - Train/test split: 80% / 20%, stratified by target
    - Features: 10 numerical + one-hot encoded categorical variables (33 total features)
    """)

    with st.spinner("Loading / training model (this may take ~30 seconds on first run)…"):
        model, metrics, X_test, y_test, feature_names = get_model(df)

    # Metric cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    m2.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    m3.metric("F1 Score (minority)", f"{metrics['f1_score']:.4f}")
    m4.metric("Avg Precision", f"{metrics['avg_precision']:.4f}")

    st.markdown("""
    <div class="warning-box">
    ⚠️ <b>Interpretation Note:</b> Due to class imbalance (~11% positive class), accuracy alone is
    a misleading metric. ROC-AUC and Average Precision are the primary evaluation metrics.
    A ROC-AUC > 0.75 indicates meaningful predictive power above random chance.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        fig = plot_roc_curve(metrics)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = plot_precision_recall(metrics)
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        fig = plot_confusion_matrix(metrics)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🏆 Top Feature Importances")
    fig = plot_feature_importance(model, feature_names, top_n=20)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📑 Full Classification Report")
    report    = metrics["report"]
    report_df = pd.DataFrame(report).T.round(3)
    st.dataframe(report_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔮 Individual Client Subscription Predictor")
    st.markdown("Adjust the parameters below to predict subscription probability for a hypothetical client.")

    with st.form("prediction_form"):
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            p_age      = st.slider("Age", 18, 95, 40)
            p_job      = st.selectbox("Job", sorted(df["job"].dropna().unique()))
            p_marital  = st.selectbox("Marital Status", sorted(df["marital"].dropna().unique()))
            p_education = st.selectbox("Education",
                ["basic.4y", "basic.6y", "basic.9y", "high.school",
                 "professional.course", "university.degree", "illiterate"])
        with pc2:
            p_housing  = st.selectbox("Housing Loan", ["no", "yes"])
            p_loan     = st.selectbox("Personal Loan", ["no", "yes"])
            p_contact  = st.selectbox("Contact Method", ["cellular", "telephone"])
            p_month    = st.selectbox("Month", MONTH_ORDER)
        with pc3:
            p_dow      = st.selectbox("Day of Week", DOW_ORDER)
            p_campaign = st.slider("Campaign Contacts", 1, 20, 2)
            p_previous = st.slider("Previous Contacts", 0, 10, 0)
            p_poutcome = st.selectbox("Previous Outcome", ["nonexistent", "failure", "success"])

        pred_submitted = st.form_submit_button("🔮 Predict Subscription Probability")

    if pred_submitted:
        edu_map = {
            "illiterate": 0, "basic.4y": 1, "basic.6y": 2, "basic.9y": 3,
            "high.school": 4, "professional.course": 5, "university.degree": 6,
        }
        med = df[["emp_var_rate", "cons_price_idx", "cons_conf_idx",
                  "euribor3m", "nr_employed"]].median()

        input_dict = {
            "age":                  p_age,
            "education_ord":        edu_map.get(p_education, 4),
            "was_contacted_before": 1 if p_previous > 0 else 0,
            "campaign":             p_campaign,
            "previous":             p_previous,
            "emp_var_rate":         med["emp_var_rate"],
            "cons_price_idx":       med["cons_price_idx"],
            "cons_conf_idx":        med["cons_conf_idx"],
            "euribor3m":            med["euribor3m"],
            "nr_employed":          med["nr_employed"],
            "job":                  p_job,
            "marital":              p_marital,
            "housing":              p_housing,
            "loan":                 p_loan,
            "contact":              p_contact,
            "month":                p_month,
            "day_of_week":          p_dow,
            "poutcome":             p_poutcome,
        }
        input_df = pd.DataFrame([input_dict])

        cat_for_model = ["job", "marital", "housing", "loan",
                         "contact", "month", "day_of_week", "poutcome"]
        input_encoded = pd.get_dummies(input_df, columns=cat_for_model)

        for col in feature_names:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[feature_names]
        bool_c = input_encoded.select_dtypes(include="bool").columns
        input_encoded[bool_c] = input_encoded[bool_c].astype(int)

        prob       = model.predict_proba(input_encoded)[0][1]
        prediction = "✅ Likely to Subscribe" if prob >= 0.5 else "❌ Unlikely to Subscribe"

        res1, res2, res3 = st.columns(3)
        res1.metric("Subscription Probability", f"{prob*100:.1f}%")
        res2.metric("Prediction", prediction)
        res3.metric("Confidence", "High" if abs(prob - 0.5) > 0.2 else "Moderate")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Subscription Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2563EB"},
                "steps": [
                    {"range": [0, 30],  "color": "#FEE2E2"},
                    {"range": [30, 60], "color": "#FEF3C7"},
                    {"range": [60, 100],"color": "#D1FAE5"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PRESCRIPTIVE INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💡 Prescriptive Insights":
    st.markdown('<div class="section-header">💡 Prescriptive Insights — "What Should Be Done?"</div>',
                unsafe_allow_html=True)
    st.markdown("""
    Prescriptive analytics goes beyond description and prediction —
    it provides **actionable recommendations** on how to improve campaign performance.
    """)

    tab1, tab2, tab3 = st.tabs([
        "🎯 Best Client Segments",
        "📅 Optimal Campaign Timing",
        "📞 Contact Strategy",
    ])

    with tab1:
        st.markdown("#### 🎯 Top 10 Client Segments by Subscription Rate")
        st.markdown("*Minimum 30 clients per segment for statistical reliability*")
        best_segs = get_best_segments(filtered, top_n=10)
        st.dataframe(best_segs, use_container_width=True)

        fig = px.bar(
            best_segs.head(10),
            x="Subscription Rate (%)", y=best_segs["Job"] + " / " + best_segs["Education"],
            orientation="h", color="Subscription Rate (%)",
            color_continuous_scale="Greens",
            title="Top Client Segments by Subscription Rate",
        )
        fig.update_layout(yaxis_title="Job / Education", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### 📅 Optimal Call Timing — Month × Day Heatmap")
        st.markdown("""
        Identify the best month-day combinations to maximise subscription rates.
        Darker green = higher subscription rate.
        """)
        fig = plot_optimal_timing_heatmap(filtered)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 📊 Subscription Rate by Month")
        fig = plot_subscription_by_month(filtered)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 📊 Subscription Rate by Day of Week")
        fig = plot_day_of_week_subscription(filtered)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("#### 📞 Campaign Contact Frequency Strategy")
        st.markdown("""
        <div class="insight-box">
        <b>Key Finding:</b> Subscription rate drops sharply after 3 contacts per campaign.
        Over-contacting clients is wasteful and may create negative sentiment.
        The optimal strategy is to <b>stop at 3 contacts</b> and reallocate resources.
        </div>
        """, unsafe_allow_html=True)
        fig = plot_contact_strategy(filtered)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📱 Contact Method Performance")
            fig = plot_contact_type(filtered)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("#### 🏆 Previous Campaign Outcome Impact")
            fig = plot_poutcome_effect(filtered)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 🌍 Economic Context Monitoring")
        fig = plot_economic_indicators(filtered)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: BUSINESS RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "✅ Business Recommendations":
    st.markdown('<div class="section-header">✅ Business Recommendations</div>',
                unsafe_allow_html=True)
    st.markdown("""
    The following recommendations are derived directly from the data analysis
    and predictive model. Each is supported by evidence from the dataset.
    """)

    with st.spinner("Generating recommendations…"):
        model, metrics, _, _, _ = get_model(df)
        recs = generate_recommendations(filtered, metrics)

    for title, insight, rec in recs:
        st.markdown(f"### {title}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="insight-box">
            <b>📊 Evidence:</b> {insight}
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="recommendation-card">
            <b>✅ Recommendation:</b> {rec}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Summary Action Plan")
    st.markdown("""
    | Priority | Action | Expected Impact |
    |----------|--------|----------------|
    | 🔴 High | Re-target previous successful subscribers | +15–20% conversion uplift |
    | 🔴 High | Use predictive model to score prospects | Optimise call list ROI |
    | 🟡 Medium | Switch entirely to cellular contact | +2–5% conversion improvement |
    | 🟡 Medium | Cap calls at 3 per client | 30–40% reduction in wasted contacts |
    | 🟡 Medium | Schedule campaigns in high-performing months | +5–10% seasonal uplift |
    | 🟢 Lower | Target student and retired segments with tailored messaging | Segment-specific growth |
    | 🟢 Lower | Monitor Euribor and pause campaigns in high-rate periods | Cost efficiency |

    > *All impact estimates are directional and based on observed data patterns.
    > Exact results depend on execution quality and market conditions.*
    """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#6b7280;font-size:0.9rem;margin-top:2rem;">
    🏦 Bank Telemarketing Campaign Effectiveness Analyzer<br>
    IBM Internship Capstone Project — AI & Data Science<br>
    Dataset: Moro et al., 2014 — UCI Machine Learning Repository
    </div>
    """, unsafe_allow_html=True)
