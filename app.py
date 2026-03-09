"""
MSIS 522 HW1 — Streamlit App
Household Electric Power Consumption — Regression Analysis
"""

import os, json, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib
import shap
import streamlit as st

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Household Power Consumption | MSIS 522 HW1",
    page_icon="⚡",
    layout="wide",
)

MODELS_DIR = "models"

# ── Load artifacts (cached) ──────────────────────────────────────────────────
@st.cache_resource
def load_all():
    data = {}
    data["split"]    = joblib.load(os.path.join(MODELS_DIR, "train_test_split.pkl"))
    data["scaler"]   = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    data["lr"]       = joblib.load(os.path.join(MODELS_DIR, "linear_regression.pkl"))
    data["dt"]       = joblib.load(os.path.join(MODELS_DIR, "decision_tree.pkl"))
    data["rf"]       = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    data["xgb"]      = joblib.load(os.path.join(MODELS_DIR, "xgboost.pkl"))

    with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
        data["meta"] = json.load(f)
    with open(os.path.join(MODELS_DIR, "best_params.json")) as f:
        data["best_params"] = json.load(f)
    with open(os.path.join(MODELS_DIR, "mlp_history.json")) as f:
        data["mlp_history"] = json.load(f)

    data["metrics_df"] = pd.read_csv(os.path.join(MODELS_DIR, "model_metrics.csv"))
    data["sample_df"]  = pd.read_parquet(os.path.join(MODELS_DIR, "sample_data.parquet"))
    data["shap_values"]  = np.load(os.path.join(MODELS_DIR, "shap_values.npy"))
    data["shap_sample"]  = joblib.load(os.path.join(MODELS_DIR, "shap_sample.pkl"))
    data["shap_explainer"] = joblib.load(os.path.join(MODELS_DIR, "shap_explainer.pkl"))

    # predictions
    data["pred_lr"]  = np.load(os.path.join(MODELS_DIR, "pred_lr.npy"))
    data["pred_dt"]  = np.load(os.path.join(MODELS_DIR, "pred_dt.npy"))
    data["pred_rf"]  = np.load(os.path.join(MODELS_DIR, "pred_rf.npy"))
    data["pred_xgb"] = np.load(os.path.join(MODELS_DIR, "pred_xgb.npy"))
    data["pred_mlp"] = np.load(os.path.join(MODELS_DIR, "pred_mlp.npy"))

    # Load MLP
    import tensorflow as tf
    data["mlp"] = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, "mlp_model.h5")
    )
    return data

art = load_all()
meta        = art["meta"]
split       = art["split"]
sample_df   = art["sample_df"]
metrics_df  = art["metrics_df"]
best_params = art["best_params"]
scaler      = art["scaler"]
FEATURES    = meta["features"]
TARGET      = meta["target"]
X_test      = split["X_test"]
y_test      = split["y_test"]

MODEL_MAP = {
    "Linear Regression": art["lr"],
    "Decision Tree":     art["dt"],
    "Random Forest":     art["rf"],
    "XGBoost":           art["xgb"],
    "MLP Neural Network": art["mlp"],
}

PRED_MAP = {
    "Linear Regression":  art["pred_lr"],
    "Decision Tree":      art["pred_dt"],
    "Random Forest":      art["pred_rf"],
    "XGBoost":            art["pred_xgb"],
    "MLP Neural Network": art["pred_mlp"],
}

# ── Colour palette ─────────────────────────────────────────────────────────
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
MODEL_COLORS = dict(zip(MODEL_MAP.keys(), PALETTE))

# ── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Summary",
    "Descriptive Analytics",
    "Model Performance",
    "Explainability & Interactive Prediction",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE SUMMARY
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("⚡ Household Electric Power Consumption")
    st.subheader("MSIS 522 HW1 — Complete Data Science Workflow")

    st.markdown("""
    ---
    ### Dataset Overview
    The **Individual Household Electric Power Consumption** dataset (UCI ML Repository) records
    **minute-by-minute** electrical measurements collected from a single household in Sceaux, France,
    between **December 2006 and November 2010** — nearly 4 years of high-resolution data.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Observations", f"{meta['n_total']:,}")
    col2.metric("Sampled for Training", f"{meta['n_sample']:,}")
    col3.metric("Features Used", len(FEATURES))
    col4.metric("Prediction Target", "Global Active Power")

    st.markdown("""
    ---
    ### Prediction Task
    We tackle a **regression** problem: predict `Global_active_power` (kilowatts),
    the total minute-averaged power drawn by the household. This is a fundamental
    signal that captures overall energy demand.

    **Why this matters:**
    - Accurate power forecasting enables utilities to balance the grid, reduce waste, and
      plan infrastructure investment.
    - At the household level, understanding which factors drive consumption supports
      demand-response programs and energy-efficiency audits.
    - With the rise of smart meters and dynamic tariffs, real-time consumption prediction
      has direct financial value to consumers.

    ---
    ### Key Findings
    """)

    best_row = metrics_df.loc[metrics_df["RMSE"].idxmin()]
    st.markdown(f"""
    | Finding | Detail |
    |---------|--------|
    | **Best model** | **{best_row['Model']}** with RMSE = {best_row['RMSE']:.4f} kW and R² = {best_row['R2']:.4f} |
    | **Top features** | `Global_intensity`, `Sub_metering_3`, and `Global_reactive_power` dominate predictions (SHAP) |
    | **Time patterns** | Consumption peaks in evening hours (18–22h) and is lower on weekends |
    | **Data quality** | ~1.25% missing values (marked `?`) — dropped before modeling |
    | **All models** | Every model achieves R² > 0.998, indicating strong predictability |

    ---
    ### Approach
    1. **Part 1 — Descriptive Analytics**: EDA with target/feature distributions, correlation heatmap.
    2. **Part 2 — Predictive Modeling**: Linear Regression baseline → Decision Tree → Random Forest →
       XGBoost → MLP Neural Network, each tuned with 5-fold GridSearchCV.
    3. **Part 3 — Explainability**: SHAP TreeExplainer on XGBoost to reveal feature impacts.
    4. **Part 4 — Deployment**: This Streamlit app loads pre-trained models for interactive use.
    """)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — DESCRIPTIVE ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Part 1: Descriptive Analytics")

    # ── 1.1 Dataset Intro ─────────────────────────────────────────────────
    st.subheader("1.1 Dataset Introduction")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Features used for modeling:**
        | Feature | Description |
        |---------|-------------|
        | `Global_reactive_power` | Reactive power (kVAR) |
        | `Voltage` | Minute-average voltage (V) |
        | `Global_intensity` | Current intensity (A) |
        | `Sub_metering_1` | Kitchen energy (Wh) |
        | `Sub_metering_2` | Laundry energy (Wh) |
        | `Sub_metering_3` | Water heater / AC (Wh) |
        | `hour` | Hour of day (0–23) |
        | `day_of_week` | Day (0=Mon … 6=Sun) |
        | `month` | Month (1–12) |
        | `is_weekend` | 1 if Sat/Sun, else 0 |
        """)
    with col_b:
        st.markdown(f"""
        **Basic statistics (training sample, n = {meta['n_sample']:,}):**
        """)
        desc = sample_df[FEATURES + [TARGET]].describe().round(3)
        st.dataframe(desc, use_container_width=True)

    st.divider()

    # ── 1.2 Target Distribution ───────────────────────────────────────────
    st.subheader("1.2 Target Distribution — Global Active Power (kW)")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(sample_df[TARGET], bins=60, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Global Active Power (kW)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Histogram of Global Active Power")

    sample_df[TARGET].plot.kde(ax=axes[1], color="#C44E52", linewidth=2)
    axes[1].set_xlabel("Global Active Power (kW)")
    axes[1].set_title("KDE of Global Active Power")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption("""
    **Interpretation:** The target is right-skewed with a large spike near 0 kW (idle/standby) and a long tail
    reaching ~10 kW (heavy appliance use). The bulk of readings fall below 2 kW, meaning most minutes
    the household draws modest power; extreme spikes correspond to simultaneous high-demand appliances.
    No class imbalance concern for regression, but the skew motivates tree-based models that handle it natively.
    """)

    st.divider()

    # ── 1.3 Feature Distributions ─────────────────────────────────────────
    st.subheader("1.3 Feature Distributions and Relationships")

    # Plot 1: Hourly average consumption
    st.markdown("**Plot 1: Average Power Consumption by Hour of Day**")
    hourly = sample_df.groupby("hour")[TARGET].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(hourly["hour"], hourly[TARGET], color="#4C72B0", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Global Active Power (kW)")
    ax.set_title("Hourly Mean Power Consumption")
    ax.set_xticks(range(0, 24))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("""
    **Interpretation:** Consumption is lowest in the early morning hours (2–6 AM) and rises sharply
    in the evening (18–22 h), reflecting post-work cooking, lighting, and entertainment loads.
    This clear diurnal pattern means `hour` is a strong predictive feature.
    """)

    # Plot 2: Boxplot by month
    st.markdown("**Plot 2: Global Active Power Distribution by Month**")
    fig, ax = plt.subplots(figsize=(12, 4))
    month_order = list(range(1, 13))
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    sample_df.boxplot(column=TARGET, by="month", ax=ax, notch=True,
                      patch_artist=True,
                      boxprops=dict(facecolor="#4C72B0", alpha=0.6),
                      medianprops=dict(color="red", linewidth=2),
                      flierprops=dict(marker=".", markersize=2, alpha=0.3))
    ax.set_xticklabels(month_labels)
    ax.set_xlabel("Month")
    ax.set_ylabel("Global Active Power (kW)")
    ax.set_title("Power Consumption by Month")
    plt.suptitle("")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("""
    **Interpretation:** Winter months (Dec–Feb) show noticeably higher median consumption and wider
    spread than summer months, consistent with heating and longer nights driving additional demand.
    Summer months (Jun–Aug) have the lowest medians, reflecting mild French weather.
    """)

    # Plot 3: Weekday vs Weekend violin
    st.markdown("**Plot 3: Power Consumption — Weekday vs Weekend**")
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_data = sample_df.copy()
    plot_data["Day Type"] = plot_data["is_weekend"].map({0: "Weekday", 1: "Weekend"})
    sns.violinplot(data=plot_data, x="Day Type", y=TARGET, palette=["#4C72B0","#DD8452"],
                   inner="quartile", ax=ax)
    ax.set_ylabel("Global Active Power (kW)")
    ax.set_title("Weekday vs. Weekend Power Distribution")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("""
    **Interpretation:** Weekends exhibit a slightly higher median and heavier upper tail compared to weekdays.
    This suggests occupants are home more during weekends, running more appliances throughout the day
    rather than concentrating use in morning and evening commuter windows.
    """)

    # Plot 4: Sub-metering breakdown (stacked bar by hour)
    st.markdown("**Plot 4: Sub-metering Energy Breakdown by Hour**")
    sub_hourly = sample_df.groupby("hour")[
        ["Sub_metering_1","Sub_metering_2","Sub_metering_3"]
    ].mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    sub_hourly.plot(kind="bar", stacked=True, ax=ax,
                    color=["#4C72B0","#DD8452","#55A868"],
                    alpha=0.85, edgecolor="white")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Sub-metering (Wh)")
    ax.set_title("Average Sub-metering Energy by Hour")
    ax.legend(["Kitchen (SM1)","Laundry (SM2)","Water Heater/AC (SM3)"])
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("""
    **Interpretation:** Sub_metering_3 (water heater / AC) dominates across all hours, peaking in
    the morning (~7 AM) and evening (~20 h). Sub_metering_1 (kitchen) spikes around mealtimes
    (8 AM, noon, 7 PM). Laundry (SM2) usage is relatively flat, suggesting habitual scheduling.
    These patterns explain why SM3 and SM1 are top SHAP contributors.
    """)

    # Plot 5: Scatter — Global_intensity vs Target
    st.markdown("**Plot 5: Global Intensity vs. Global Active Power**")
    fig, ax = plt.subplots(figsize=(7, 4))
    sample_sub = sample_df.sample(2000, random_state=42)
    ax.scatter(sample_sub["Global_intensity"], sample_sub[TARGET],
               alpha=0.3, s=10, color="#4C72B0")
    ax.set_xlabel("Global Intensity (A)")
    ax.set_ylabel("Global Active Power (kW)")
    ax.set_title("Current Intensity vs. Active Power (sample n=2,000)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("""
    **Interpretation:** There is a near-perfect linear relationship between Global_intensity and
    Global_active_power, which is expected from Ohm's law (P ≈ V × I). This single feature
    almost fully determines the target, making it the dominant predictor across all models.
    """)

    st.divider()

    # ── 1.4 Correlation Heatmap ────────────────────────────────────────────
    st.subheader("1.4 Correlation Heatmap")
    corr_cols = FEATURES + [TARGET]
    corr = sample_df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
                annot_kws={"size": 8})
    ax.set_title("Pearson Correlation Matrix")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("""
    **Interpretation:** `Global_intensity` has a near-perfect positive correlation with
    `Global_active_power` (r ≈ 0.99), confirming the physics-based relationship (P = V·I).
    `Global_reactive_power` is moderately correlated (r ≈ 0.63), while sub-metering
    variables show weaker but meaningful correlations. Voltage is slightly negatively
    correlated with active power — higher voltage at low load is a known grid phenomenon.
    """)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Part 2: Model Performance")

    # ── 2.7 Comparison Table ──────────────────────────────────────────────
    st.subheader("2.7 Model Comparison Summary")
    st.dataframe(
        metrics_df.style.highlight_min(subset=["MAE","RMSE"], color="#c8f7c5")
                        .highlight_max(subset=["R2"], color="#c8f7c5")
                        .format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R2": "{:.4f}"}),
        use_container_width=True,
    )

    # Bar chart — RMSE comparison
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [MODEL_COLORS[m] for m in metrics_df["Model"]]
    bars = ax.bar(metrics_df["Model"], metrics_df["RMSE"], color=colors, edgecolor="white", alpha=0.85)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_ylabel("RMSE (kW)")
    ax.set_title("RMSE Comparison Across Models")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    **Discussion:** XGBoost achieves the lowest RMSE (0.0350 kW) and highest R² (0.9989),
    outperforming all other models. This is expected: gradient-boosted trees excel at
    capturing nonlinear relationships with minimal preprocessing. The MLP performs comparably
    (RMSE 0.0375), showing that neural networks can match tree ensembles given sufficient
    architecture design. Linear Regression provides a remarkably strong baseline (R² = 0.9986)
    because `Global_intensity` is almost linearly related to the target. The Decision Tree
    underperforms the ensemble methods as expected, but still achieves excellent R².

    **Trade-offs:** Linear Regression is fully interpretable but slightly weaker; XGBoost is
    near-optimal but is a black box (mitigated by SHAP); MLP is flexible but requires scaled
    inputs and more training time; Decision Tree is fast and visualisable but less accurate.
    """)

    st.divider()

    # ── Predicted vs Actual plots ─────────────────────────────────────────
    st.subheader("Predicted vs. Actual (all models)")
    y_test_arr = y_test.values

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    for ax, (mname, preds) in zip(axes, PRED_MAP.items()):
        # Use same-length arrays (LR/DT use full X_test, MLP may differ)
        n = min(len(y_test_arr), len(preds))
        ax.scatter(y_test_arr[:n], preds[:n], alpha=0.15, s=5,
                   color=MODEL_COLORS[mname])
        lims = [y_test_arr[:n].min(), y_test_arr[:n].max()]
        ax.plot(lims, lims, "r--", lw=1)
        ax.set_title(mname, fontsize=9)
        ax.set_xlabel("Actual (kW)", fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel("Predicted (kW)", fontsize=8)
    plt.suptitle("Predicted vs. Actual Power Consumption", fontsize=12, y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Points on the red diagonal line indicate perfect predictions. All models cluster tightly around it, confirming high predictive accuracy.")

    st.divider()

    # ── Best Hyperparameters ──────────────────────────────────────────────
    st.subheader("Best Hyperparameters per Model")
    for mname, params in best_params.items():
        with st.expander(mname):
            st.json(params)

    st.divider()

    # ── MLP Training History ───────────────────────────────────────────────
    st.subheader("2.6 MLP Neural Network — Training History")
    hist = art["mlp_history"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist["loss"], label="Train Loss", color="#4C72B0")
    axes[0].plot(hist["val_loss"], label="Val Loss", color="#DD8452")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training & Validation Loss"); axes[0].legend()

    axes[1].plot(hist["mae"], label="Train MAE", color="#4C72B0")
    axes[1].plot(hist["val_mae"], label="Val MAE", color="#DD8452")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MAE (kW)")
    axes[1].set_title("Training & Validation MAE"); axes[1].legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("The MLP converges quickly (within ~10 epochs) with train and validation curves tracking closely, indicating no significant overfitting.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPLAINABILITY & INTERACTIVE PREDICTION
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Part 3 & 4: Explainability & Interactive Prediction")

    # ── SHAP Summary ──────────────────────────────────────────────────────
    st.subheader("3.1 SHAP Analysis — XGBoost (Best Model)")

    shap_values = art["shap_values"]
    shap_sample = art["shap_sample"]

    # Bar plot of mean absolute SHAP
    st.markdown("**SHAP Feature Importance (Mean |SHAP|)**")
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({"Feature": FEATURES, "Mean |SHAP|": mean_shap})
    shap_df = shap_df.sort_values("Mean |SHAP|", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(shap_df["Feature"], shap_df["Mean |SHAP|"],
                   color="#4C72B0", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Mean |SHAP value| (impact on prediction)")
    ax.set_title("XGBoost SHAP Feature Importance")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption("""
    **Interpretation:** `Global_intensity` is by far the most important feature, consistent with
    the physics-based linear relationship (P = V·I). `Sub_metering_3` (water heater/AC) and
    `Global_reactive_power` rank second and third — they capture electrical characteristics
    not redundant with intensity alone. Time-based features (`hour`, `month`) add incremental
    value by capturing behavioral patterns.
    """)

    # SHAP beeswarm-style dot plot
    st.markdown("**SHAP Summary — Direction of Feature Impact**")
    fig, ax = plt.subplots(figsize=(9, 6))
    # Build beeswarm manually for full control
    sorted_idx = np.argsort(mean_shap)[::-1]
    feat_names = [FEATURES[i] for i in sorted_idx]
    for row_i, fi in enumerate(sorted_idx):
        sv = shap_values[:, fi]
        fv = shap_sample.iloc[:, fi].values
        fv_norm = (fv - fv.min()) / (fv.ptp() + 1e-9)
        scatter = ax.scatter(sv, np.full_like(sv, row_i) + np.random.uniform(-0.3, 0.3, len(sv)),
                             c=fv_norm, cmap="coolwarm", alpha=0.4, s=8, vmin=0, vmax=1)
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_names)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title("SHAP Beeswarm Plot\n(color = feature value: blue=low, red=high)")
    plt.colorbar(scatter, ax=ax, label="Feature value (normalized)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption("""
    **Interpretation:** High values of `Global_intensity` (red dots, right side) push predictions up,
    while low values (blue dots, left side) pull them down — mirroring the direct physical relationship.
    For `hour`, high values (evening) push predictions up, confirming the peak-demand evening pattern.
    `Sub_metering_3` shows a wide spread, indicating it captures both low-use and high-use states.
    For decision-makers, these insights mean that current intensity and water heater / AC usage
    are the primary drivers of household power demand — targeted efficiency improvements in these
    areas would yield the greatest savings.
    """)

    st.divider()

    # ── Interactive Prediction ────────────────────────────────────────────
    st.subheader("Interactive Prediction")
    st.markdown("""
    Adjust the sliders below to set feature values, select a model, and see the
    predicted Global Active Power in real time. A SHAP waterfall explains the prediction
    (only available for tree-based models).
    """)

    feat_means = meta["feature_means"]
    feat_mins  = meta["feature_mins"]
    feat_maxs  = meta["feature_maxs"]

    col_inp, col_out = st.columns([1, 1])

    with col_inp:
        st.markdown("**Key feature inputs** (others use training mean):")
        selected_model = st.selectbox("Model", list(MODEL_MAP.keys()))

        global_intensity = st.slider(
            "Global Intensity (A)",
            min_value=float(feat_mins["Global_intensity"]),
            max_value=float(feat_maxs["Global_intensity"]),
            value=float(feat_means["Global_intensity"]),
            step=0.1,
        )
        voltage = st.slider(
            "Voltage (V)",
            min_value=float(max(feat_mins["Voltage"], 220.0)),
            max_value=float(min(feat_maxs["Voltage"], 255.0)),
            value=float(feat_means["Voltage"]),
            step=0.5,
        )
        global_reactive = st.slider(
            "Global Reactive Power (kVAR)",
            min_value=float(feat_mins["Global_reactive_power"]),
            max_value=float(feat_maxs["Global_reactive_power"]),
            value=float(feat_means["Global_reactive_power"]),
            step=0.01,
        )
        sub3 = st.slider(
            "Sub-metering 3 — Water Heater/AC (Wh)",
            min_value=float(feat_mins["Sub_metering_3"]),
            max_value=float(feat_maxs["Sub_metering_3"]),
            value=float(feat_means["Sub_metering_3"]),
            step=1.0,
        )
        hour = st.slider("Hour of Day", 0, 23, int(feat_means["hour"]))
        month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun",
                                    "Jul","Aug","Sep","Oct","Nov","Dec"][x-1],
            index=int(feat_means["month"]) - 1,
        )
        is_weekend = st.checkbox("Weekend?", value=False)

    # Build input vector — use means for non-displayed features
    user_input = {f: feat_means[f] for f in FEATURES}
    user_input["Global_intensity"]       = global_intensity
    user_input["Voltage"]                = voltage
    user_input["Global_reactive_power"]  = global_reactive
    user_input["Sub_metering_3"]         = sub3
    user_input["hour"]                   = float(hour)
    user_input["month"]                  = float(month)
    user_input["is_weekend"]             = float(int(is_weekend))

    X_user = pd.DataFrame([user_input])[FEATURES]

    # Predict
    model = MODEL_MAP[selected_model]
    if selected_model == "MLP Neural Network":
        X_user_scaled = scaler.transform(X_user)
        prediction = float(model.predict(X_user_scaled, verbose=0).flatten()[0])
    elif selected_model == "Linear Regression":
        X_user_scaled = scaler.transform(X_user)
        prediction = float(model.predict(X_user_scaled)[0])
    else:
        prediction = float(model.predict(X_user)[0])

    with col_out:
        st.markdown("**Prediction Result**")
        st.metric(
            label=f"Predicted Global Active Power [{selected_model}]",
            value=f"{prediction:.3f} kW",
            delta=f"{prediction - meta['y_mean']:.3f} kW vs. mean",
        )

        # Typical equivalents
        st.markdown(f"""
        **Context:**
        - Mean household consumption: **{meta['y_mean']:.2f} kW**
        - Your input predicts **{prediction:.3f} kW**
        - Equivalent to ≈ {prediction / 0.060:.0f} W LED bulbs running simultaneously
        - Over an hour: **{prediction:.3f} kWh** of energy consumed
        """)

        # SHAP waterfall (tree models only)
        if selected_model in ("Decision Tree", "Random Forest", "XGBoost"):
            st.markdown("**SHAP Waterfall — This Prediction**")
            try:
                explainer = art["shap_explainer"] if selected_model == "XGBoost" else \
                            shap.TreeExplainer(model)
                sv_user = explainer.shap_values(X_user)
                if isinstance(sv_user, list):
                    sv_user = sv_user[0]
                sv_user = sv_user.flatten()

                base_val = float(explainer.expected_value)
                contrib = list(zip(FEATURES, sv_user))
                contrib_sorted = sorted(contrib, key=lambda x: abs(x[1]), reverse=True)[:8]

                fig, ax = plt.subplots(figsize=(7, 5))
                feats_plot = [c[0] for c in contrib_sorted][::-1]
                vals_plot  = [c[1] for c in contrib_sorted][::-1]
                colors_wf  = ["#C44E52" if v > 0 else "#4C72B0" for v in vals_plot]
                ax.barh(feats_plot, vals_plot, color=colors_wf, edgecolor="white", alpha=0.85)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("SHAP value (kW)")
                ax.set_title(f"SHAP Waterfall\nBase value: {base_val:.3f} kW → Predicted: {prediction:.3f} kW")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Red bars push the prediction higher than the baseline; blue bars push it lower.")
            except Exception as e:
                st.warning(f"SHAP waterfall unavailable: {e}")
        else:
            st.info("SHAP waterfall is available for tree-based models (Decision Tree, Random Forest, XGBoost). Select one of those to see it.")

    st.divider()

    # Single-instance SHAP explanation from the test set
    st.subheader("Sample Prediction Explanation (from Test Set)")
    idx = st.slider("Test set sample index", 0, 499, 0)
    sample_row = art["shap_sample"].iloc[[idx]]
    sv_row = art["shap_values"][idx]
    base_val = float(art["shap_explainer"].expected_value)
    pred_val = base_val + sv_row.sum()

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(f"**Actual value:** {y_test.iloc[idx] if idx < len(y_test) else 'N/A':.3f} kW")
        st.markdown(f"**Predicted (XGBoost):** {pred_val:.3f} kW")
        st.dataframe(sample_row.T.rename(columns={sample_row.index[0]: "Value"}).round(3),
                     use_container_width=True)
    with col_s2:
        contrib_sorted = sorted(zip(FEATURES, sv_row), key=lambda x: abs(x[1]), reverse=True)[:8]
        feats_s = [c[0] for c in contrib_sorted][::-1]
        vals_s  = [c[1] for c in contrib_sorted][::-1]
        colors_s = ["#C44E52" if v > 0 else "#4C72B0" for v in vals_s]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(feats_s, vals_s, color=colors_s, edgecolor="white", alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value")
        ax.set_title(f"SHAP Waterfall — Sample #{idx}\nBase: {base_val:.3f} → Pred: {pred_val:.3f} kW")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>MSIS 522 HW1 · Foster School of Business · University of Washington · "
    "Dataset: UCI Individual Household Electric Power Consumption</small>",
    unsafe_allow_html=True,
)
