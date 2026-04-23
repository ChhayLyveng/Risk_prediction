"""
app.py — Illness Risk Prediction  ·  Streamlit Web App
=======================================================
Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os, io, json, csv, pathlib

import Risk_Prediction as rp   # model module

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Illness Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS – clean card look
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { font-size: 2.4rem; margin: 0; }
    .main-header p  { font-size: 1.05rem; opacity: .85; margin: .4rem 0 0; }

    .metric-card {
        background: #f0f6ff;
        border-left: 4px solid #2d6a9f;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: .8rem;
    }
    .risk-high {
        background: #fff0f0;
        border: 2px solid #e74c3c;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .risk-low {
        background: #f0fff4;
        border: 2px solid #27ae60;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .section-divider { border-top: 2px solid #e0e0e0; margin: 2rem 0 1.5rem; }
    .sidebar-tip { font-size:.82rem; color:#666; font-style:italic; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────
for key in ("model_trained", "theta", "mean_X", "std_X",
            "cost_history", "metrics", "df"):
    if key not in st.session_state:
        st.session_state[key] = None
if "feedbacks" not in st.session_state:
    st.session_state.feedbacks = []

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🏥 Illness Risk Predictor</h1>
  <p>A beginner-friendly Machine Learning app — Logistic Regression built from scratch with NumPy</p>
</div>
""", unsafe_allow_html=True)

st.subheader("How it works")
st.markdown("""
1. **Upload** your dataset (CSV) or use the built-in demo data.
2. The model **trains** a logistic regression classifier on the data.
3. Fill in your personal info in the **sidebar** and press *Predict*.
4. Explore **EDA charts** and leave **feedback** below.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Data source + user inputs
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/caduceus.png", width=72)
    st.title("🔧 Settings & Inputs")

    # ── Data ──────────────────────────────────────────────────────────────
    st.header("📂 Step 1 — Load Data")
    data_source = st.radio("Data source", ["Upload CSV", "Generate demo data"])

    df_raw = None

    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Choose raw_data.csv", type=["csv"])
        if uploaded:
            df_raw = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df_raw)} rows ✅")
    else:
        if st.button("🎲 Generate demo data"):
            np.random.seed(0)
            n = 300
            ages = np.random.randint(19, 24, n)
            sleep_labels = np.random.choice(list(rp.SLEEP_MAP.keys()), n)
            sleep_vals = [rp.SLEEP_MAP[s] for s in sleep_labels]
            water = np.random.randint(1, 5, n)
            stress = np.random.randint(1, 5, n)
            screen = np.random.randint(1, 8, n)
            illness_prob = (stress * 0.15 + (8 - np.array(sleep_vals)) * 0.1
                            + screen * 0.05 - water * 0.08)
            illness_prob = 1 / (1 + np.exp(-illness_prob + 1))
            illness = (np.random.rand(n) < illness_prob).astype(int)
            illness_labels = ["Yes" if i == 1 else "No" for i in illness]
            df_raw = pd.DataFrame({
                "age": ages,
                "sleep_duration": sleep_labels,
                "water_intake": water,
                "stress": stress,
                "screentime": screen,
                "illness": illness_labels,
            })
            st.success(f"Generated {n} rows ✅")

    # ── Train ─────────────────────────────────────────────────────────────
    st.header("🚀 Step 2 — Train Model")
    lr = st.slider("Learning rate", 0.01, 1.0, 0.1, 0.01)
    iters = st.slider("Iterations", 100, 2000, 1000, 100)

    if df_raw is not None:
        if st.button("🏋️ Train Model"):
            with st.spinner("Training in progress…"):
                df_clean = rp.load_and_clean(
                    None) if False else _clean_from_df(df_raw)
                theta, mean_X, std_X, cost_hist, metrics = rp.train(
                    df_clean, lr=lr, iterations=iters)

                st.session_state.model_trained = True
                st.session_state.theta = theta
                st.session_state.mean_X = mean_X
                st.session_state.std_X = std_X
                st.session_state.cost_history = cost_hist
                st.session_state.metrics = metrics
                st.session_state.df = df_clean
            st.success("Model trained! 🎉")
    else:
        st.info("Load data first to enable training.")

    # ── User inputs ────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("🧑 Step 3 — Your Info")
    st.markdown('<p class="sidebar-tip">Fill in your details below, then click Predict.</p>',
                unsafe_allow_html=True)

    age = st.slider("🎂 Age", 18, 30, 22)
    sleep_label = st.selectbox("😴 Sleep duration", list(rp.SLEEP_MAP.keys()), index=5)
    sleep_val = rp.SLEEP_MAP[sleep_label]
    water = st.slider("💧 Water intake (litres/day)", 1, 6, 2)
    stress = st.slider("😰 Stress level (1=Low, 5=High)", 1, 5, 2)
    screen = st.slider("📱 Screen time (hours/day)", 1, 12, 4)

    predict_btn = st.button("🔍 Predict My Risk", use_container_width=True,
                             type="primary")


# ─────────────────────────────────────────────────────────────────────────────
# Helper — clean a raw DataFrame without a file path
# ─────────────────────────────────────────────────────────────────────────────
def _clean_from_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Timestamp' in df.columns:
        df.drop(['Timestamp'], axis=1, inplace=True)
    df['illness'] = df['illness'].map({'Yes': 1, 'No': 0})
    df['sleep_duration'] = df['sleep_duration'].map(rp.SLEEP_MAP)
    df.dropna(inplace=True)
    cols = ['sleep_duration', 'water_intake', 'screentime']
    Q1, Q3 = df[cols].quantile(0.25), df[cols].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((df[cols] < (Q1 - 1.5 * IQR)) | (df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    return df[mask].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main content tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_predict, tab_eda, tab_feedback = st.tabs(
    ["🔍 Prediction", "📊 EDA", "💬 Feedback"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:

    # ── Model metrics ──────────────────────────────────────────────────────
    if st.session_state.model_trained:
        m = st.session_state.metrics
        st.subheader("📈 Model Performance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{m['accuracy']*100:.1f}%")
        c2.metric("Precision", f"{m['precision']:.3f}")
        c3.metric("Recall",    f"{m['recall']:.3f}")
        c4.metric("F1 Score",  f"{m['f1_score']:.3f}")

        # Cost curve
        with st.expander("📉 Training cost curve"):
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(st.session_state.cost_history, color="#2d6a9f", linewidth=1.5)
            ax.set_xlabel("Iteration"); ax.set_ylabel("Cost")
            ax.set_title("Logistic Regression — Cost over Iterations")
            ax.grid(alpha=.3)
            st.pyplot(fig)
            plt.close(fig)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    else:
        st.info("ℹ️ Train the model first using the sidebar (Steps 1 & 2).")

    # ── Prediction result ──────────────────────────────────────────────────
    if predict_btn:
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train the model before predicting.")
        else:
            rp._theta       = st.session_state.theta
            rp._mean_X_train = st.session_state.mean_X
            rp._std_X_train  = st.session_state.std_X

            label, prob = rp.predict_single(
                age, sleep_val, water, stress, screen)

            st.subheader("🎯 Prediction Result")
            col_res, col_gauge = st.columns([1, 1])

            with col_res:
                if label == 1:
                    st.markdown(f"""
                    <div class="risk-high">
                      <h2>⚠️ High Risk</h2>
                      <p style="font-size:1.1rem">Our model estimates you have a
                      <strong>{prob*100:.1f}%</strong> probability of illness risk.</p>
                      <p>Consider improving your sleep, hydration, and reducing screen time.</p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="risk-low">
                      <h2>✅ Low Risk</h2>
                      <p style="font-size:1.1rem">Our model estimates you have a
                      <strong>{prob*100:.1f}%</strong> probability of illness risk.</p>
                      <p>Great habits! Keep up the good work.</p>
                    </div>""", unsafe_allow_html=True)

            with col_gauge:
                # Probability bar
                fig, ax = plt.subplots(figsize=(4, 3))
                color = "#e74c3c" if label == 1 else "#27ae60"
                ax.barh(["Risk probability"], [prob], color=color, height=0.4)
                ax.barh(["Risk probability"], [1 - prob],
                        left=[prob], color="#eeeeee", height=0.4)
                ax.set_xlim(0, 1)
                ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, alpha=.6)
                ax.set_xlabel("Probability")
                ax.set_title("Risk Probability")
                ax.text(prob / 2, 0, f"{prob*100:.1f}%",
                        ha="center", va="center", color="white", fontweight="bold")
                st.pyplot(fig)
                plt.close(fig)

            # Input summary
            st.markdown("**Your inputs:**")
            summary = pd.DataFrame({
                "Feature": ["Age", "Sleep duration", "Water intake (L)",
                             "Stress level", "Screen time (hrs)"],
                "Value": [age, sleep_label, water, stress, screen]
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.header("📊 Exploratory Data Analysis")

    if st.session_state.df is None:
        st.info("ℹ️ Train the model first to unlock EDA charts.")
    else:
        df = st.session_state.df

        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Age Distribution**")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.countplot(x='age', data=df, palette='Set1', ax=ax)
            ax.set_title("Age Distribution"); ax.set_xlabel("Age"); ax.set_ylabel("Count")
            st.pyplot(fig); plt.close(fig)

        with col2:
            st.markdown("**Stress Level Distribution**")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            vc = df['stress'].value_counts().sort_index()
            sns.barplot(x=vc.index, y=vc.values, palette='Set2', ax=ax)
            ax.set_title("Stress Distribution"); ax.set_xlabel("Stress"); ax.set_ylabel("Count")
            st.pyplot(fig); plt.close(fig)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**Illness Rate by Age**")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.countplot(x='age', hue='illness', data=df, palette='Set3', ax=ax)
            ax.legend(title="Illness", labels=["No", "Yes"])
            ax.set_title("Age vs Illness"); ax.set_xlabel("Age"); ax.set_ylabel("Count")
            st.pyplot(fig); plt.close(fig)

        with col4:
            st.markdown("**Sleep Duration Distribution**")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.histplot(df['sleep_duration'], bins=10, kde=True, color="#2d6a9f", ax=ax)
            ax.set_title("Sleep Duration"); ax.set_xlabel("Hours"); ax.set_ylabel("Frequency")
            st.pyplot(fig); plt.close(fig)

        # Correlation heatmap
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("**Feature Correlation Heatmap**")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(df[rp.FEATURE_COLS + ['illness']].corr(),
                    annot=True, fmt=".2f", cmap="coolwarm",
                    linewidths=0.5, ax=ax)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig); plt.close(fig)

        # Descriptive stats
        with st.expander("📋 Descriptive Statistics"):
            st.dataframe(df.describe(), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Feedback
# ══════════════════════════════════════════════════════════════════════════════
with tab_feedback:
    st.header("💬 User Feedback")
    st.markdown("Your feedback helps improve the app. It is stored in this session only.")

    with st.form("feedback_form"):
        name = st.text_input("Your name (optional)", placeholder="e.g. Alice")
        rating = st.select_slider(
            "Rate this app ⭐",
            options=["1 ⭐", "2 ⭐⭐", "3 ⭐⭐⭐", "4 ⭐⭐⭐⭐", "5 ⭐⭐⭐⭐⭐"],
            value="3 ⭐⭐⭐"
        )
        prediction_helpful = st.radio(
            "Was the prediction helpful?", ["Yes 👍", "No 👎", "Somewhat 🤔"])
        comment = st.text_area("Any comments or suggestions?",
                               placeholder="Tell us what you think…", height=100)
        submitted = st.form_submit_button("Submit Feedback ✅", type="primary")

        if submitted:
            entry = {
                "name":   name or "Anonymous",
                "rating": rating,
                "helpful": prediction_helpful,
                "comment": comment,
            }
            st.session_state.feedbacks.append(entry)
            st.success("Thank you for your feedback! 🙏")

    if st.session_state.feedbacks:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader(f"📝 All Feedback ({len(st.session_state.feedbacks)} responses)")
        for i, fb in enumerate(reversed(st.session_state.feedbacks), 1):
            with st.expander(f"{i}. {fb['name']} — {fb['rating']}"):
                st.write(f"**Helpful?** {fb['helpful']}")
                if fb['comment']:
                    st.write(f"**Comment:** {fb['comment']}")

        # Rating distribution
        ratings = [int(fb['rating'][0]) for fb in st.session_state.feedbacks]
        avg = sum(ratings) / len(ratings)
        st.metric("Average rating", f"{avg:.1f} / 5 ⭐")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#888;font-size:.85rem'>"
    "🏥 Illness Risk Predictor · Built with Streamlit & NumPy "
    "· For educational purposes only</p>",
    unsafe_allow_html=True
)