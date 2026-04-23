"""
app.py  ─  Illness Risk Predictor  (Streamlit)
Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, sys

# ── Make sure Risk_Prediction.py is importable ────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
import Risk_Prediction as rp

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HealthRisk AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f1923 0%, #152233 50%, #0f1923 100%);
    color: #e8edf2;
}

/* Title block */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    color: #e8edf2;
    line-height: 1.15;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1.05rem;
    color: #7a9bbf;
    letter-spacing: 0.04em;
    margin-bottom: 2rem;
}
.badge {
    display: inline-block;
    background: rgba(56,189,248,0.12);
    color: #38bdf8;
    border: 1px solid rgba(56,189,248,0.3);
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* Result cards */
.result-card {
    border-radius: 14px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.2rem;
}
.result-low {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.05));
    border: 1px solid rgba(16,185,129,0.4);
}
.result-high {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
    border: 1px solid rgba(239,68,68,0.4);
}
.result-label {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    margin-bottom: 0.3rem;
}
.result-prob {
    font-size: 1rem;
    opacity: 0.75;
}

/* Metric boxes */
.metric-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: #38bdf8;
}
.metric-lbl {
    font-size: 0.78rem;
    color: #7a9bbf;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1720 !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] label {
    color: #a0b4c8 !important;
    font-size: 0.85rem !important;
}

/* Section headers */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #e8edf2;
    margin-bottom: 0.2rem;
}
.section-rule {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 0.5rem 0 1.5rem 0;
}

/* Feedback pill buttons */
div[data-testid="stHorizontalBlock"] button {
    border-radius: 20px !important;
}

/* Progress bar override */
.stProgress > div > div {
    background: linear-gradient(90deg, #38bdf8, #818cf8) !important;
    border-radius: 4px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Train model once (cached) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model on dataset…")
def get_trained_model():
    data_path = os.path.join(os.path.dirname(__file__), "raw_data.csv")
    df = rp.load_and_clean(data_path)
    theta, mean_X, std_X, cost_history, metrics = rp.train(df)
    return theta, mean_X, std_X, cost_history, metrics, df


theta, mean_X, std_X, cost_history, metrics, df_clean = get_trained_model()


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding: 1rem 0 0.5rem;'>
            <span style='font-size:2.5rem;'>🩺</span>
            <div style='font-family:"DM Serif Display",serif; font-size:1.3rem; color:#e8edf2; margin-top:0.3rem;'>
                HealthRisk AI
            </div>
            <div style='font-size:0.72rem; color:#7a9bbf; letter-spacing:0.08em; text-transform:uppercase;'>
                Illness Risk Predictor
            </div>
        </div>
        <hr style='border:none; border-top:1px solid rgba(255,255,255,0.07); margin:1rem 0;'>
        <div style='font-size:0.82rem; color:#7a9bbf; margin-bottom:1.2rem;'>
            Fill in your daily health habits and get an instant risk assessment.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("**Your Profile**")
    age = st.slider("Age", min_value=15, max_value=70, value=20, step=1)

    st.markdown("**Sleep & Hydration**")
    sleep_label = st.select_slider(
        "Sleep Duration (hrs/night)",
        options=['<3', '3-4', '4-5', '5-6', '6-7', '7-8', '>8'],
        value='6-7'
    )
    water_intake = st.slider(
        "Water Intake (litres/day)",
        min_value=0.5, max_value=10.0, value=2.5, step=0.5
    )

    st.markdown("**Lifestyle**")
    stress = st.slider(
        "Stress Level (1 = low, 10 = high)",
        min_value=1, max_value=10, value=5
    )
    screentime = st.slider(
        "Daily Screen Time (hrs)",
        min_value=0.0, max_value=16.0, value=5.0, step=0.5
    )

    st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.07); margin:1rem 0;'>",
                unsafe_allow_html=True)
    predict_btn = st.button("⚡ Predict Risk", use_container_width=True, type="primary")


# ── MAIN AREA ─────────────────────────────────────────────────────────────────
st.markdown('<div class="badge">Machine Learning · Logistic Regression</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Illness Risk<br><i>Prediction</i></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">A lightweight health screening tool trained on real survey data.</div>',
            unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔮  Prediction", "📊  Data Insights", "💬  Feedback"])


# ── TAB 1: PREDICTION ─────────────────────────────────────────────────────────
with tab1:

    # Model metrics strip
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    for col, label, key in zip(
        [m1, m2, m3, m4],
        ["Accuracy", "Precision", "Recall", "F1 Score"],
        ["accuracy", "precision", "recall", "f1_score"]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val">{metrics[key]:.0%}</div>
                <div class="metric-lbl">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Prediction result
    st.markdown('<div class="section-title">Your Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    sleep_val = rp.SLEEP_MAP[sleep_label]
    label, prob = rp.predict_single(age, sleep_val, water_intake, stress, screentime)
    prob_pct = prob * 100

    if predict_btn or True:   # always show result after sidebar change
        if label == 1:
            card_cls = "result-high"
            verdict = "⚠️ Higher Risk Detected"
            verdict_color = "#f87171"
            advice = "Consider improving sleep, reducing stress, and staying hydrated."
        else:
            card_cls = "result-low"
            verdict = "✅ Lower Risk Detected"
            verdict_color = "#34d399"
            advice = "Your current habits look healthy — keep it up!"

        col_res, col_gauge = st.columns([3, 2])

        with col_res:
            st.markdown(f"""
            <div class="result-card {card_cls}">
                <div class="result-label" style="color:{verdict_color}">{verdict}</div>
                <div class="result-prob">Illness probability: <strong>{prob_pct:.1f}%</strong></div>
                <div style="margin-top:1rem; font-size:0.9rem; color:#a0b4c8;">{advice}</div>
            </div>""", unsafe_allow_html=True)

            # Input summary
            st.markdown("**Your inputs**")
            summary_data = {
                "Feature": ["Age", "Sleep Duration", "Water Intake", "Stress Level", "Screen Time"],
                "Value": [f"{age} yrs", sleep_label + " hrs", f"{water_intake} L", f"{stress}/10", f"{screentime} hrs"]
            }
            st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

        with col_gauge:
            # Probability gauge chart
            fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')

            theta_range = np.linspace(0, np.pi, 200)
            # background arc
            ax.fill_between(theta_range, 0.65, 1.0,
                             color='#1e2d3d', alpha=0.8)
            # color arc for risk
            fill_up = np.pi * prob
            t_fill = np.linspace(0, fill_up, 200)
            color = "#f87171" if prob >= 0.5 else "#34d399"
            ax.fill_between(t_fill, 0.65, 1.0, color=color, alpha=0.85)

            ax.set_ylim(0, 1)
            ax.set_xlim(0, np.pi)
            ax.set_theta_zero_location("W")
            ax.set_theta_direction(1)
            ax.axis('off')

            ax.text(np.pi / 2, 0.3, f"{prob_pct:.0f}%",
                    ha='center', va='center',
                    fontsize=28, fontweight='bold', color=color,
                    fontfamily='serif')
            ax.text(np.pi / 2, 0.08, "Risk Score",
                    ha='center', va='center',
                    fontsize=10, color='#7a9bbf')

            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Probability bar
        st.markdown("**Risk probability**")
        st.progress(float(prob))


# ── TAB 2: EDA ────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    # Chart style helper
    def style_ax(ax):
        ax.set_facecolor('#0d1720')
        ax.tick_params(colors='#7a9bbf', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e2d3d')
        ax.title.set_color('#e8edf2')
        ax.xaxis.label.set_color('#7a9bbf')
        ax.yaxis.label.set_color('#7a9bbf')

    palette = {"Yes": "#f87171", "No": "#34d399"}

    # ① Illness distribution
    with col_a:
        st.markdown("**Illness Distribution**")
        fig, ax = plt.subplots(figsize=(4, 3))
        fig.patch.set_facecolor('#0d1720')
        counts = df_clean['illness'].value_counts()
        bars = ax.bar(['No Illness', 'Illness'], counts.values,
                      color=['#34d399', '#f87171'], width=0.5)
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(v), ha='center', va='bottom', color='#e8edf2', fontsize=9)
        ax.set_ylabel("Count")
        style_ax(ax)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ② Stress vs illness
    with col_b:
        st.markdown("**Stress Level by Illness**")
        fig, ax = plt.subplots(figsize=(4, 3))
        fig.patch.set_facecolor('#0d1720')
        for val, color, label in [(1, '#f87171', 'Illness'), (0, '#34d399', 'No Illness')]:
            subset = df_clean[df_clean['illness'] == val]['stress']
            ax.hist(subset, bins=8, alpha=0.7, color=color, label=label)
        ax.set_xlabel("Stress Level")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8, facecolor='#0d1720', labelcolor='#e8edf2', framealpha=0.5)
        style_ax(ax)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    col_c, col_d = st.columns(2)

    # ③ Water intake vs illness
    with col_c:
        st.markdown("**Water Intake by Illness**")
        fig, ax = plt.subplots(figsize=(4, 3))
        fig.patch.set_facecolor('#0d1720')
        ill = df_clean[df_clean['illness'] == 1]['water_intake']
        no_ill = df_clean[df_clean['illness'] == 0]['water_intake']
        ax.boxplot([no_ill, ill], labels=['No Illness', 'Illness'],
                   patch_artist=True,
                   boxprops=dict(facecolor='#1e2d3d', color='#38bdf8'),
                   medianprops=dict(color='#f87171'),
                   whiskerprops=dict(color='#7a9bbf'),
                   capprops=dict(color='#7a9bbf'),
                   flierprops=dict(marker='o', color='#7a9bbf', markersize=4))
        ax.set_ylabel("Litres/day")
        style_ax(ax)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ④ Sleep duration vs illness
    with col_d:
        st.markdown("**Sleep Duration vs Illness Rate**")
        fig, ax = plt.subplots(figsize=(4, 3))
        fig.patch.set_facecolor('#0d1720')
        order = ['<3', '3-4', '4-5', '5-6', '6-7', '7-8', '>8']
        # map back numeric → label using reverse SLEEP_MAP
        rev = {v: k for k, v in rp.SLEEP_MAP.items()}
        df_plot = df_clean.copy()
        df_plot['sleep_label'] = df_plot['sleep_duration'].map(rev)
        rates = []
        labels_used = []
        for sl in order:
            sub = df_plot[df_plot['sleep_label'] == sl]
            if len(sub) > 0:
                rates.append(sub['illness'].mean())
                labels_used.append(sl)
        colors_bar = ['#f87171' if r > 0.5 else '#34d399' for r in rates]
        ax.bar(labels_used, rates, color=colors_bar, width=0.6)
        ax.set_xlabel("Sleep (hrs)")
        ax.set_ylabel("Illness Rate")
        ax.set_ylim(0, 1.1)
        style_ax(ax)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Training cost curve
    st.markdown("**Model Training — Cost Curve**")
    fig, ax = plt.subplots(figsize=(8, 2.8))
    fig.patch.set_facecolor('#0d1720')
    ax.plot(cost_history, color='#38bdf8', linewidth=1.5)
    ax.fill_between(range(len(cost_history)), cost_history, alpha=0.15, color='#38bdf8')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    style_ax(ax)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Raw data preview
    with st.expander("📄  View cleaned dataset"):
        df_display = df_clean.copy()
        df_display['illness'] = df_display['illness'].map({1: 'Yes', 0: 'No'})
        st.dataframe(df_display, use_container_width=True, height=280)


# ── TAB 3: FEEDBACK ───────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Share Your Feedback</div>', unsafe_allow_html=True)
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    st.markdown("Help us improve the model accuracy and your experience.")

    # Prediction accuracy check
    st.markdown("**Was the prediction accurate for you?**")
    fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 4])
    with fb_col1:
        yes_btn = st.button("👍  Yes", use_container_width=True)
    with fb_col2:
        no_btn = st.button("👎  No", use_container_width=True)

    if yes_btn:
        st.success("Thanks! Your feedback helps improve the model. 🎉")
    if no_btn:
        st.warning("Thanks for letting us know. We'll use this to improve.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature rating
    st.markdown("**How useful did you find this tool?**")
    rating = st.select_slider(
        " ",
        options=["Not useful", "Somewhat useful", "Useful", "Very useful", "Excellent!"],
        value="Useful"
    )

    # Free text
    st.markdown("**Any comments or suggestions?**")
    comment = st.text_area("", placeholder="Type your thoughts here…", height=120, label_visibility="collapsed")

    submit_fb = st.button("Submit Feedback", type="primary")
    if submit_fb:
        if comment.strip():
            st.success(f"Thank you! You rated us: **{rating}**. Your comment has been noted. 💬")
        else:
            st.success(f"Thank you! You rated us: **{rating}**.")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.78rem; color:#4a6a8a; text-align:center;'>
        HealthRisk AI · Built with Streamlit & NumPy · Logistic Regression from scratch<br>
        This tool is for educational purposes only and does not constitute medical advice.
    </div>
    """, unsafe_allow_html=True)
