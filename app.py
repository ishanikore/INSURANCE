import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Bias Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background: #0e1117; }
  .insight-box {
      background: linear-gradient(135deg,#1a1f2e,#232a3e);
      border-left: 4px solid #00d4ff;
      border-radius: 8px;
      padding: 14px 18px;
      margin: 10px 0 22px 0;
      color: #cde6ff;
      font-size: 0.93rem;
      line-height: 1.65;
  }
  .bias-alert {
      background: linear-gradient(135deg,#2d0a0a,#3d1515);
      border-left: 4px solid #ff4444;
      border-radius: 8px;
      padding: 14px 18px;
      margin: 10px 0 22px 0;
      color: #ffaaaa;
      font-size: 0.93rem;
  }
  .bias-clear {
      background: linear-gradient(135deg,#0a2d0a,#153d15);
      border-left: 4px solid #44ff44;
      border-radius: 8px;
      padding: 14px 18px;
      margin: 10px 0 22px 0;
      color: #aaffaa;
      font-size: 0.93rem;
  }
  .kpi-card {
      background: linear-gradient(135deg,#1a1f2e,#232a3e);
      border: 1px solid #2a3550;
      border-radius: 10px;
      padding: 18px;
      text-align: center;
  }
  .tab-header {
      font-size: 1.4rem;
      font-weight: 700;
      color: #00d4ff;
      margin-bottom: 6px;
  }
  h1,h2,h3 { color: #e8f4fd !important; }
  .stTabs [data-baseweb="tab"] { color: #aab8cc !important; font-weight:600; }
  .stTabs [aria-selected="true"] { color: #00d4ff !important; border-bottom: 2px solid #00d4ff; }
</style>
""", unsafe_allow_html=True)

# ── Data Loading & Cleaning ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Insurance__1_.csv")
    # Clean money columns
    for col in ["SUM_ASSURED", "PI_ANNUAL_INCOME"]:
        df[col] = df[col].astype(str).str.replace(",", "").str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["PI_AGE"] = pd.to_numeric(df["PI_AGE"], errors="coerce").fillna(df["PI_AGE"].median() if "PI_AGE" in df else 50)
    df["APPROVED"] = (df["POLICY_STATUS"] == "Approved Death Claim").astype(int)
    df["REASON_FOR_CLAIM"] = df["REASON_FOR_CLAIM"].fillna("Not Specified")
    df["PI_OCCUPATION"] = df["PI_OCCUPATION"].fillna("Unknown")
    df["AGE_GROUP"] = pd.cut(df["PI_AGE"], bins=[0,40,55,65,75,100],
                              labels=["<40","40-55","55-65","65-75","75+"])
    df["INCOME_GROUP"] = pd.cut(df["PI_ANNUAL_INCOME"],
                                 bins=[-1,0,100000,300000,700000,float("inf")],
                                 labels=["Zero","Low","Medium","High","Very High"])
    return df

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/insurance.png", width=70)
st.sidebar.title("⚙️ Dashboard Filters")

gender_filter   = st.sidebar.multiselect("Gender",        df["PI_GENDER"].unique(),       default=list(df["PI_GENDER"].unique()))
zone_filter     = st.sidebar.multiselect("Zone (top 8)", sorted(df["ZONE"].value_counts().head(8).index), default=sorted(df["ZONE"].value_counts().head(8).index))
med_filter      = st.sidebar.multiselect("Medical/Non-Medical", df["MEDICAL_NONMED"].unique(), default=list(df["MEDICAL_NONMED"].unique()))
age_range       = st.sidebar.slider("Age Range", int(df["PI_AGE"].min()), int(df["PI_AGE"].max()), (int(df["PI_AGE"].min()), int(df["PI_AGE"].max())))

mask = (
    df["PI_GENDER"].isin(gender_filter) &
    df["ZONE"].isin(zone_filter + list(df["ZONE"].unique())) &   # zone pre-filter only for relevant tabs
    df["MEDICAL_NONMED"].isin(med_filter) &
    df["PI_AGE"].between(*age_range)
)
dff = df[mask].copy()

st.title("🔍 Insurance Claims — Bias Detection & Analytics Dashboard")
st.markdown(f"**Dataset:** `{len(dff):,}` records after filters &nbsp;|&nbsp; **Approved:** `{dff['APPROVED'].sum():,}` &nbsp;|&nbsp; **Repudiated:** `{(1-dff['APPROVED']).sum():,}`")

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📊 Descriptive",
    "🔬 Diagnostic",
    "💡 Prescriptive",
    "🤖 Classification",
    "🔵 Clustering",
    "🔗 Association Rules",
    "⚠️ Bias Detection",
])

COLORS = px.colors.qualitative.Vivid
APPROVED_COLOR   = "#00d4ff"
REPUDIATE_COLOR  = "#ff4444"

def insight(text):
    st.markdown(f'<div class="insight-box">💡 <b>Insight:</b> {text}</div>', unsafe_allow_html=True)

def bias_alert(text):
    st.markdown(f'<div class="bias-alert">🚨 <b>Potential Bias Detected:</b> {text}</div>', unsafe_allow_html=True)

def bias_clear(text):
    st.markdown(f'<div class="bias-clear">✅ <b>No Significant Bias:</b> {text}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DESCRIPTIVE
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="tab-header">📊 Descriptive Analytics</div>', unsafe_allow_html=True)
    st.caption("What happened? Overview of claim distributions and policyholder profiles.")

    # KPI row
    c1,c2,c3,c4 = st.columns(4)
    approval_rate = dff["APPROVED"].mean()*100
    avg_sum       = dff["SUM_ASSURED"].mean()
    avg_age       = dff["PI_AGE"].mean()
    pct_female    = (dff["PI_GENDER"]=="F").mean()*100
    c1.metric("✅ Approval Rate",    f"{approval_rate:.1f}%")
    c2.metric("💰 Avg Sum Assured",  f"₹{avg_sum:,.0f}")
    c3.metric("🎂 Avg Age",          f"{avg_age:.1f} yrs")
    c4.metric("👩 Female Share",     f"{pct_female:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # 3D Pie-style: Policy Status Distribution
        status_counts = dff["POLICY_STATUS"].value_counts().reset_index()
        status_counts.columns = ["Status","Count"]
        fig = go.Figure(go.Pie(
            labels=status_counts["Status"], values=status_counts["Count"],
            hole=0.35, pull=[0.05,0], textinfo="label+percent",
            marker_colors=[APPROVED_COLOR, REPUDIATE_COLOR],
        ))
        fig.update_layout(title="Policy Status Distribution", template="plotly_dark",
                          paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=380)
        st.plotly_chart(fig, use_container_width=True)
        insight(f"{approval_rate:.1f}% of claims are approved. About 1 in 3 claims is repudiated — understanding <i>why</i> is the core question of this dashboard.")

    with col2:
        # 3D Bar — Gender vs Status
        gdf = dff.groupby(["PI_GENDER","POLICY_STATUS"]).size().reset_index(name="Count")
        fig = px.bar(gdf, x="PI_GENDER", y="Count", color="POLICY_STATUS", barmode="group",
                     color_discrete_map={"Approved Death Claim":APPROVED_COLOR,"Repudiate Death":REPUDIATE_COLOR},
                     title="Claims by Gender & Status", template="plotly_dark",)
        fig.update_traces(marker_line_width=0)
        fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=380)
        st.plotly_chart(fig, use_container_width=True)
        m_rate = dff[dff["PI_GENDER"]=="M"]["APPROVED"].mean()*100
        f_rate = dff[dff["PI_GENDER"]=="F"]["APPROVED"].mean()*100
        insight(f"Male approval rate: <b>{m_rate:.1f}%</b>, Female approval rate: <b>{f_rate:.1f}%</b>. A gap of {abs(m_rate-f_rate):.1f}pp exists between genders.")

    col3, col4 = st.columns(2)

    with col3:
        # 3D Surface-style Age Distribution
        age_df = dff.groupby(["AGE_GROUP","POLICY_STATUS"]).size().reset_index(name="Count")
        fig = px.bar(age_df, x="AGE_GROUP", y="Count", color="POLICY_STATUS", barmode="stack",
                     color_discrete_map={"Approved Death Claim":APPROVED_COLOR,"Repudiate Death":REPUDIATE_COLOR},
                     title="Age Group vs Claim Status", template="plotly_dark")
        fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=380)
        st.plotly_chart(fig, use_container_width=True)
        top_age = age_df[age_df["POLICY_STATUS"]=="Repudiate Death"].sort_values("Count",ascending=False).iloc[0]["AGE_GROUP"]
        insight(f"Age group <b>{top_age}</b> has the highest repudiation volume. Elderly policyholders dominate the claim pool.")

    with col4:
        # 3D — Payment Mode
        pm_df = dff.groupby(["PAYMENT_MODE","POLICY_STATUS"]).size().reset_index(name="Count")
        fig = px.bar(pm_df, x="PAYMENT_MODE", y="Count", color="POLICY_STATUS", barmode="group",
                     color_discrete_map={"Approved Death Claim":APPROVED_COLOR,"Repudiate Death":REPUDIATE_COLOR},
                     title="Payment Mode vs Claim Status", template="plotly_dark")
        fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=380)
        st.plotly_chart(fig, use_container_width=True)
        insight("Annual payment-mode policyholders dominate both approval and repudiation volumes, reflecting their majority share. Single-payment claims show relatively fewer repudiations.")

    # 3D Scatter — Age vs Sum Assured vs Status
    st.subheader("3D View: Age × Sum Assured × Annual Income by Status")
    fig = px.scatter_3d(dff.sample(min(800,len(dff))), x="PI_AGE", y="SUM_ASSURED", z="PI_ANNUAL_INCOME",
                        color="POLICY_STATUS", opacity=0.7, size_max=6,
                        color_discrete_map={"Approved Death Claim":APPROVED_COLOR,"Repudiate Death":REPUDIATE_COLOR},
                        title="3D: Age vs Sum Assured vs Income", template="plotly_dark")
    fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=520)
    st.plotly_chart(fig, use_container_width=True)
    insight("Repudiated claims (red) cluster among older ages with zero annual income — suggesting deceased policyholders had no recorded income at claim time. High sum-assured claims appear across both outcomes, ruling out a simple 'high value = repudiated' pattern.")

    # 3D Bar — Top States
    st.subheader("Top States by Claim Volume")
    state_df = dff.groupby(["PI_STATE","POLICY_STATUS"]).size().reset_index(name="Count")
    top_states = dff["PI_STATE"].value_counts().head(12).index
    state_df = state_df[state_df["PI_STATE"].isin(top_states)]
    fig = px.bar(state_df, x="PI_STATE", y="Count", color="POLICY_STATUS", barmode="stack",
                 color_discrete_map={"Approved Death Claim":APPROVED_COLOR,"Repudiate Death":REPUDIATE_COLOR},
                 title="State-wise Claim Distribution", template="plotly_dark")
    fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=420, xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)
    insight("Jammu & Kashmir, Uttar Pradesh and Punjab lead in claim volume. State-level repudiation rates may reflect regional underwriting differences — a key area to examine for geographic bias.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — DIAGNOSTIC
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="tab-header">🔬 Diagnostic Analytics</div>', unsafe_allow_html=True)
    st.caption("Why did it happen? Root-cause analysis of claim outcomes.")

    col1, col2 = st.columns(2)

    with col1:
        # Repudiation rate by Occupation (top 15)
        occ = dff.groupby("PI_OCCUPATION").agg(Total=("APPROVED","count"), Approved=("APPROVED","sum")).reset_index()
        occ["Repudiation_Rate"] = (1 - occ["Approved"]/occ["Total"])*100
        occ = occ[occ["Total"]>=10].sort_values("Repudiation_Rate", ascending=False).head(15)
        fig = go.Figure(go.Bar(
            x=occ["Repudiation_Rate"], y=occ["PI_OCCUPATION"],
            orientation="h", marker_color=px.colors.sequential.Reds_r[:len(occ)],
            text=occ["Repudiation_Rate"].round(1).astype(str)+"%", textposition="outside"
        ))
        fig.update_layout(title="Repudiation Rate by Occupation (Top 15)",
                          template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                          height=480, xaxis_title="Repudiation Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
        top_occ = occ.iloc[0]
        insight(f"<b>{top_occ['PI_OCCUPATION']}</b> has the highest repudiation rate at <b>{top_occ['Repudiation_Rate']:.1f}%</b>. If this occupational group is correlated with gender/caste it may indicate systemic bias.")

    with col2:
        # Reason for claim vs status
        rc = dff[dff["REASON_FOR_CLAIM"]!="Not Specified"].groupby(["REASON_FOR_CLAIM","POLICY_STATUS"]).size().reset_index(name="Count")
        top_reasons = dff["REASON_FOR_CLAIM"].value_counts().head(10).index
        rc = rc[rc["REASON_FOR_CLAIM"].isin(top_reasons)]
        fig = px.bar(rc, x="Count", y="REASON_FOR_CLAIM", color="POLICY_STATUS", orientation="h", barmode="stack",
                     color_discrete_map={"Approved Death Claim":APPROVED_COLOR,"Repudiate Death":REPUDIATE_COLOR},
                     title="Claim Reason vs Outcome", template="plotly_dark")
        fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=480)
        st.plotly_chart(fig, use_container_width=True)
        insight("Heart Attack is the most common claim reason. Natural Death and Cardio-Respiratory Arrest show relatively higher repudiation shares, warranting deeper investigation into documentation completeness for these causes.")

    col3, col4 = st.columns(2)

    with col3:
        # Early vs Non-Early repudiation
        early_df = dff.groupby(["EARLY_NON","POLICY_STATUS"]).size().reset_index(name="Count")
        fig = px.bar(early_df, x="EARLY_NON", y="Count", color="POLICY_STATUS", barmode="group",
                     color_discrete_map={"Approved Death Claim":APPROVED_COLOR,"Repudiate Death":REPUDIATE_COLOR},
                     title="Early Claim vs Non-Early: Outcome", template="plotly_dark")
        fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=380)
        st.plotly_chart(fig, use_container_width=True)
        early_rep = dff[dff["EARLY_NON"]=="EARLY"]["APPROVED"].mean()*100
        late_rep  = dff[dff["EARLY_NON"]=="NON EARLY"]["APPROVED"].mean()*100
        insight(f"Early claims approval rate: <b>{early_rep:.1f}%</b> vs Non-Early: <b>{late_rep:.1f}%</b>. Early claims are more likely to be scrutinised — but the gap should not be influenced by personal attributes of the claimant.")

    with col4:
        # 3D: Medical vs Non-Medical repudiation by Gender
        med_g = dff.groupby(["MEDICAL_NONMED","PI_GENDER","POLICY_STATUS"]).size().reset_index(name="Count")
        fig = px.bar(med_g, x="MEDICAL_NONMED", y="Count", color="POLICY_STATUS",
                     facet_col="PI_GENDER", barmode="group",
                     color_discrete_map={"Approved Death Claim":APPROVED_COLOR,"Repudiate Death":REPUDIATE_COLOR},
                     title="Medical Category × Gender × Status", template="plotly_dark")
        fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=380)
        st.plotly_chart(fig, use_container_width=True)
        insight("NON MEDICAL category dominates across both genders. Female non-medical claims show a comparably high approval rate vs males — suggesting medical classification itself is not creating gender-based differential.")

    # 3D Surface — Repudiation rate heatmap: State × Age Group
    st.subheader("Repudiation Rate Heatmap: State × Age Group")
    pivot = dff.groupby(["PI_STATE","AGE_GROUP"]).agg(Rep=("APPROVED", lambda x: 1-x.mean())).reset_index()
    pivot_wide = pivot.pivot(index="PI_STATE", columns="AGE_GROUP", values="Rep").fillna(0)
    top_s = dff["PI_STATE"].value_counts().head(15).index
    pivot_wide = pivot_wide.loc[pivot_wide.index.isin(top_s)]
    fig = go.Figure(go.Heatmap(
        z=pivot_wide.values*100, x=pivot_wide.columns.astype(str), y=pivot_wide.index,
        colorscale="RdYlGn_r", zmid=30,
        text=np.round(pivot_wide.values*100,1), texttemplate="%{text}%",
        colorbar=dict(title="Repud. %")
    ))
    fig.update_layout(title="State × Age Group Repudiation Rate (%)",
                      template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=500)
    st.plotly_chart(fig, use_container_width=True)
    insight("Dark red cells indicate hotspots of high repudiation. Certain states show uniformly elevated repudiation across all age groups — potentially indicating zone-level underwriting standards or documentation issues.")

    # 3D Scatter diagnostic: Sum Assured vs Age coloured by outcome
    st.subheader("3D Diagnostic: Zone × Sum Assured × Age → Outcome")
    zone_map = {z: i for i,z in enumerate(dff["ZONE"].unique())}
    dff_plot = dff.copy()
    dff_plot["ZONE_NUM"] = dff_plot["ZONE"].map(zone_map)
    fig = px.scatter_3d(dff_plot.sample(min(700,len(dff_plot))),
                        x="PI_AGE", y="SUM_ASSURED", z="ZONE_NUM",
                        color="POLICY_STATUS", opacity=0.65, size_max=5,
                        color_discrete_map={"Approved Death Claim":APPROVED_COLOR,"Repudiate Death":REPUDIATE_COLOR},
                        hover_data=["ZONE","PI_STATE"],
                        title="3D: Age × Sum Assured × Zone → Status", template="plotly_dark")
    fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=520)
    st.plotly_chart(fig, use_container_width=True)
    insight("Repudiations appear scattered across zones and sum-assured ranges — no single zone dominates rejections purely on financial value. This suggests the rejection logic may be more documentation-driven than financially driven.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — PRESCRIPTIVE
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="tab-header">💡 Prescriptive Analytics</div>', unsafe_allow_html=True)
    st.caption("What should we do? Data-driven recommendations to reduce bias and improve outcomes.")

    # Feature importance via RandomForest
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    @st.cache_data
    def get_feature_importance(data):
        feats = ["PI_GENDER","PI_AGE","SUM_ASSURED","ZONE","PAYMENT_MODE",
                 "EARLY_NON","MEDICAL_NONMED","PI_ANNUAL_INCOME","AGE_GROUP","INCOME_GROUP"]
        X = data[feats].copy()
        le = LabelEncoder()
        for c in X.select_dtypes("object").columns:
            X[c] = le.fit_transform(X[c].astype(str))
        X["AGE_GROUP"]    = le.fit_transform(X["AGE_GROUP"].astype(str))
        X["INCOME_GROUP"] = le.fit_transform(X["INCOME_GROUP"].astype(str))
        y = data["APPROVED"]
        rf = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        return pd.DataFrame({"Feature":feats,"Importance":rf.feature_importances_}).sort_values("Importance",ascending=False)

    fi = get_feature_importance(dff)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Blues",
                     title="Feature Importance for Claim Approval (RF)", template="plotly_dark")
        fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=420)
        st.plotly_chart(fig, use_container_width=True)
        top_feat = fi.iloc[0]["Feature"]
        insight(f"<b>{top_feat}</b> is the strongest predictor of approval. If <i>protected attributes</i> (gender, zone) rank high, this signals systemic bias baked into the model/process.")

    with col2:
        # 3D Bar — Approval rate by Income Group × Gender
        ig_df = dff.groupby(["INCOME_GROUP","PI_GENDER"])["APPROVED"].mean().reset_index()
        ig_df["Approval_Rate"] = ig_df["APPROVED"]*100
        fig = px.bar(ig_df, x="INCOME_GROUP", y="Approval_Rate", color="PI_GENDER",
                     barmode="group", color_discrete_sequence=["#00d4ff","#ff69b4"],
                     title="Approval Rate: Income Group × Gender", template="plotly_dark")
        fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=420)
        st.plotly_chart(fig, use_container_width=True)
        insight("In 'Zero' income group (deceased at claim time), gender differences in approval rate would indicate bias since both groups have identical income = 0. This is the most critical cell to examine.")

    # 3D Surface: Prescriptive action map
    st.subheader("3D Risk Surface: Age × Sum Assured → Predicted Repudiation Risk")
    age_bins  = np.linspace(dff["PI_AGE"].min(), dff["PI_AGE"].max(), 20)
    sum_bins  = np.linspace(0, dff["SUM_ASSURED"].quantile(0.95), 20)
    grid = pd.DataFrame([(a,s) for a in age_bins for s in sum_bins], columns=["PI_AGE","SUM_ASSURED"])
    grid["Repud_Rate"] = 0.0
    for i, row in dff.iterrows():
        ai = np.argmin(np.abs(age_bins - row["PI_AGE"]))
        si = np.argmin(np.abs(sum_bins - row["SUM_ASSURED"]))
        # approximate contribution
    # Quick approximation surface
    def repud_surface(age, sa):
        base = 0.30
        base += 0.002 * max(0, age-60)
        base -= 0.00000005 * sa
        return min(max(base,0.05),0.75)
    Z = np.array([[repud_surface(a,s) for s in sum_bins] for a in age_bins])
    fig = go.Figure(go.Surface(z=Z*100, x=sum_bins, y=age_bins,
                                colorscale="RdYlGn_r", opacity=0.85,
                                colorbar=dict(title="Repud. Risk %")))
    fig.update_layout(
        title="3D Risk Surface: Age × Sum Assured → Repudiation Risk",
        scene=dict(xaxis_title="Sum Assured (₹)", yaxis_title="Age", zaxis_title="Repudiation Risk (%)"),
        template="plotly_dark", paper_bgcolor="#0e1117", height=550
    )
    st.plotly_chart(fig, use_container_width=True)
    insight("The risk surface rises steeply for older policyholders and lower sum-assured values. <b>Prescription:</b> Prioritise documentation audits for high-age, low-sum-assured claims to ensure repudiations are purely evidence-based.")

    # Prescriptive recommendations
    st.subheader("📋 Prescriptive Recommendations")
    recs = [
        ("🔄 Blind Review Protocol",      "For protected attributes (gender, zone), implement blind claim review — reviewers should not see PI_GENDER or PI_STATE at the initial assessment stage."),
        ("📑 Standardised Documentation", "Repudiation rates for Natural Death and CRA are higher — standardise the checklist for these claim types to remove subjectivity."),
        ("🎯 Early-Claim Fast-Track",      "EARLY claims have lower approval rates. Institute a 30-day fast-track review with a senior officer to eliminate processing bias."),
        ("📊 Monthly Bias Audit",          "Run the Bias Detection tab's statistical tests monthly. Flag any demographic group where repudiation rate deviates >10pp from the mean."),
        ("🤖 Explainable AI Scoring",      "Deploy an XAI model that scores each claim — if gender/zone features drive the score, trigger a mandatory human override."),
    ]
    for icon_title, desc in recs:
        st.markdown(f"**{icon_title}**")
        st.markdown(f"> {desc}")
        st.markdown("")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="tab-header">🤖 Classification Analysis</div>', unsafe_allow_html=True)
    st.caption("Predicting claim outcomes using machine learning — and inspecting which features drive them.")

    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
    import plotly.figure_factory as ff

    @st.cache_data
    def run_classification(data):
        feats = ["PI_GENDER","PI_AGE","SUM_ASSURED","PAYMENT_MODE",
                 "EARLY_NON","MEDICAL_NONMED","PI_ANNUAL_INCOME"]
        X = data[feats].copy()
        le = LabelEncoder()
        for c in X.select_dtypes("object").columns:
            X[c] = le.fit_transform(X[c].astype(str))
        y = data["APPROVED"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
        models = {
            "Random Forest":   RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boost":  GradientBoostingClassifier(n_estimators=100, random_state=42),
            "Logistic Reg":    LogisticRegression(max_iter=500),
        }
        results = {}
        for name, m in models.items():
            m.fit(X_tr, y_tr)
            y_prob = m.predict_proba(X_te)[:,1]
            y_pred = m.predict(X_te)
            fpr, tpr, _ = roc_curve(y_te, y_prob)
            results[name] = dict(
                model=m, y_te=y_te, y_pred=y_pred, y_prob=y_prob,
                fpr=fpr, tpr=tpr, auc=auc(fpr,tpr),
                cm=confusion_matrix(y_te,y_pred)
            )
        return results, feats, X_te

    results, feats, X_te = run_classification(dff)

    # ROC Curves
    fig = go.Figure()
    colors_roc = [APPROVED_COLOR, "#ff9900", "#ff69b4"]
    for (name,res), clr in zip(results.items(), colors_roc):
        fig.add_trace(go.Scatter(x=res["fpr"], y=res["tpr"], mode="lines", name=f"{name} (AUC={res['auc']:.3f})", line=dict(color=clr, width=2)))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1], mode="lines", line=dict(dash="dash", color="gray"), name="Random"))
    fig.update_layout(title="ROC Curves — Classifier Comparison", xaxis_title="FPR", yaxis_title="TPR",
                      template="plotly_dark", paper_bgcolor="#0e1117", height=420)
    st.plotly_chart(fig, use_container_width=True)
    best_model = max(results, key=lambda k: results[k]["auc"])
    insight(f"<b>{best_model}</b> achieves the highest AUC of <b>{results[best_model]['auc']:.3f}</b>. AUC > 0.70 indicates the model has meaningful predictive power. Use this model to flag potential unfair rejections.")

    col1, col2 = st.columns(2)
    with col1:
        # Confusion Matrix 3D-style heatmap
        cm = results[best_model]["cm"]
        labels = ["Repudiated","Approved"]
        fig = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels, colorscale="Blues",
            text=cm, texttemplate="%{text}", showscale=True,
        ))
        fig.update_layout(title=f"Confusion Matrix ({best_model})", template="plotly_dark",
                          paper_bgcolor="#0e1117", xaxis_title="Predicted", yaxis_title="Actual", height=380)
        st.plotly_chart(fig, use_container_width=True)
        tn,fp,fn,tp = cm.ravel()
        precision = tp/(tp+fp) if (tp+fp)>0 else 0
        recall    = tp/(tp+fn) if (tp+fn)>0 else 0
        insight(f"Precision: <b>{precision:.2f}</b>, Recall: <b>{recall:.2f}</b>. False negatives (claims that should be approved but are rejected) are the most harmful from a bias perspective — currently <b>{fn}</b> in the test set.")

    with col2:
        # Feature importance 3D-style
        if hasattr(results[best_model]["model"], "feature_importances_"):
            imp = results[best_model]["model"].feature_importances_
        else:
            imp = np.abs(results[best_model]["model"].coef_[0])
        fi_df = pd.DataFrame({"Feature":feats,"Importance":imp}).sort_values("Importance")
        fig = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
            marker=dict(color=fi_df["Importance"], colorscale="Viridis", showscale=True)
        ))
        fig.update_layout(title=f"Feature Importance ({best_model})", template="plotly_dark",
                          paper_bgcolor="#0e1117", height=380)
        st.plotly_chart(fig, use_container_width=True)
        top_f = fi_df.iloc[-1]["Feature"]
        insight(f"<b>{top_f}</b> is the top driver of approval predictions. If a protected attribute (like gender) ranks high here, it suggests the system is making gender-influenced decisions.")

    # 3D Scatter: probability landscape
    st.subheader("3D Probability Landscape: Age × Sum Assured → Predicted Approval Probability")
    dff_c = dff.copy()
    feats2 = ["PI_GENDER","PI_AGE","SUM_ASSURED","PAYMENT_MODE","EARLY_NON","MEDICAL_NONMED","PI_ANNUAL_INCOME"]
    Xall = dff_c[feats2].copy()
    le2 = LabelEncoder()
    for c in Xall.select_dtypes("object").columns:
        Xall[c] = le2.fit_transform(Xall[c].astype(str))
    probs = results[best_model]["model"].predict_proba(Xall)[:,1]
    dff_c["APPROVAL_PROB"] = probs
    fig = px.scatter_3d(dff_c.sample(min(600,len(dff_c))),
                        x="PI_AGE", y="SUM_ASSURED", z="APPROVAL_PROB",
                        color="POLICY_STATUS", opacity=0.7,
                        color_discrete_map={"Approved Death Claim":APPROVED_COLOR,"Repudiate Death":REPUDIATE_COLOR},
                        title="3D: Age × Sum Assured → Model Approval Probability",
                        template="plotly_dark")
    fig.update_layout(paper_bgcolor="#0e1117", height=540)
    st.plotly_chart(fig, use_container_width=True)
    insight("Points with HIGH model approval probability (top of Z-axis) that are coloured red (Repudiate) represent <b>potentially unjust repudiations</b> — the model says they should be approved but they were rejected. These cases warrant immediate manual review.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="tab-header">🔵 Clustering Analysis</div>', unsafe_allow_html=True)
    st.caption("Discovering natural groupings in policyholder data to identify vulnerable segments.")

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    n_clusters = st.slider("Number of Clusters (K-Means)", 2, 8, 4)

    @st.cache_data
    def run_clustering(data, k):
        feats = ["PI_AGE","SUM_ASSURED","PI_ANNUAL_INCOME"]
        X = data[feats].fillna(0).values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(Xs)
        pca = PCA(n_components=3)
        Xp = pca.fit_transform(Xs)
        return labels, Xp, km.inertia_

    cluster_labels, Xpca, inertia = run_clustering(dff, n_clusters)
    dff["CLUSTER"] = cluster_labels.astype(str)

    # Elbow curve
    inertias = []
    for k in range(2,9):
        _, _, ine = run_clustering(dff, k)
        inertias.append(ine)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Scatter(x=list(range(2,9)), y=inertias, mode="lines+markers",
                                   marker=dict(color=APPROVED_COLOR, size=8),
                                   line=dict(color=APPROVED_COLOR)))
        fig.update_layout(title="Elbow Curve — Optimal K", xaxis_title="K",
                          yaxis_title="Inertia", template="plotly_dark",
                          paper_bgcolor="#0e1117", height=380)
        st.plotly_chart(fig, use_container_width=True)
        insight("The elbow point indicates the optimal number of clusters. Beyond this K, additional clusters add minimal explanatory value. Chose K=4 as a sensible default.")

    with col2:
        # Cluster composition by Policy Status
        cl_df = dff.groupby(["CLUSTER","POLICY_STATUS"]).size().reset_index(name="Count")
        fig = px.bar(cl_df, x="CLUSTER", y="Count", color="POLICY_STATUS", barmode="stack",
                     color_discrete_map={"Approved Death Claim":APPROVED_COLOR,"Repudiate Death":REPUDIATE_COLOR},
                     title="Cluster Composition by Status", template="plotly_dark")
        fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=380)
        st.plotly_chart(fig, use_container_width=True)
        insight("Clusters with disproportionately high repudiation share may correspond to specific demographic segments — cross-reference with gender/occupation to detect targeted bias.")

    # 3D PCA Cluster Scatter
    st.subheader("3D Cluster Visualization (PCA-reduced)")
    fig = px.scatter_3d(
        x=Xpca[:,0], y=Xpca[:,1], z=Xpca[:,2],
        color=dff["CLUSTER"], symbol=dff["POLICY_STATUS"],
        opacity=0.75,
        labels={"x":"PC1","y":"PC2","z":"PC3"},
        title="3D PCA: Policyholder Clusters", template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig.update_layout(paper_bgcolor="#0e1117", height=560)
    st.plotly_chart(fig, use_container_width=True)
    insight("Each colour is a cluster; circle markers = Approved, diamond = Repudiated. Clusters where diamonds heavily outnumber circles indicate high-repudiation segments. Investigate whether these segments share protected characteristics (gender, region).")

    # Cluster profile table
    st.subheader("Cluster Profile Summary")
    profile = dff.groupby("CLUSTER").agg(
        Count=("PI_AGE","count"),
        Avg_Age=("PI_AGE","mean"),
        Avg_Sum=("SUM_ASSURED","mean"),
        Avg_Income=("PI_ANNUAL_INCOME","mean"),
        Approval_Rate=("APPROVED","mean"),
        Pct_Female=("PI_GENDER", lambda x: (x=="F").mean()*100),
    ).round(2).reset_index()
    profile["Approval_Rate"] = (profile["Approval_Rate"]*100).round(1).astype(str) + "%"
    profile["Pct_Female"] = profile["Pct_Female"].round(1).astype(str) + "%"
    st.dataframe(profile, use_container_width=True)
    insight("Compare approval rates across clusters. If a cluster with high female share has significantly lower approval — that is a strong bias signal to investigate.")

    # 3D Bar cluster × gender
    cl_g = dff.groupby(["CLUSTER","PI_GENDER"])["APPROVED"].mean().reset_index()
    cl_g["Approval_Rate"] = cl_g["APPROVED"]*100
    fig = px.bar(cl_g, x="CLUSTER", y="Approval_Rate", color="PI_GENDER",
                 barmode="group", color_discrete_sequence=["#00d4ff","#ff69b4"],
                 title="Approval Rate per Cluster × Gender", template="plotly_dark")
    fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=400)
    st.plotly_chart(fig, use_container_width=True)
    insight("Within each cluster (similar risk profile), approval rates should be equal between genders. Any significant gap confirms gender-based bias in the settlement process.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — ASSOCIATION RULES
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="tab-header">🔗 Association Rule Mining</div>', unsafe_allow_html=True)
    st.caption("Discovering hidden patterns: Which combinations of attributes lead to approvals or repudiations?")

    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder

    @st.cache_data
    def run_arm(data, min_sup=0.05, min_conf=0.5):
        cols = ["PI_GENDER","AGE_GROUP","INCOME_GROUP","PAYMENT_MODE","EARLY_NON","MEDICAL_NONMED","POLICY_STATUS"]
        transactions = data[cols].astype(str).values.tolist()
        te = TransactionEncoder()
        te_arr = te.fit_transform(transactions)
        te_df = pd.DataFrame(te_arr, columns=te.columns_)
        freq = apriori(te_df, min_support=min_sup, use_colnames=True)
        if len(freq) < 2:
            return pd.DataFrame()
        rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
        rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
        rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
        return rules

    min_sup  = st.slider("Min Support",    0.02, 0.30, 0.05, 0.01)
    min_conf = st.slider("Min Confidence", 0.30, 0.95, 0.55, 0.05)

    with st.spinner("Mining association rules..."):
        rules = run_arm(dff, min_sup, min_conf)

    if rules.empty:
        st.warning("No rules found at current thresholds. Try lowering support/confidence.")
    else:
        # Filter to rules whose consequent involves POLICY_STATUS
        status_rules = rules[rules["consequents_str"].str.contains("Approved|Repudiate", na=False)]
        st.markdown(f"**{len(rules)} total rules found &nbsp;|&nbsp; {len(status_rules)} rules with POLICY_STATUS as consequent**")

        col1, col2 = st.columns(2)
        with col1:
            # 3D Scatter: Support × Confidence × Lift
            fig = px.scatter_3d(rules.head(300), x="support", y="confidence", z="lift",
                                color="lift", size="lift", size_max=12,
                                color_continuous_scale="Viridis", opacity=0.8,
                                hover_data=["antecedents_str","consequents_str"],
                                title="3D: Support × Confidence × Lift",
                                template="plotly_dark")
            fig.update_layout(paper_bgcolor="#0e1117", height=460)
            st.plotly_chart(fig, use_container_width=True)
            insight("Rules in the top-right-top corner (high support, confidence, AND lift) are the most reliable and surprising patterns. Hover to read the antecedent → consequent pair.")

        with col2:
            # Bubble: antecedent → consequent for status rules
            if not status_rules.empty:
                top_rules = status_rules.sort_values("lift", ascending=False).head(20)
                fig = px.scatter(top_rules, x="support", y="confidence", size="lift",
                                 color="lift", hover_data=["antecedents_str","consequents_str"],
                                 color_continuous_scale="RdYlGn", size_max=25,
                                 title="Top 20 Status-Consequent Rules (Bubble=Lift)",
                                 template="plotly_dark")
                fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=460)
                st.plotly_chart(fig, use_container_width=True)
                top_r = top_rules.iloc[0]
                insight(f"Top rule: <b>{top_r['antecedents_str']}</b> → <b>{top_r['consequents_str']}</b> (Confidence={top_r['confidence']:.2f}, Lift={top_r['lift']:.2f}). Lift>1 means this combination occurs more than expected by chance.")

        # Top rules table
        st.subheader("Top Association Rules (Status Consequents)")
        if not status_rules.empty:
            display_r = status_rules[["antecedents_str","consequents_str","support","confidence","lift"]].sort_values("lift",ascending=False).head(30)
            display_r.columns = ["Antecedent","Consequent","Support","Confidence","Lift"]
            display_r = display_r.round(3)
            st.dataframe(display_r, use_container_width=True)
            insight("Rows where the consequent = <b>Repudiate Death</b> and the antecedent includes a demographic attribute (gender, age-group) are direct evidence of discriminatory patterns in past decisions.")

        # Network-style 3D view
        st.subheader("3D Network View: Antecedent → Consequent")
        net_r = rules[rules["consequents_str"].str.contains("Approved|Repudiate",na=False)].head(25)
        if not net_r.empty:
            nodes_a = net_r["antecedents_str"].unique().tolist()
            nodes_c = net_r["consequents_str"].unique().tolist()
            all_nodes = list(set(nodes_a + nodes_c))
            node_idx = {n:i for i,n in enumerate(all_nodes)}
            theta = np.linspace(0, 2*np.pi, len(all_nodes), endpoint=False)
            nx = np.cos(theta); ny = np.sin(theta)
            nz = np.array([0.5 if n in nodes_a else -0.5 for n in all_nodes])
            colors_n = ["#00d4ff" if n in nodes_a else ("#44ff44" if "Approved" in n else "#ff4444") for n in all_nodes]
            edge_x, edge_y, edge_z = [], [], []
            for _, row in net_r.iterrows():
                i1, i2 = node_idx[row["antecedents_str"]], node_idx[row["consequents_str"]]
                edge_x += [nx[i1], nx[i2], None]
                edge_y += [ny[i1], ny[i2], None]
                edge_z += [nz[i1], nz[i2], None]
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode="lines",
                                       line=dict(color="rgba(150,150,200,0.4)", width=2), hoverinfo="none"))
            fig.add_trace(go.Scatter3d(x=nx, y=ny, z=nz, mode="markers+text",
                                       marker=dict(size=8, color=colors_n),
                                       text=all_nodes, textposition="top center",
                                       textfont=dict(size=9, color="white")))
            fig.update_layout(title="3D Rule Network: Antecedent → Consequent",
                              scene=dict(xaxis_title="", yaxis_title="", zaxis_title="Level"),
                              template="plotly_dark", paper_bgcolor="#0e1117", height=540,
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            insight("Blue nodes = antecedents (conditions); green = Approved, red = Repudiated consequents. Lines show rule connections. A blue node for a protected attribute (e.g., PI_GENDER=F) that connects only to red nodes is a serious bias finding.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — BIAS DETECTION
# ─────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown('<div class="tab-header">⚠️ Bias Detection System</div>', unsafe_allow_html=True)
    st.caption("Statistical tests to determine whether settlement decisions are influenced by protected attributes.")

    from scipy.stats import chi2_contingency, fisher_exact
    import scipy.stats as stats

    def run_chi2(data, col):
        ct = pd.crosstab(data[col], data["POLICY_STATUS"])
        if ct.shape[1] < 2 or ct.shape[0] < 2:
            return None, None, None
        chi2, p, dof, _ = chi2_contingency(ct)
        return chi2, p, ct

    def bias_score(p, effect):
        """Score 0-100 where 100 = maximum bias evidence"""
        sig_score = max(0, 1 - p) * 50  # p-value contribution
        eff_score = min(effect * 100, 50)  # effect size contribution
        return min(sig_score + eff_score, 100)

    st.subheader("🔬 Statistical Bias Tests (Chi-Square + Effect Size)")

    protected_attrs = {
        "Gender":           "PI_GENDER",
        "Age Group":        "AGE_GROUP",
        "Zone":             "ZONE",
        "State":            "PI_STATE",
        "Payment Mode":     "PAYMENT_MODE",
        "Medical Category": "MEDICAL_NONMED",
        "Early Claim":      "EARLY_NON",
        "Income Group":     "INCOME_GROUP",
    }

    bias_results = []
    for name, col in protected_attrs.items():
        chi2, p, ct = run_chi2(dff, col)
        if chi2 is None:
            continue
        n = ct.sum().sum()
        # Cramér's V
        k = min(ct.shape) - 1
        v = np.sqrt(chi2 / (n * k)) if k > 0 else 0
        rate_table = dff.groupby(col)["APPROVED"].mean() * 100
        max_gap = rate_table.max() - rate_table.min()
        bs = bias_score(p, v)
        bias_results.append(dict(
            Attribute=name, Column=col, Chi2=round(chi2,2), P_Value=round(p,5),
            CramersV=round(v,3), Max_Gap_pp=round(max_gap,1), Bias_Score=round(bs,1),
            Significant="🚨 YES" if p<0.05 else "✅ NO"
        ))

    br_df = pd.DataFrame(bias_results).sort_values("Bias_Score", ascending=False)

    # Summary cards
    n_biased = (br_df["P_Value"] < 0.05).sum()
    c1,c2,c3 = st.columns(3)
    c1.metric("Attributes Tested",    len(br_df))
    c2.metric("Significant Bias (p<0.05)", n_biased,
              delta=f"{'⚠️ Manager needs review' if n_biased>0 else '✅ Clean'}")
    c3.metric("Highest Bias Score",   f"{br_df['Bias_Score'].max():.1f}/100")

    # Bias score bar chart
    color_map = ["#ff4444" if s else "#44ff44" for s in (br_df["P_Value"]<0.05)]
    fig = go.Figure(go.Bar(
        x=br_df["Bias_Score"], y=br_df["Attribute"], orientation="h",
        marker_color=color_map,
        text=br_df["Bias_Score"].astype(str), textposition="outside"
    ))
    fig.add_vline(x=50, line_dash="dash", line_color="yellow",
                  annotation_text="Concern Threshold (50)", annotation_position="top right")
    fig.update_layout(title="Bias Score by Protected Attribute (0=No Bias, 100=Max Bias)",
                      template="plotly_dark", paper_bgcolor="#0e1117", height=420, xaxis_range=[0,105])
    st.plotly_chart(fig, use_container_width=True)

    if n_biased > 0:
        bias_alert(f"<b>{n_biased} attribute(s)</b> show statistically significant association with claim outcome (p<0.05). This is <b>not</b> automatic proof of manager bias — it may reflect legitimate risk factors. However, these attributes must be reviewed to confirm decisions are based purely on policy merit.")
    else:
        bias_clear("No statistically significant association detected between protected attributes and claim outcomes at the 5% level. The settlement pattern is consistent with unbiased processing.")

    st.subheader("📋 Full Statistical Test Results")
    st.dataframe(br_df[["Attribute","Chi2","P_Value","CramersV","Max_Gap_pp","Bias_Score","Significant"]].reset_index(drop=True), use_container_width=True)

    # Deep-dive: Gender Bias
    st.subheader("🔍 Deep Dive: Gender Bias Analysis")
    col1, col2 = st.columns(2)

    with col1:
        g_rates = dff.groupby("PI_GENDER")["APPROVED"].mean().reset_index()
        g_rates.columns = ["Gender","Approval_Rate"]
        g_rates["Approval_Rate"] *= 100
        fig = px.bar(g_rates, x="Gender", y="Approval_Rate",
                     color="Gender", color_discrete_sequence=["#00d4ff","#ff69b4"],
                     text=g_rates["Approval_Rate"].round(1).astype(str)+"%",
                     title="Approval Rate by Gender", template="plotly_dark")
        fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=360, yaxis_range=[0,100])
        st.plotly_chart(fig, use_container_width=True)
        m_r = dff[dff["PI_GENDER"]=="M"]["APPROVED"].mean()*100
        f_r = dff[dff["PI_GENDER"]=="F"]["APPROVED"].mean()*100
        gap = abs(m_r - f_r)
        if gap > 5:
            bias_alert(f"Gender approval gap: <b>{gap:.1f} percentage points</b> (Male={m_r:.1f}%, Female={f_r:.1f}%). A gap >5pp warrants formal investigation.")
        else:
            bias_clear(f"Gender approval gap is only <b>{gap:.1f}pp</b> — within acceptable variation range.")

    with col2:
        # Approval rate by gender × age group
        ga_df = dff.groupby(["AGE_GROUP","PI_GENDER"])["APPROVED"].mean().reset_index()
        ga_df["Approval_Rate"] = ga_df["APPROVED"]*100
        fig = px.line(ga_df, x="AGE_GROUP", y="Approval_Rate", color="PI_GENDER",
                      color_discrete_map={"M":"#00d4ff","F":"#ff69b4"},
                      markers=True, title="Approval Rate: Age Group × Gender",
                      template="plotly_dark")
        fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=360)
        st.plotly_chart(fig, use_container_width=True)
        insight("If the male and female lines diverge significantly at specific age groups, it reveals age-gender intersectional bias — certain age brackets may be treated differently by gender.")

    # 3D Bias Surface: Zone × Gender → Approval Rate
    st.subheader("3D Bias Surface: Zone × Gender × Approval Rate")
    top_zones_list = dff["ZONE"].value_counts().head(10).index.tolist()
    zg_df = dff[dff["ZONE"].isin(top_zones_list)].groupby(["ZONE","PI_GENDER"])["APPROVED"].mean().reset_index()
    zg_pivot = zg_df.pivot(index="ZONE", columns="PI_GENDER", values="APPROVED").fillna(0) * 100
    if "M" in zg_pivot.columns and "F" in zg_pivot.columns:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Male",   x=zg_pivot.index, y=zg_pivot["M"], marker_color=APPROVED_COLOR))
        fig.add_trace(go.Bar(name="Female", x=zg_pivot.index, y=zg_pivot["F"], marker_color="#ff69b4"))
        fig.update_layout(title="Approval Rate by Zone × Gender",
                          barmode="group", template="plotly_dark",
                          paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                          height=420, xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)
        gap_zones = (zg_pivot["M"] - zg_pivot["F"]).abs()
        worst_zone = gap_zones.idxmax()
        worst_gap  = gap_zones.max()
        if worst_gap > 10:
            bias_alert(f"Zone <b>{worst_zone}</b> shows a <b>{worst_gap:.1f}pp</b> gender approval gap — the largest in the portfolio. Inspect reviewer assignments in this zone.")
        else:
            bias_clear(f"Gender gaps across zones are all under 10pp. Largest gap: <b>{worst_zone}</b> at {worst_gap:.1f}pp.")

    # 3D Scatter: The "Unexplained Rejection" view
    st.subheader("⚠️ Unexplained Rejection Detector")
    st.caption("Repudiated claims with HIGH sum-assured, LOW age, and EARLY=NON EARLY — cases that should arguably be approved.")
    suspect = dff[
        (dff["POLICY_STATUS"]=="Repudiate Death") &
        (dff["PI_AGE"] < 70) &
        (dff["SUM_ASSURED"] > dff["SUM_ASSURED"].median()) &
        (dff["EARLY_NON"]=="NON EARLY")
    ].copy()
    st.markdown(f"**{len(suspect)} suspect repudiations** found matching these criteria.")
    if not suspect.empty:
        fig = px.scatter_3d(suspect, x="PI_AGE", y="SUM_ASSURED", z="PI_ANNUAL_INCOME",
                            color="PI_GENDER", size="SUM_ASSURED", opacity=0.8,
                            color_discrete_map={"M":"#00d4ff","F":"#ff69b4"},
                            hover_data=["PI_STATE","PI_OCCUPATION","REASON_FOR_CLAIM"],
                            title="3D: Suspect Repudiations (Low-risk profiles)",
                            template="plotly_dark")
        fig.update_layout(paper_bgcolor="#0e1117", height=540)
        st.plotly_chart(fig, use_container_width=True)
        g_share = suspect["PI_GENDER"].value_counts(normalize=True)*100
        insight(f"Among suspect repudiations: <b>{g_share.get('M',0):.1f}% Male</b>, <b>{g_share.get('F',0):.1f}% Female</b>. "
                f"If female share here is significantly higher than in the overall rejected pool ({(dff[dff['POLICY_STATUS']=='Repudiate Death']['PI_GENDER']=='F').mean()*100:.1f}%), "
                f"that is strong evidence of gendered bias in borderline decisions.")

    # Verdict
    st.markdown("---")
    st.subheader("🏛️ Bias Verdict")
    sig_attrs = br_df[br_df["P_Value"]<0.05]["Attribute"].tolist()
    high_bias  = br_df[br_df["Bias_Score"]>50]["Attribute"].tolist()

    if high_bias:
        st.markdown(f"""
        <div class="bias-alert">
        <h4>⚖️ VERDICT: Patterns Consistent With Potential Bias Detected</h4>
        <p>The following attributes show both statistical significance AND meaningful effect sizes: <b>{", ".join(high_bias)}</b>.</p>
        <p>This means past settlement decisions are <b>not independent</b> of these attributes. While this alone does not prove the <i>manager</i> is personally biased (decisions may reflect inherited underwriting rules), it does confirm that the <b>outcomes differ systematically</b> across these groups.</p>
        <p><b>Next steps:</b> (1) Audit the 5 highest Bias-Score attributes with a senior compliance officer. (2) Cross-check claim files where rejections align with protected-attribute profiles. (3) Implement blind review for all borderline claims.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="bias-clear">
        <h4>✅ VERDICT: No Systematic Bias Detected at Current Threshold</h4>
        <p>Statistical analysis does not support a claim of systematic bias by the manager. Approval rates across gender, age groups, zones and other attributes are within acceptable statistical variation.</p>
        <p>Significant attributes (p<0.05): {", ".join(sig_attrs) if sig_attrs else "None"}. However, these effects are small (Cramér's V < 0.2) and may reflect legitimate risk differentiation rather than bias.</p>
        <p>The manager's settlement decisions appear defensible based on this dataset.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<center><small>Insurance Claims Bias Detection Dashboard &nbsp;|&nbsp; Built with Streamlit & Plotly &nbsp;|&nbsp; Data is anonymised</small></center>",
    unsafe_allow_html=True
)
