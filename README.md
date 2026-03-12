# 🔍 Insurance Claims — Bias Detection & Analytics Dashboard

A production-grade **Streamlit** dashboard that provides 7 analytics tabs to help insurance managers prove (or disprove) bias in claim settlement decisions.

---

## 📊 Dashboard Tabs

| Tab | Type | Description |
|-----|------|-------------|
| 📊 Descriptive | Descriptive Analytics | What happened? KPIs, distributions, 3D scatter |
| 🔬 Diagnostic | Diagnostic Analytics | Why did it happen? Root-cause heatmaps, occupation rates |
| 💡 Prescriptive | Prescriptive Analytics | What should we do? Feature importance + recommendations |
| 🤖 Classification | ML Classification | Random Forest, Gradient Boost, Logistic Regression + ROC |
| 🔵 Clustering | K-Means Clustering | 3D PCA clusters, elbow curve, cluster profiles |
| 🔗 Association Rules | Apriori ARM | 3D rule scatter, network graph, rule tables |
| ⚠️ Bias Detection | **Bias Detection System** | Chi-square tests, Cramér's V, bias scores, verdict |

---

## ⚡ Quick Start

### Local
```bash
pip install -r requirements.txt
# Put Insurance__1_.csv in the same folder as app.py
streamlit run app.py
```

### Deploy to Streamlit Cloud
1. Fork / upload this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy**

> ⚠️ Make sure `Insurance__1_.csv` is in the **root of the repo** (same level as `app.py`)

---

## 📁 File Structure

```
insurance_dashboard/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── Insurance__1_.csv       # Dataset (place here before running)
└── README.md
```

---

## 🔬 Bias Detection Methodology

The bias detection tab uses:

- **Chi-Square Test of Independence** — tests if claim outcomes are independent of each attribute
- **Cramér's V** — effect size measure (0 = no association, 1 = perfect association)
- **Bias Score (0–100)** — composite score combining p-value significance and effect size
- **Unexplained Rejection Detector** — flags low-risk profiles that were repudiated
- **Gender × Zone × Age Cross-tabs** — intersection analysis

### Interpretation
| Bias Score | Interpretation |
|------------|----------------|
| 0–25       | No concern |
| 25–50      | Minor variation, monitor |
| 50–75      | Significant pattern, investigate |
| 75–100     | Strong bias evidence, escalate |

---

## 📦 Dependencies

- `streamlit` — dashboard framework
- `plotly` — 3D interactive charts
- `pandas` / `numpy` — data processing
- `scikit-learn` — ML classification + clustering
- `mlxtend` — association rule mining (Apriori)
- `scipy` — statistical tests (chi-square, Fisher exact)

---

## ⚖️ Disclaimer

Statistical association between an attribute and outcome does **not** automatically constitute proof of intentional bias. Results should be reviewed by a qualified compliance officer.
