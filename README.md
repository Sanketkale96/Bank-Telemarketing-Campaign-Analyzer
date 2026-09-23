# 🏦 Bank Telemarketing Campaign Effectiveness Analyzer

> **IBM Internship Capstone Project — AI & Data Science**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange)](https://scikit-learn.org)
[![Dataset](https://img.shields.io/badge/Dataset-UCI_Bank_Marketing-green)](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

---

## 📌 Project Overview

This project analyses the **direct marketing campaigns** of a Portuguese banking institution
conducted via telephone calls. The goal is to determine which clients are likely to subscribe
to a **bank term deposit**, enabling the bank to optimise its marketing strategy, reduce costs,
and increase conversion rates.

The project is implemented as a fully interactive **Streamlit dashboard** featuring
a complete **4-Tier Analytics Pipeline**: Descriptive → Diagnostic → Predictive → Prescriptive.

---

## 🎯 Problem Statement

A Portuguese bank runs telephone marketing campaigns to sell term deposit products.
Each campaign involves multiple calls per client, incurring significant cost.
The bank needs to:
1. Understand which client profiles and campaign conditions lead to subscriptions
2. Predict the probability that a given client will subscribe before calling
3. Prioritise their call lists to maximise ROI and reduce wasted contacts

---

## 🧭 Objectives

1. **Descriptive**: Summarise historical campaign outcomes (who subscribed, when, how)
2. **Diagnostic**: Identify the root causes and key drivers of subscription behaviour
3. **Predictive**: Train a machine learning model to score client subscription probability
4. **Prescriptive**: Provide data-driven recommendations to improve future campaigns

---

## 📊 Dataset Description

| Attribute | Value |
|-----------|-------|
| Source | UCI Machine Learning Repository |
| Citation | Moro et al. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems. |
| File Used | `bank-additional-full.csv` |
| Raw Records | 41,188 |
| Clean Records (after dedup) | 41,176 (12 duplicates removed) |
| Features | 21 (20 input + 1 target) |
| Target Variable | `y` — has the client subscribed a term deposit? (yes/no) |
| Subscription Rate | 11.27% (4,639 subscriptions) |
| Time Period | May 2008 – November 2010 |
| Missing Values | None (unknowns encoded as "unknown") |

### Feature Groups

**Client Data:**
`age`, `job`, `marital`, `education`, `default`, `housing`, `loan`

**Campaign Contact:**
`contact`, `month`, `day_of_week`, `duration`, `campaign`, `pdays`, `previous`, `poutcome`

**Social / Economic Context:**
`emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`

---

## 🛠️ Technologies Used

| Library | Purpose |
|---------|---------|
| Python 3.10+ | Primary language |
| Pandas | Data loading, manipulation, aggregation |
| NumPy | Numerical computation |
| Scikit-learn | Machine learning (Random Forest, metrics, in-memory training) |
| Plotly | Interactive visualisations |
| Streamlit | Interactive web dashboard |

---

## 📁 Project Structure

```
bank-telemarketing-campaign-analyzer/
├── app.py
├── README.md
├── requirements.txt
└── Sanket_ProjectReport.docx
```

> **No `data/`, `models/`, or `src/` folders required.**
> All preprocessing, analytics, machine learning, and dashboard logic is contained within `app.py`.
> The UCI Bank Marketing dataset is downloaded automatically at first run — no manual download needed.

---

## 🔄 Data Preprocessing

| Step | Action |
|------|--------|
| Duplicate removal | 12 exact duplicate rows removed (41,188 → 41,176) |
| Column renaming | Dots in names replaced with underscores |
| Target encoding | `y` → `y_binary` (1 = subscribed, 0 = not subscribed) |
| 'Unknown' handling | Replaced with `NaN`, then imputed with column mode |
| `pdays = 999` | Replaced with `NaN` (means client was not previously contacted) |
| Feature flag | `was_contacted_before` binary flag created from `pdays` |
| Education ordinal | Mapped to ordinal scale (0 = illiterate → 6 = university degree) |
| Feature engineering | `call_duration_min` (duration / 60), `contact_rate_category` |
| Ordered categoricals | `month`, `day_of_week` cast to ordered Categorical types |

---

## 📈 Analytics Methodology

### 📋 Tier 1: Descriptive Analytics
- Subscription outcome distribution
- Monthly contact volume and subscription rate
- Subscription rate by job, education, marital status, age
- Contact method distribution
- Day-of-week patterns

### 🔬 Tier 2: Diagnostic Analytics
- Call duration vs subscription (strongest correlation)
- Previous campaign outcome effect
- Economic indicator impact (Euribor, employment variation)
- Loan status and credit default effect
- Campaign contact frequency effect
- Full numerical correlation heatmap

### 🤖 Tier 3: Predictive Analytics
- **Algorithm**: Random Forest Classifier (`n_estimators=200`, `max_depth=12`)
- **Class balancing**: `class_weight="balanced"`
- **Split**: 80% train (32,940) / 20% test (8,236), stratified
- **Features**: 40 features (10 numerical + 30 one-hot encoded from 8 categorical variables)
- **Note**: `duration` excluded to prevent data leakage
- **Results**: ROC-AUC = **0.8131** | F1 = **0.4979** | Avg Precision = **0.4875** | Accuracy = **85.38%**
- **Interactive predictor**: Single-client probability estimation

### 💡 Tier 4: Prescriptive Analytics
- Top client segments by subscription rate (job × education × contact)
- Optimal call timing heatmap (month × day of week)
- Contact frequency strategy (stop at 3 contacts)
- Economic monitoring dashboard
- 7 data-backed business recommendations

---

## 🔑 Key Findings

1. **Only 11.27%** of contacted clients subscribed (4,639 / 41,176 clean records)
2. **Call duration** is the strongest predictor (r = 0.41) — subscribers average 9.22 min vs 3.68 min
3. **Previous successful outcome** clients convert at **65.11%** vs 8.83% for never-contacted
4. **Cellular contact (14.74%)** nearly triples telephone contact rate (5.23%)
5. **Students (31.43%) and retirees (25.26%)** show the highest job-segment conversion rates
6. **Low Euribor rates** (≤ 2%) yield 24.46% subscription rate vs 4.84% when Euribor > 4%
7. **Diminishing returns** after 3 contacts — rate drops from 13.04% (1 contact) to 7.5% (5 contacts)
8. **March (50.55%)** shows the highest monthly subscription rate

---

## ✅ Business Recommendations

1. Prioritise clients with prior successful campaign history
2. Focus on cellular contact over telephone
3. Cap contacts at 3 per client per campaign
4. Time campaigns to low-Euribor economic environments
5. Target high-performing months (March, September, October, December)
6. Develop tailored messaging for students and retired clients
7. Use the predictive model to score and rank prospects before calling

---

## 📥 Dataset

The application uses the **UCI Bank Marketing (bank-additional-full.csv)** dataset (Moro et al., 2014).

- **Automatic download**: On first launch, `app.py` automatically downloads the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/static/public/222/bank+marketing.zip), extracts the nested ZIP archive (`bank+marketing.zip` → `bank-additional.zip` → `bank-additional-full.csv`), and loads it into memory.
- **No manual setup required**: There is no `data/` folder to create and no CSV file to place manually.
- If a local copy is already present, it is used directly; otherwise the download runs automatically.

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/bank-telemarketing-campaign-analyzer.git
cd bank-telemarketing-campaign-analyzer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

The dashboard opens automatically at `http://localhost:8501`.
On first launch, the dataset is downloaded and the Random Forest model is trained in memory (~60 s).
Subsequent launches reuse the Streamlit-cached data and model instantly.

---

## 💻 How to Use the Application

1. **Home** — Project overview and navigation guide
2. **Dataset Overview** — Browse raw data, schema, and quality report
3. **KPI Dashboard** — Key metrics with sidebar filters (month, job, contact, age)
4. **Exploratory Analysis** — Distributions, relationships, correlation heatmap
5. **Descriptive Analytics** — Historical campaign summaries
6. **Diagnostic Analytics** — Root cause analysis
7. **Predictive Analytics** — Model performance + individual client predictor
8. **Prescriptive Insights** — Segment targeting, timing, contact strategy
9. **Business Recommendations** — Actionable 7-point strategy

### Sidebar Filters
All dashboard pages respond to the global sidebar filters:
- 📅 Month(s)
- 💼 Job Type(s)
- 📞 Contact Method(s)
- 🧑 Age Range

---

## 🔮 Future Improvements

1. Integrate real-time client database for live scoring
2. Add SHAP explainability values per prediction
3. Explore XGBoost / LightGBM for improved model performance
4. Add A/B testing module for campaign variant comparison
5. Time-series analysis of subscription trends
6. Geographic segmentation if regional data becomes available
7. Deploy to Streamlit Community Cloud / IBM Cloud

---

## 📚 Citation

> S. Moro, P. Cortez and P. Rita.
> *A Data-Driven Approach to Predict the Success of Bank Telemarketing.*
> Decision Support Systems, In press.
> http://dx.doi.org/10.1016/j.dss.2014.03.001

---

## 👤 Author

Sanket Kale

**IBM Internship Project**
AI and Data Science Engineering Student
IBM Internship Programme

---

*This project was developed as part of the IBM Internship Capstone submission.
All analysis is based on the actual dataset — no data or results have been fabricated.*
