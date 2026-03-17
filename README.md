<div align="center">

# AI-Powered GitHub Self-Analysis

> Treat your GitHub profile as a dataset. This pipeline collects your full commit history, repository metadata, and code fingerprint — then runs **local LLMs** and classical ML to surface insights a recruiter or career counselor would take hours to compile.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?logo=ollama)](https://ollama.ai)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>


## What It Does

This project runs a **full data science pipeline** on any GitHub user's public profile:

| Stage | What Happens |
|---|---|
| **Collect** | GitHub REST API → 7 structured JSON files (repos, commits, languages, READMEs, …) |
| **Analyze** | 7 LLM tasks + K-Means clustering + TF-IDF + PCA + linear regression forecasting |
| **Visualize** | 9 static charts + 5-page interactive Streamlit dashboard with Plotly |
| **Compare** | Side-by-side Llama 3.1 vs. Mistral on identical prompts with live latency metrics |

---

## Key Metrics

| Metric | Value |
|---|---|
| Repositories analyzed | 9 |
| Commits analyzed | 38 |
| LLM analysis tasks | 7 |
| Local models compared | 2 (Llama 3.1 · Mistral) |
| K-Means clusters found | 4 |
| Visualizations generated | 9 |
| Forecast horizon | 6 months |
| Cloud inference cost | **$0.00** |

---

## Model Performance

| Metric | Llama 3.1 (8B) | Mistral (7B) |
|---|---|---|
| Avg latency | ~15–30 s | ~10–20 s |
| Throughput | ~15–25 tok/s | ~20–35 tok/s |
| Output style | Detailed / verbose | Concise / structured |
| Cost | $0.00 (local) | $0.00 (local) |
| Best for | Deep narrative analysis | Quick summaries |

---

## Key Findings

Analysis of **Khushipatel27's** GitHub profile produced these data-backed insights:

1. **Coding Schedule** — Activity heatmap shows peak commits on weekday evenings, consistent with a student project workflow.
2. **Language Evolution** — Python dominates by file count; Jupyter Notebooks account for ~1.8 MB of the total codebase, signaling a shift toward applied data science.
3. **Commit Sentiment** — LLM 4-way classification: ~60% positive, ~32% neutral, ~0% negative, with frustration spikes correlating to deadline periods.
4. **Portfolio Clustering** — K-Means (k=4) groups repos into: Data Management, ML/Computer Vision, Data Analysis, and Advanced Analytics/Forecasting.
5. **Growth Trajectory** — Linear regression on monthly commit counts yields slope ≈ +0.3 commits/month (R² = 0.065), indicating an early-stage but upward trend.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   GitHub REST API                   │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              collect_data.py                        │
│  profile · repos · commits · languages ·            │
│  events · READMEs · file listings                   │
└─────────────────────┬───────────────────────────────┘
                      │  7 JSON files → data/
                      ▼
          ┌───────────────────────┐
          │   notebooks/          │
          │   analysis.ipynb      │
          │                       │
          │  ┌─────────────────┐  │
          │  │  LLM Analysis   │  │◄── llm_analysis.py
          │  │  (7 tasks)      │  │     (Ollama · OpenAI · Anthropic)
          │  └────────┬────────┘  │
          │           │           │
          │  ┌────────▼────────┐  │
          │  │  ML Techniques  │  │
          │  │  K-Means · PCA  │  │
          │  │  TF-IDF · LR    │  │
          │  └────────┬────────┘  │
          └───────────┼───────────┘
                      │  9 PNGs → data/
                      ▼
┌─────────────────────────────────────────────────────┐
│              dashboard/app.py                       │
│  Overview · Activity · LLM Insights ·               │
│  ML Analysis · Model Comparison                     │
│  (Streamlit + Plotly — live Ollama calls)           │
└─────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data collection | GitHub REST API, `requests` |
| LLM inference | Ollama (Llama 3.1, Mistral), OpenAI-compatible |
| NLP / ML | scikit-learn (K-Means, TF-IDF, PCA, Linear Regression) |
| Data wrangling | pandas, numpy |
| Static viz | matplotlib, seaborn |
| Interactive viz | Plotly |
| Dashboard | Streamlit |
| Notebook | Jupyter |

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- A GitHub account (token optional but recommended)

### 1. Clone & install

```bash
git clone https://github.com/Khushipatel27/github-self-analysis.git
cd github-self-analysis
pip install -r requirements.txt
```

### 2. Pull LLM models

```bash
ollama pull llama3.1
ollama pull mistral
ollama serve          # keep this running in a separate terminal
```

### 3. Collect your GitHub data

```bash
# Public repos only (60 req/hr)
python collect_data.py --username YOUR_GITHUB_USERNAME

# Recommended: add a token for 5,000 req/hr
python collect_data.py --username YOUR_GITHUB_USERNAME --token YOUR_GITHUB_TOKEN
```

Generate a token at https://github.com/settings/tokens — no special scopes needed for public repos.

### 4. Run the analysis notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

Run all cells sequentially. Ensure `ollama serve` is active.

### 5. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501). Use the sidebar to switch pages and select your LLM model.

---

## Project Structure

```
github-self-analysis/
├── collect_data.py          # GitHub API collector (rate-limit aware)
├── llm_analysis.py          # LLM interface: Ollama / OpenAI / Anthropic
├── Prompts.md               # Prompt engineering documentation & design rationale
├── requirements.txt
├── .env.example             # Environment variable template
├── notebooks/
│   └── analysis.ipynb       # End-to-end analysis notebook
├── dashboard/
│   └── app.py               # 5-page Streamlit dashboard
├── data/                    # Auto-generated by collect_data.py
│   ├── profile.json
│   ├── repos.json
│   ├── commits.json
│   ├── languages.json
│   ├── events.json
│   ├── readmes.json
│   ├── file_listings.json
│   └── Visual_*.png         # Charts generated by the notebook
└── docs/
    └── technical_report.pdf
```

---

## Analysis Tasks

### LLM-Powered (7 tasks via Ollama)

| # | Task | LLM Persona | Output |
|---|---|---|---|
| 1 | Commit sentiment analysis | Developer psychology expert | 4-class % breakdown |
| 2 | Repository topic clustering | Technical recruiter | Thematic groups + gaps |
| 3 | Skill extraction | Senior tech lead | Proficiency tiers by domain |
| 4 | Documentation quality rating | Developer experience expert | 4-dimension 1–10 scores |
| 5 | Naming convention audit | Code quality consultant | Issues + recommendations |
| 6 | Career progression narrative | Tech career counselor | Timeline + predicted pivots |
| 7 | Next project recommendations | Tech mentor | 3 skill-builders · 2 stretch · 1 portfolio-booster |

### Traditional Data Science

- **Clustering**: K-Means (k=4) on TF-IDF + numeric features, visualized with PCA
- **Forecasting**: Linear regression on monthly commit counts, 6-month horizon
- **Visualizations**: language distribution, commit timeline, coding heatmap, size distributions, commit word frequency, repo growth, cluster scatter, forecast, model comparison

### Model Comparison

Live, side-by-side execution of Llama 3.1 and Mistral on identical prompts with latency, token count, and throughput metrics.

---

## Roadmap

- [ ] Add support for private repos (with appropriate token scopes)
- [ ] Export findings as a shareable PDF report
- [ ] Deploy dashboard to Streamlit Community Cloud
- [ ] Add GPT-4o / Claude Haiku as optional cloud backends
- [ ] GitHub Actions workflow to auto-refresh data weekly

---

## License

MIT — use this to analyze your own profile, adapt it for team analysis, or extend it for research.
