# AI-Powered GitHub Self-Analysis

Analyze your GitHub digital footprint using **local LLMs** and traditional data science techniques.

## Demo Video

Watch the Demo:- https://youtu.be/kv3Vd-hu0kM

## Key Findings (Top 5 Insights)

1. **Coding Schedule**: Activity heatmap reveals peak productivity windows — most commits cluster on weekday evenings, suggesting a student/side-project workflow.
2. **Language Evolution**: The repository timeline shows a clear progression from beginner-friendly languages to more specialized tools, mapping a learning journey.
3. **Commit Sentiment**: LLM analysis shows commits are predominantly neutral/task-oriented, with frustration spikes correlating to deadline periods.
4. **Portfolio Gaps**: Despite strong backend/data skills, the portfolio lacks frontend and deployment projects — a common gap for CS students.
5. **Growth Trajectory**: Repository creation rate is accelerating, with forecasting suggesting continued upward momentum.

> _Note: These are template findings — your actual insights will be generated when you run the analysis on your own data._

## Model Comparison Table

| Metric         | Llama 3.1 (8B) | Mistral (7B)    |
| -------------- | -------------- | --------------- |
| Avg Latency    | ~15-30s        | ~10-20s         |
| Tokens/sec     | ~15-25         | ~20-35          |
| Output Quality | More detailed  | More concise    |
| Cost           | $0.00 (local)  | $0.00 (local)   |
| Best For       | Deep analysis  | Quick summaries |

## Setup Instructions

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/YOUR_USERNAME/github-self-analysis.git
cd github-self-analysis
pip install -r requirements.txt
```

### 2. Set Up Local LLM (Ollama)

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull at least 2 models for comparison
ollama pull llama3.1
ollama pull mistral

# Verify Ollama is running
ollama list
```

### 3. Collect Your GitHub Data

```bash
# Without token (60 requests/hr limit)
python collect_data.py --username YOUR_GITHUB_USERNAME

# With token (recommended — 5000 requests/hr)
python collect_data.py --username YOUR_GITHUB_USERNAME --token YOUR_GITHUB_TOKEN
```

Generate a token at https://github.com/settings/tokens (no special scopes needed for public repos).

### 4. Run the Analysis Notebook

```bash
cd notebooks
jupyter notebook analysis.ipynb
```

Run all cells sequentially. Ensure Ollama is running (`ollama serve`).

### 5. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser.

## Project Structure

```
github-self-analysis/
├── collect_data.py          # GitHub API data collector
├── llm_analysis.py          # LLM interface (Ollama/OpenAI/Anthropic)
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── Prompts.md               # Prompt engineering documentation
├── data/                    # Collected data (gitignored)
│   ├── profile.json
│   ├── repos.json
│   ├── commits.json
│   ├── languages.json
│   ├── events.json
│   ├── readmes.json
│   └── file_listings.json
├── notebooks/
│   └── analysis.ipynb       # Main analysis notebook
├── dashboard/
│   └── app.py               # Streamlit dashboard
└── docs/
    └── technical_report.pdf # Technical write-up
```

## Analysis Tasks Completed

### LLM-Powered (7 tasks)

1.  Sentiment analysis of commit messages
2.  Topic clustering of repositories
3.  Skill extraction from repos
4.  Documentation quality assessment
5.  Naming convention analysis
6.  Career progression narrative
7.  "What should you build next?" recommendations

### Traditional Data Science

- 6+ visualizations (languages, timeline, heatmap, distributions, word freq, repo growth)
- Statistical analysis (distributions, correlations, trends)
- K-Means clustering of repositories
- Linear regression time series forecasting

### Model Comparison

- Llama 3.1 vs Mistral on identical prompts
- Latency, throughput, and quality metrics
- Side-by-side output comparison

## Technologies Used

- **Local LLMs**: Ollama (Llama 3.1, Mistral)
- **Data Collection**: GitHub REST API, requests
- **Analysis**: pandas, numpy, scikit-learn, matplotlib, seaborn
- **Dashboard**: Streamlit, Plotly
- **ML**: K-Means clustering, TF-IDF, PCA, Linear Regression
