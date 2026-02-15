"""
GitHub Self-Analysis Dashboard
================================
This is the interactive dashboard for the AI-Powered GitHub Self-Analysis project.
It visualizes all the data collected from my GitHub profile and lets me run
LLM-powered analyses in real time through a web interface.

I chose Streamlit because it's the fastest way to build data dashboards in Python
without needing any frontend/HTML knowledge. It converts Python scripts directly
into interactive web apps.

The dashboard has 5 pages:
    1. Overview 
    2. Activity 
    3. LLM Insights 
    4. ML Analysis 
    5. Model Comparison 

Usage: streamlit run dashboard/app.py
This opens a local web server at http://localhost:8501
"""

# Imports
# streamlit is the web framework that renders everything as a dashboard
# plotly gives us interactive charts (hover, zoom, pan) instead of static matplotlib
# pandas/numpy handle data manipulation
# Counter and defaultdict help with aggregating language and word frequency data
import streamlit as st
import json, sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

# adding the parent directory to the path so we can import our custom LLM module
# this is needed because the dashboard lives in a subfolder (dashboard/)
# while llm_analysis.py lives in the project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_analysis import LLMAnalyzer, LLMResult

#  Page Configuration 
# setting the browser tab title, icon, and using wide layout so charts
# have more horizontal space to display properly
st.set_page_config(page_title="GitHub Self-Analysis", layout="wide")

# pointing to the data directory where collect_data.py saved all the JSON files
DATA_DIR = Path(__file__).parent.parent / "data"


#  Data Loading Functions 
# using @st.cache_data so the JSON files are only read once from disk
# even if the user refreshes the page or switches between tabs
# without this, every page navigation would re-read all 7 files
@st.cache_data
def load_data():
    """Load all 7 JSON files that collect_data.py generated."""
    data = {}
    for fname in ["profile","repos","commits","languages","events","readmes","file_listings"]:
        fpath = DATA_DIR / f"{fname}.json"
        if fpath.exists():
            with open(fpath) as f:
                data[fname] = json.load(f)
        else:
            # if a file is missing, use empty list or dict as fallback
            # so the dashboard doesn't crash, it just shows empty sections
            data[fname] = [] if fname in ["repos","commits","events"] else {}
    return data


def parse_repos(raw):
    """
    Converting the raw repos JSON into a pandas DataFrame.
    Each repo becomes a row with columns like name, language, stars, etc.
    This makes it easy to filter, sort, and visualize the data.
    """
    if not raw: return pd.DataFrame()
    return pd.DataFrame([{
        "name": r["name"],
        "description": r.get("description",""),
        "language": r.get("language","Unknown"),
        "stars": r.get("stargazers_count",0),
        "forks": r.get("forks_count",0),
        "size_kb": r.get("size",0),
        "created_at": pd.to_datetime(r["created_at"]),
        "updated_at": pd.to_datetime(r["updated_at"]),
        "pushed_at": pd.to_datetime(r.get("pushed_at")),
        "open_issues": r.get("open_issues_count",0),
        "is_fork": r.get("fork",False),
        "topics": r.get("topics",[]),
    } for r in raw])


def parse_commits(raw):
    """
    Convert raw commits JSON into a DataFrame.
    Extracts the commit message, date, repo name, and short SHA hash.
    The date is parsed into a proper datetime so we can do time-based analysis.
    """
    if not raw: return pd.DataFrame()
    rows = []
    for c in raw:
        ci = c.get("commit",{})
        ai = ci.get("author",{})
        rows.append({
            "sha": c.get("sha","")[:8],
            "repo": c.get("_repo",""),
            "message": ci.get("message",""),
            "date": pd.to_datetime(ai.get("date"))
        })
    return pd.DataFrame(rows)


#  Load and Parse All Data 
# this runs once when the dashboard starts up
data = load_data()
repos_df = parse_repos(data["repos"])
commits_df = parse_commits(data["commits"])
profile = data.get("profile",{})
languages_data = data.get("languages",{})
readmes = data.get("readmes",{})


#  Sidebar 
# the sidebar stays visible on every page and shows my profile info
# plus the LLM configuration settings
st.sidebar.title("GitHub Self-Analysis")

# show my GitHub avatar and basic profile info at the top
if profile:
    st.sidebar.image(profile.get("avatar_url",""), width=100)
    st.sidebar.markdown(f"### {profile.get('name', profile.get('login','User'))}")
    if profile.get("bio"):
        st.sidebar.markdown(f"*{profile['bio']}*")

st.sidebar.markdown("---")

# LLM settings - the user can change the Ollama URL if it's running
# on a different port, and select which two models to use
# i kept only llama3.1 and mistral since those are the two i pulled
st.sidebar.markdown("### LLM Settings")
ollama_url = st.sidebar.text_input("Ollama URL", "http://localhost:11434")
model_a = st.sidebar.selectbox("Model A", ["llama3.1","mistral"])
model_b = st.sidebar.selectbox("Model B", ["mistral","llama3.1"])

# navigation radio buttons to switch between the 5 dashboard pages
page = st.sidebar.radio("Navigate", [
    "Overview", "Activity", "LLM Insights", "ML Analysis", "Model Comparison"
])



# PAGE 1: OVERVIEW
# Shows high-level metrics and language breakdown.
# This gives a quick snapshot of my entire GitHub presence.

if page == "Overview":
    st.title("GitHub Profile Overview")

    # if there's no data, show an error and stop rendering the rest of the page
    if repos_df.empty:
        st.error("No data found. Run `python collect_data.py --username YOUR_USERNAME` first.")
        st.stop()

    # four metric cards across the top showing the key numbers
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Repositories", len(repos_df))
    c2.metric("Total Commits", len(commits_df))
    c3.metric("Total Stars", int(repos_df["stars"].sum()))
    c4.metric("Languages", repos_df["language"].nunique())

    # two charts side by side: language by repo count and language by code volume
    left, right = st.columns(2)

    with left:
        # horizontal bar chart showing which languages i use most often
        # this counts how many repos use each language
        st.subheader("Languages by Repo Count")
        lc = repos_df["language"].value_counts().head(10)
        fig = px.bar(x=lc.values, y=lc.index, orientation="h",
                     color=lc.values, color_continuous_scale="viridis")
        fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Repos",
                         coloraxis_showscale=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        # pie chart showing total bytes of code written in each language
        # this gives a different perspective - a repo might be listed as Python
        # but actually have more HTML/CSS by volume
        st.subheader("Languages by Code Volume")
        tb = defaultdict(int)
        for rl in languages_data.values():
            for lang, b in rl.items():
                tb[lang] += b
        lb = pd.Series(tb).sort_values(ascending=False).head(10)
        fig = px.pie(names=lb.index, values=lb.values, hole=0.4)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # sortable table of all repos so i can browse them
    st.subheader("Repository Details")
    display_cols = ["name","language","stars","forks","size_kb","created_at"]
    st.dataframe(repos_df[display_cols].sort_values("stars", ascending=False), use_container_width=True)



# PAGE 2: ACTIVITY
# Shows temporal patterns in my coding - when i code, how often,
# and how my commit messages look. This is all traditional data
# science (no LLM needed).

elif page == "Activity":
    st.title("Activity Timeline")

    if commits_df.empty:
        st.warning("No commit data available.")
        st.stop()

    #  Monthly Commit Activity 
    # grouping commits by month to see the overall trend
    # using plotly Scatter with fill to create an area chart
    commits_df["month"] = commits_df["date"].dt.to_period("M")
    monthly = commits_df.groupby("month").size().reset_index(name="commits")
    monthly["month"] = monthly["month"].dt.to_timestamp()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["month"], y=monthly["commits"],
        mode="lines+markers", fill="tozeroy",
        name="Commits", line=dict(color="#2ecc71")
    ))
    fig.update_layout(title="Monthly Commit Activity", xaxis_title="", yaxis_title="Commits", height=400)
    st.plotly_chart(fig, use_container_width=True)

    #  Coding Schedule Heatmap 
    # this shows what day and hour i typically commit code
    # rows are days of the week, columns are hours (0-23)
    # darker colors mean more commits at that time slot
    st.subheader("Coding Schedule Heatmap")
    commits_df["dow"] = commits_df["date"].dt.day_name()
    commits_df["hour"] = commits_df["date"].dt.hour
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    hm = commits_df.groupby(["dow","hour"]).size().unstack(fill_value=0).reindex(day_order)

    fig = px.imshow(hm, color_continuous_scale="YlOrRd", aspect="auto",
                    labels=dict(x="Hour", y="Day", color="Commits"))
    fig.update_layout(title="When Do You Code?", height=350)
    st.plotly_chart(fig, use_container_width=True)

    #  Repository Growth 
    # a cumulative line chart showing how many total repos i had over time
    # this shows whether i'm creating repos at an increasing or decreasing rate
    st.subheader("Repository Growth")
    rds = repos_df.sort_values("created_at").copy()
    rds["cumulative"] = range(1, len(rds)+1)
    fig = px.line(rds, x="created_at", y="cumulative", markers=True)
    fig.update_layout(xaxis_title="", yaxis_title="Total Repos", height=350)
    st.plotly_chart(fig, use_container_width=True)

    #  Commit Message Analytics 
    # two charts: message length distribution and most common words
    # this helps understand my commit writing habits
    st.subheader("Commit Message Analytics")
    msg_lens = commits_df["message"].str.split("\n").str[0].str.len()
    col1, col2 = st.columns(2)

    with col1:
        # histogram of how long my commit messages are (first line only)
        fig = px.histogram(msg_lens, nbins=30, title="Message Length Distribution")
        fig.update_layout(xaxis_title="Characters", yaxis_title="Count", showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # bar chart of most frequently used words in commit messages
        # filtering out common english stop words so we see meaningful terms
        stop = {"the","a","an","and","or","in","on","at","to","for","of","is","it",
                "this","that","with","from","by","as"}
        words = []
        for msg in commits_df["message"]:
            fl = msg.split("\n")[0].lower()
            words.extend([w.strip(".,!?()[]{}:") for w in fl.split() if len(w) > 2 and w not in stop])
        wf = Counter(words).most_common(12)
        if wf:
            wdf = pd.DataFrame(wf, columns=["word","count"])
            fig = px.bar(wdf, x="count", y="word", orientation="h", title="Top Commit Words")
            fig.update_layout(yaxis_title="", height=300)
            st.plotly_chart(fig, use_container_width=True)



# PAGE 3: LLM INSIGHTS
# This is where the local LLM integration happens. The user picks
# one of 7 analysis tasks from a dropdown, clicks "Run Analysis",
# and the prompt is sent to Ollama in real time. The response and
# performance metrics are displayed on the page.

# I used this approach instead of pre-computed results so that
# the dashboard is interactive and the grader can see the LLM
# working live during the demo video.

elif page == "LLM Insights":
    st.title("LLM-Powered Insights")
    st.info(f"Using **{model_a}** via Ollama at `{ollama_url}`")

    # creating the LLM analyzer instance with the selected model
    # this connects to Ollama running locally
    llm = LLMAnalyzer(provider="ollama", model=model_a, ollama_url=ollama_url)

    # dropdown to pick which analysis task to run
    task = st.selectbox("Select Analysis", [
        "Sentiment Analysis", "Topic Clustering", "Skill Extraction",
        "Documentation Quality", "Naming Conventions", "Career Narrative", "Next Project Ideas"
    ])

    if st.button("Run Analysis", type="primary"):
        # building the appropriate prompt based on which task was selected
        # each task injects real data from my GitHub profile into the prompt
        with st.spinner(f"Running {task} with {model_a}..."):

            if task == "Sentiment Analysis":
                # sending my commit messages to the LLM to classify their tone
                sample = commits_df["message"].str.split("\n").str[0].head(40).tolist()
                prompt = f"Analyze sentiment of these commit messages:\n" + "\n".join(f"{i+1}. {m}" for i,m in enumerate(sample))
                prompt += "\n\nGive: 1) sentiment breakdown %, 2) patterns, 3) developer personality"
                sys_p = "You are a developer psychology expert."

            elif task == "Topic Clustering":
                # sending repo names and descriptions for the LLM to group into themes
                rs = [f"- {r['name']}: {r.get('description','')} ({r.get('language','')})" for r in data["repos"][:25]]
                prompt = f"Group these repos into thematic clusters:\n" + "\n".join(rs)
                prompt += "\n\nGive: clusters with repo lists, expertise areas, gaps"
                sys_p = "You are a technical recruiter."

            elif task == "Skill Extraction":
                # sending language data and repo names to build a skill profile
                prompt = f"Extract skills from this GitHub data:\nLanguages: {json.dumps(dict(list(languages_data.items())[:15]))}"
                prompt += f"\nRepos: " + ", ".join(r["name"] for r in data["repos"][:20])
                prompt += "\n\nGive: skills with proficiency levels, domain knowledge, next skills to learn"
                sys_p = "You are a senior tech lead."

            elif task == "Documentation Quality":
                # sending README content for quality evaluation
                samples = {n: c[:800] for n, c in list(readmes.items())[:4]}
                prompt = f"Evaluate README quality:\n{json.dumps(samples, indent=2)}"
                prompt += "\n\nRate each 1-10 on clarity, setup, examples, completeness."
                sys_p = "You are a developer experience expert."

            elif task == "Naming Conventions":
                # sending repo names for the LLM to analyze naming patterns
                prompt = f"Analyze naming conventions:\nRepo names: {json.dumps([r['name'] for r in data['repos'][:25]])}"
                prompt += "\n\nAnalyze: conventions, consistency, descriptiveness, improvements."
                sys_p = "You are a code quality consultant."

            elif task == "Career Narrative":
                # sending the chronological timeline of repo creation
                # so the LLM can write a narrative of my technical journey
                timeline = [{"date": r.get("created_at","")[:7], "name": r["name"], "lang": r.get("language","")}
                           for r in sorted(data["repos"], key=lambda x: x.get("created_at",""))][:30]
                prompt = f"Write a career narrative from this timeline:\n{json.dumps(timeline, indent=2)}"
                prompt += "\n\nGive: journey phases, pivots, predicted next move, interview strengths"
                sys_p = "You are a tech career counselor."

            else:
                # next project ideas - sending current skills and recent work
                # so the LLM can suggest what to build next
                prompt = f"Suggest next projects based on:\nSkills: {[r.get('language','') for r in data['repos'][:15]]}"
                prompt += f"\nRecent: {[r['name'] for r in data['repos'][:8]]}"
                prompt += "\n\nSuggest: 3 skill-building, 2 stretch, 1 portfolio-booster"
                sys_p = "You are a tech mentor."

            # send the prompt to Ollama and get the response
            result = llm.analyze(prompt, sys_p)

        # display the result
        st.success(f"Completed in {result.latency_seconds}s | {result.completion_tokens} tokens")
        st.markdown(result.response)

        # expandable section showing detailed performance numbers
        # useful for the model comparison part of the assignment
        with st.expander("Performance Metrics"):
            st.json({
                "model": result.model,
                "latency_seconds": result.latency_seconds,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "tokens_per_sec": round(result.completion_tokens / max(result.latency_seconds, 0.1), 1),
                "cost_usd": result.cost_usd,
            })



# PAGE 4: ML ANALYSIS
# Traditional machine learning applied to my GitHub data.
# The assignment requires at least 2 ML techniques, so i implemented:
#   1. K-Means clustering to group similar repos together
#   2. Linear regression to forecast future commit activity

elif page == "ML Analysis":
    st.title("Machine Learning Analysis")

    if repos_df.empty:
        st.warning("No data available.")
        st.stop()

    #  ML Technique 1: K-Means Clustering 
    # this groups my repos based on similarity using both text features
    # (TF-IDF of repo name + description) and numeric features (stars, forks, size)
    # PCA reduces the high-dimensional feature space to 2D for visualization
    st.subheader("Repository Clustering (K-Means)")
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA

    if len(repos_df) >= 5:
        # interactive slider so the user can adjust the number of clusters
        k = st.slider("Number of clusters", 2, min(8, len(repos_df)-1), 4)

        # building the feature matrix:
        # text features from repo name + description using TF-IDF
        text = (repos_df["name"].fillna("") + " " + repos_df["description"].fillna("")).tolist()
        tfidf = TfidfVectorizer(max_features=20, stop_words="english")
        tf = tfidf.fit_transform(text).toarray()

        # numeric features scaled to zero mean and unit variance
        # so that stars (0-100s) don't dominate over forks (0-10s)
        nf = StandardScaler().fit_transform(repos_df[["stars","forks","size_kb","open_issues"]].fillna(0).values)

        # combining both feature types into one matrix
        combined = np.hstack([nf, tf])

        # running K-Means and projecting to 2D with PCA for the scatter plot
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(combined)
        coords = PCA(n_components=2).fit_transform(combined)

        # interactive scatter plot where each point is a repo, colored by cluster
        # hovering shows the repo name and language
        plot_df = pd.DataFrame({
            "x": coords[:,0], "y": coords[:,1],
            "cluster": labels.astype(str), "name": repos_df["name"],
            "language": repos_df["language"]
        })
        fig = px.scatter(plot_df, x="x", y="y", color="cluster",
                         hover_data=["name","language"],
                         title=f"Repository Clusters (k={k})", width=800, height=500)
        st.plotly_chart(fig, use_container_width=True)

        # listing which repos ended up in each cluster
        for c in range(k):
            members = repos_df.iloc[labels == c]["name"].tolist()
            st.write(f"**Cluster {c}** ({len(members)} repos): {', '.join(members[:8])}")

    #  ML Technique 2: Time Series Forecasting 
    # using linear regression on monthly commit counts to find the trend
    # and predict the next 6 months of activity
    st.subheader("Commit Activity Forecasting")
    if not commits_df.empty and len(commits_df) > 10:
        from sklearn.linear_model import LinearRegression

        # aggregate commits by month
        monthly = commits_df.set_index("date").resample("M").size().reset_index(name="commits")
        monthly["idx"] = range(len(monthly))

        # fit a simple linear regression: month_number -> commit_count
        X, y = monthly["idx"].values.reshape(-1,1), monthly["commits"].values
        lr = LinearRegression().fit(X, y)
        trend = lr.predict(X)

        # forecast the next 6 months using the same linear model
        # clamping to 0 so we don't predict negative commits
        fX = np.arange(len(monthly), len(monthly)+6).reshape(-1,1)
        forecast = np.maximum(lr.predict(fX), 0)
        future_dates = pd.date_range(monthly["date"].max() + pd.DateOffset(months=1), periods=6, freq="M")

        # combined chart: actual bars + trend line + forecast bars
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly["date"], y=monthly["commits"], name="Actual", opacity=0.5))
        fig.add_trace(go.Scatter(x=monthly["date"], y=trend, name="Trend", line=dict(dash="dash", color="red")))
        fig.add_trace(go.Bar(x=future_dates, y=forecast, name="Forecast", opacity=0.3, marker_color="red"))
        fig.update_layout(title="Commit Trend & Forecast", height=400)
        st.plotly_chart(fig, use_container_width=True)

        # showing the trend direction and R-squared value
        # R-squared tells us how well the linear model fits the data
        direction = "increasing" if lr.coef_[0] > 0 else "decreasing"
        st.write(f"**Trend:** {direction} by ~{abs(lr.coef_[0]):.1f} commits/month (R²={lr.score(X,y):.3f})")



# PAGE 5: MODEL COMPARISON
# The assignment requires comparing at least 2 LLM models.
# This page runs the exact same prompt through both llama3.1 and
# mistral side by side, then shows a metrics table and bar chart
# comparing latency, token count, and throughput.
# The user can also type their own custom prompt to test both models.

elif page == "Model Comparison":
    st.title("LLM Model Comparison")
    st.info(f"Comparing **{model_a}** vs **{model_b}**")

    # editable text area with a default prompt, the user can change it
    test_prompt = st.text_area("Test Prompt", value=(
        "Analyze this developer's GitHub profile and suggest 3 ways to improve their open source presence. "
        "Be specific and actionable."
    ))

    if st.button("Run Comparison", type="primary"):
        # creating two separate analyzer instances, one per model
        llm1 = LLMAnalyzer(provider="ollama", model=model_a, ollama_url=ollama_url)
        llm2 = LLMAnalyzer(provider="ollama", model=model_b, ollama_url=ollama_url)

        # running both models and showing results in two columns
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"{model_a}")
            with st.spinner(f"Running {model_a}..."):
                r1 = llm1.analyze(test_prompt, "You are a GitHub expert.")
            st.success(f"{r1.latency_seconds}s | {r1.completion_tokens} tokens")
            st.markdown(r1.response)

        with col2:
            st.subheader(f"{model_b}")
            with st.spinner(f"Running {model_b}..."):
                r2 = llm2.analyze(test_prompt, "You are a GitHub expert.")
            st.success(f"{r2.latency_seconds}s | {r2.completion_tokens} tokens")
            st.markdown(r2.response)

        #  Performance Metrics Table 
        # showing a direct comparison of both models' numbers
        st.subheader("Performance Metrics")
        metrics_df = pd.DataFrame([
            {"Metric": "Latency (s)", model_a: r1.latency_seconds, model_b: r2.latency_seconds},
            {"Metric": "Tokens Generated", model_a: r1.completion_tokens, model_b: r2.completion_tokens},
            {"Metric": "Tokens/sec",
             model_a: round(r1.completion_tokens/max(r1.latency_seconds,0.1),1),
             model_b: round(r2.completion_tokens/max(r2.latency_seconds,0.1),1)},
            {"Metric": "Cost (USD)", model_a: r1.cost_usd, model_b: r2.cost_usd},
        ])
        st.dataframe(metrics_df, use_container_width=True)

        # grouped bar chart for visual comparison of latency and throughput
        fig = go.Figure(data=[
            go.Bar(name=model_a, x=["Latency (s)","Tokens/sec"],
                   y=[r1.latency_seconds, r1.completion_tokens/max(r1.latency_seconds,0.1)]),
            go.Bar(name=model_b, x=["Latency (s)","Tokens/sec"],
                   y=[r2.latency_seconds, r2.completion_tokens/max(r2.latency_seconds,0.1)])
        ])
        fig.update_layout(barmode="group", title="Model Performance Comparison", height=350)
        st.plotly_chart(fig, use_container_width=True)