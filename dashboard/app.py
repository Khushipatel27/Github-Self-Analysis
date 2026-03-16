"""
GitHub Self-Analysis Dashboard
================================
Interactive dashboard for AI-Powered GitHub Self-Analysis.
5 pages: Overview · Activity · LLM Insights · ML Analysis · Model Comparison

Usage: streamlit run dashboard/app.py
"""

import streamlit as st
import json, sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_analysis import LLMAnalyzer, LLMResult

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GitHub Self-Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- metric cards ---------- */
.metric-card {
    background: linear-gradient(135deg, #1a1f35, #252b45);
    border-radius: 14px;
    padding: 1.4rem 1rem;
    text-align: center;
    border-top: 3px solid #6e40c9;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-size: 2.1rem;
    font-weight: 800;
    color: #a78bfa;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.78rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.35rem;
}
.metric-icon { font-size: 1.5rem; margin-bottom: 0.3rem; }

/* ---------- section headers ---------- */
.section-head {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #6e40c9;
    margin: 1.2rem 0 0.8rem;
}

/* ---------- info / insight boxes ---------- */
.insight-box {
    background: #1a1f35;
    border-left: 4px solid #6e40c9;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 0;
    color: #cbd5e1;
    font-size: 0.95rem;
}
.insight-box b { color: #a78bfa; }

/* ---------- task cards (LLM page) ---------- */
.task-pill {
    display: inline-block;
    background: #252b45;
    border: 1px solid #3d4568;
    border-radius: 20px;
    padding: 0.3rem 0.85rem;
    margin: 0.25rem;
    font-size: 0.82rem;
    color: #a78bfa;
}

/* ---------- result box ---------- */
.result-box {
    background: #161b2e;
    border: 1px solid #2d3562;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-top: 0.8rem;
    color: #e2e8f0;
    line-height: 1.7;
}

/* ---------- cluster chip ---------- */
.cluster-chip {
    display: inline-block;
    background: #252b45;
    border-radius: 6px;
    padding: 0.15rem 0.55rem;
    margin: 0.15rem;
    font-size: 0.8rem;
    color: #cbd5e1;
    border: 1px solid #3d4568;
}

/* ---------- landing hero ---------- */
.hero-wrap {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero-title {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(135deg, #6e40c9 0%, #a78bfa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 0.6rem;
}
.hero-sub {
    font-size: 1.1rem;
    color: #94a3b8;
    max-width: 580px;
    margin: 0 auto 1.5rem;
    line-height: 1.6;
}
.chip {
    display: inline-block;
    background: #1a1f35;
    border: 1px solid #3d4568;
    border-radius: 20px;
    padding: 0.3rem 0.85rem;
    margin: 0.2rem;
    font-size: 0.82rem;
    color: #a78bfa;
}

/* ---------- sidebar tweaks ---------- */
section[data-testid="stSidebar"] {
    background: #0f1320;
}
.sidebar-profile {
    text-align: center;
    padding: 0.5rem 0 0.8rem;
}
.sidebar-stat {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    color: #94a3b8;
    padding: 0.15rem 0;
}
.sidebar-stat span { color: #a78bfa; font-weight: 600; }

/* ---------- page title ---------- */
.page-title {
    font-size: 1.9rem;
    font-weight: 800;
    color: #e2e8f0;
    margin-bottom: 0.2rem;
}
.page-sub {
    font-size: 0.95rem;
    color: #64748b;
    margin-bottom: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ───────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cbd5e1", family="Inter, sans-serif"),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor="#1e2535", zerolinecolor="#1e2535"),
    yaxis=dict(gridcolor="#1e2535", zerolinecolor="#1e2535"),
)
COLORS = px.colors.qualitative.Vivid


# ── Helpers ────────────────────────────────────────────────────────────────────
def metric_card(icon, label, value, color="#a78bfa"):
    st.markdown(f"""
    <div class="metric-card" style="border-top-color:{color}">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="section-head">{title}</div>', unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)


def apply_theme(fig, height=400):
    fig.update_layout(**PLOT_LAYOUT, height=height)
    return fig


# ── Username gate ──────────────────────────────────────────────────────────────
if "username" not in st.session_state:
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-title">GitHub Self-Analysis</div>
        <div class="hero-sub">
            Drop any public GitHub username and get an AI-powered breakdown
            of their coding profile — skill clusters, commit patterns, career
            narrative, and live LLM insights.
        </div>
        <div>
            <span class="chip">🤖 Local LLMs (Ollama)</span>
            <span class="chip">📊 9 Visualizations</span>
            <span class="chip">🧠 7 AI Analysis Tasks</span>
            <span class="chip">⚡ K-Means Clustering</span>
            <span class="chip">📈 6-Month Forecast</span>
            <span class="chip">💰 $0 Cloud Cost</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown("#### Enter a GitHub Username")
        username_input = st.text_input(
            "GitHub Username", placeholder="e.g. torvalds, gvanrossum, Khushipatel27",
            label_visibility="collapsed"
        )
        token_input = st.text_input(
            "GitHub Token *(optional — raises rate limit from 60 → 5,000 req/hr)*",
            type="password",
            help="Generate at github.com/settings/tokens — no special scopes needed for public repos.",
            placeholder="ghp_xxxxxxxxxxxx  (optional)",
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 Analyze Profile", type="primary", width='stretch'):
            if username_input.strip():
                st.session_state.username = username_input.strip()
                st.session_state.token = token_input.strip() or None
                st.rerun()
            else:
                st.error("Please enter a GitHub username.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center; color:#475569; font-size:0.82rem">
            ⚠️ Requires <a href="https://ollama.ai" style="color:#a78bfa">Ollama</a> running locally for LLM pages
            &nbsp;·&nbsp; All analysis runs on your machine
        </div>
        """, unsafe_allow_html=True)
    st.stop()


username = st.session_state.username
DATA_DIR = Path(__file__).parent.parent / "data" / username

# ── Data collection ────────────────────────────────────────────────────────────
if not (DATA_DIR / "profile.json").exists():
    import collect_data as cd
    cd.Dataset_Path = DATA_DIR
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    headers = {"Accept": "application/vnd.github.v3+json"}
    # priority: 1) token from landing page input, 2) GITHUB_TOKEN secret, 3) unauthenticated
    _gh_token = (st.session_state.get("token")
                 or st.secrets.get("GITHUB_TOKEN", "")
                 if hasattr(st, "secrets") else st.session_state.get("token"))
    if _gh_token:
        headers["Authorization"] = f"token {_gh_token}"

    # warn if the token looks like it was never replaced
    if _gh_token and _gh_token in ("paste_your_github_token_here", "ghp_your_token_here", ""):
        st.warning("⚠️ GitHub token in secrets.toml looks like a placeholder. "
                   "Collection will run without a token (60 req/hr limit).")
        _gh_token = None
        headers.pop("Authorization", None)

    profile_check = None
    with st.status(f"Collecting GitHub data for @{username}…", expanded=True) as status:
        st.write("📡 Fetching profile…")
        profile_check = cd.collecting_profile_user(username, headers)
        if not profile_check:
            del st.session_state["username"]
            if "token" in st.session_state:
                del st.session_state["token"]
            status.update(label="User not found.", state="error")
        else:
            st.write("📦 Fetching repositories…")
            repos_raw = cd.collecting_repos(username, headers)
            st.write("💬 Fetching commits…")
            cd.collecting_commits(username, repos_raw, headers)
            st.write("🔤 Fetching language breakdown…")
            cd.collecting_languages(repos_raw, headers)
            st.write("📋 Fetching events & READMEs…")
            cd.collecting_events(username, headers)
            cd.collect_repo_contents(repos_raw, headers)
            status.update(label="Data collected!", state="complete")

    if not profile_check:
        st.error(f"**@{username}** not found on GitHub. Check the spelling and try again.")
        if st.button("← Search again", type="primary"):
            st.rerun()
        st.stop()

    st.rerun()


# ── Load & parse data ──────────────────────────────────────────────────────────
@st.cache_data
def load_data(data_dir: str):
    data = {}
    for fname in ["profile","repos","commits","languages","events","readmes","file_listings"]:
        fpath = Path(data_dir) / f"{fname}.json"
        if fpath.exists():
            with open(fpath) as f:
                data[fname] = json.load(f)
        else:
            data[fname] = [] if fname in ["repos","commits","events"] else {}
    return data


def parse_repos(raw):
    if not raw:
        return pd.DataFrame(columns=["name","description","language","stars","forks",
                                     "size_kb","created_at","updated_at","pushed_at",
                                     "open_issues","is_fork","topics","url"])
    return pd.DataFrame([{
        "name": r["name"],
        "description": r.get("description", ""),
        "language": r.get("language", "Unknown"),
        "stars": r.get("stargazers_count", 0),
        "forks": r.get("forks_count", 0),
        "size_kb": r.get("size", 0),
        "created_at": pd.to_datetime(r["created_at"]),
        "updated_at": pd.to_datetime(r["updated_at"]),
        "pushed_at": pd.to_datetime(r.get("pushed_at")),
        "open_issues": r.get("open_issues_count", 0),
        "is_fork": r.get("fork", False),
        "topics": r.get("topics", []),
        "url": r.get("html_url", ""),
    } for r in raw])


def parse_commits(raw):
    if not raw:
        return pd.DataFrame(columns=["sha","repo","message","date"])
    rows = []
    for c in raw:
        ci = c.get("commit", {})
        ai = ci.get("author", {})
        rows.append({
            "sha": c.get("sha", "")[:8],
            "repo": c.get("_repo", ""),
            "message": ci.get("message", ""),
            "date": pd.to_datetime(ai.get("date")),
        })
    return pd.DataFrame(rows)


data        = load_data(str(DATA_DIR))
repos_df    = parse_repos(data["repos"])
commits_df  = parse_commits(data["commits"])
profile     = data.get("profile", {})
languages_data = data.get("languages", {})
readmes     = data.get("readmes", {})


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 GitHub Analysis")

    if profile:
        avatar = profile.get("avatar_url", "")
        name   = profile.get("name") or profile.get("login", username)
        bio    = profile.get("bio", "")
        joined = profile.get("created_at", "")[:4]

        st.markdown(f"""
        <div class="sidebar-profile">
            <img src="{avatar}" width="90"
                 style="border-radius:50%; border:3px solid #6e40c9; margin-bottom:0.5rem">
            <div style="font-weight:700; font-size:1rem; color:#e2e8f0">{name}</div>
            <div style="font-size:0.8rem; color:#6e40c9">@{username}</div>
            {"<div style='font-size:0.8rem;color:#94a3b8;margin-top:0.3rem'>" + bio + "</div>" if bio else ""}
        </div>
        <div class="sidebar-stat"><div>Member since</div><span>{joined}</span></div>
        <div class="sidebar-stat"><div>Public repos</div><span>{profile.get("public_repos",0)}</span></div>
        <div class="sidebar-stat"><div>Followers</div><span>{profile.get("followers",0)}</span></div>
        <div class="sidebar-stat"><div>Following</div><span>{profile.get("following",0)}</span></div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("#### ⚙️ LLM Settings")

    # auto-detect Groq key from Streamlit secrets (set in the cloud dashboard)
    # falls back to empty string so the user can paste it manually
    _groq_secret = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""

    provider = st.radio("Provider", ["Ollama (local)", "Groq (cloud — free)"],
                        help="Groq works on Streamlit Cloud. Ollama requires a local install.")

    if provider == "Ollama (local)":
        llm_provider  = "ollama"
        ollama_url    = st.text_input("Ollama URL", "http://localhost:11434")
        groq_api_key  = ""
        model_a = st.selectbox("Model A", ["llama3.1", "mistral"], index=0)
        model_b = st.selectbox("Model B", ["mistral", "llama3.1"], index=0)
    else:
        llm_provider = "groq"
        ollama_url   = ""
        if _groq_secret:
            # key is already loaded from secrets — never render it in the UI
            groq_api_key = _groq_secret
            st.caption("✅ Groq API key loaded from secrets")
        else:
            groq_api_key = st.text_input(
                "Groq API Key", type="password", placeholder="gsk_...",
                help="Free key at console.groq.com — no credit card needed"
            )
        GROQ_MODELS = ["llama-3.1-8b-instant", "mixtral-8x7b-32768",
                       "llama-3.3-70b-versatile", "gemma2-9b-it"]
        model_a = st.selectbox("Model A", GROQ_MODELS, index=0)
        model_b = st.selectbox("Model B", GROQ_MODELS, index=1)

    st.markdown("---")

    page = st.radio("Navigate", [
        "🏠 Overview", "📅 Activity", "🤖 LLM Insights",
        "🧠 ML Analysis", "⚖️ Model Comparison"
    ])

    st.markdown("---")
    if st.button("🔄 Refresh Data", width='stretch',
                 help="Re-collect from GitHub (clears cached data for this user)"):
        import shutil
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR)
        load_data.clear()
        st.rerun()

    if st.button("← Different User", width='stretch'):
        del st.session_state["username"]
        if "token" in st.session_state:
            del st.session_state["token"]
        load_data.clear()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 · OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown(f'<div class="page-title">@{username}</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">GitHub profile at a glance</div>', unsafe_allow_html=True)

    if repos_df.empty:
        st.error(f"No public repositories found for @{username}.")
        st.stop()

    # ── metric cards ──
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("📁", "Repositories",  len(repos_df),             "#a78bfa")
    with c2: metric_card("💬", "Commits",        len(commits_df),           "#34d399")
    with c3: metric_card("⭐", "Total Stars",    int(repos_df["stars"].sum()), "#fbbf24")
    with c4: metric_card("🍴", "Total Forks",    int(repos_df["forks"].sum()), "#60a5fa")
    with c5: metric_card("🔤", "Languages",      repos_df["language"].nunique(), "#f472b6")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── language charts ──
    section("Language Breakdown")
    left, right = st.columns(2)

    with left:
        lc  = repos_df["language"].value_counts().head(10)
        fig = px.bar(x=lc.values, y=lc.index, orientation="h",
                     color=lc.index, color_discrete_sequence=COLORS)
        fig.update_layout(**PLOT_LAYOUT, height=350,
                          showlegend=False,
                          xaxis_title="Repositories", yaxis_title="")
        fig.update_traces(marker_line_width=0)
        st.markdown("**By Repo Count**")
        st.plotly_chart(fig, width='stretch')

    with right:
        tb = defaultdict(int)
        for rl in languages_data.values():
            for lang, b in rl.items():
                tb[lang] += b
        lb = pd.Series(tb).sort_values(ascending=False).head(8)
        if not lb.empty:
            fig = px.pie(names=lb.index, values=lb.values, hole=0.5,
                         color_discrete_sequence=COLORS)
            fig.update_traces(textposition="inside", textinfo="percent+label",
                              marker=dict(line=dict(color="#0f1320", width=2)))
            fig.update_layout(**PLOT_LAYOUT, height=350, showlegend=True,
                              legend=dict(orientation="v", x=1.05))
            st.markdown("**By Code Volume (bytes)**")
            st.plotly_chart(fig, width='stretch')

    # ── repo table ──
    section("All Repositories")
    display = repos_df[["name","language","stars","forks","size_kb","created_at","description"]].copy()
    display["created_at"] = display["created_at"].dt.strftime("%Y-%m-%d")
    display = display.sort_values("stars", ascending=False).reset_index(drop=True)
    st.dataframe(
        display,
        width='stretch',
        column_config={
            "name":        st.column_config.TextColumn("Repository"),
            "language":    st.column_config.TextColumn("Language"),
            "stars":       st.column_config.NumberColumn("⭐ Stars"),
            "forks":       st.column_config.NumberColumn("🍴 Forks"),
            "size_kb":     st.column_config.NumberColumn("Size (KB)"),
            "created_at":  st.column_config.TextColumn("Created"),
            "description": st.column_config.TextColumn("Description"),
        },
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 · ACTIVITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📅 Activity":
    st.markdown('<div class="page-title">Activity Timeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Coding patterns, schedule, and commit behaviour</div>',
                unsafe_allow_html=True)

    if commits_df.empty:
        st.warning(
            f"No commits found for **@{username}**. "
            "This can happen if the account uses a different email than their GitHub login, "
            "or if all activity is in private/organization repos."
        )
        st.stop()

    # ── monthly timeline ──
    section("Monthly Commit Activity")
    commits_df["month"] = commits_df["date"].dt.to_period("M")
    monthly = commits_df.groupby("month").size().reset_index(name="commits")
    monthly["month"] = monthly["month"].dt.to_timestamp()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["month"], y=monthly["commits"],
        mode="lines+markers", fill="tozeroy",
        name="Commits",
        line=dict(color="#6e40c9", width=2.5),
        fillcolor="rgba(110,64,201,0.15)",
        marker=dict(size=7, color="#a78bfa"),
    ))
    fig.update_layout(**PLOT_LAYOUT, height=320, yaxis_title="Commits", xaxis_title="")
    st.plotly_chart(fig, width='stretch')

    peak_month = monthly.loc[monthly["commits"].idxmax(), "month"].strftime("%B %Y") if not monthly.empty else "N/A"
    insight(f"<b>Peak month:</b> {peak_month} — {int(monthly['commits'].max())} commits")

    # ── heatmap + growth side by side ──
    col_left, col_right = st.columns(2)

    with col_left:
        section("Coding Schedule")
        commits_df["dow"]  = commits_df["date"].dt.day_name()
        commits_df["hour"] = commits_df["date"].dt.hour
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        hm = commits_df.groupby(["dow","hour"]).size().unstack(fill_value=0).reindex(day_order)
        fig = px.imshow(hm, color_continuous_scale="Purples", aspect="auto",
                        labels=dict(x="Hour of Day", y="", color="Commits"))
        fig.update_layout(**PLOT_LAYOUT, height=300,
                          coloraxis_colorbar=dict(thickness=12, len=0.8))
        st.plotly_chart(fig, width='stretch')

        busiest_day  = commits_df["dow"].value_counts().idxmax()  if not commits_df.empty else "N/A"
        busiest_hour = int(commits_df["hour"].value_counts().idxmax()) if not commits_df.empty else 0
        insight(f"<b>Most active:</b> {busiest_day}s around {busiest_hour:02d}:00")

    with col_right:
        section("Repository Growth")
        rds = repos_df.sort_values("created_at").copy()
        rds["cumulative"] = range(1, len(rds)+1)
        fig = px.area(rds, x="created_at", y="cumulative",
                      markers=True, color_discrete_sequence=["#34d399"])
        fig.update_traces(fillcolor="rgba(52,211,153,0.12)", line=dict(width=2.5))
        fig.update_layout(**PLOT_LAYOUT, height=300,
                          xaxis_title="", yaxis_title="Total Repos")
        st.plotly_chart(fig, width='stretch')

    # ── commit message analytics ──
    section("Commit Message Analytics")
    msg_lens = commits_df["message"].str.split("\n").str[0].str.len()
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(msg_lens, nbins=25, color_discrete_sequence=["#6e40c9"])
        fig.update_layout(**PLOT_LAYOUT, height=280,
                          xaxis_title="Characters (first line)", yaxis_title="Count",
                          showlegend=False, title="Message Length Distribution")
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, width='stretch')

    with col2:
        stop = {"the","a","an","and","or","in","on","at","to","for","of","is","it",
                "this","that","with","from","by","as","update","add","fix","remove"}
        words = []
        for msg in commits_df["message"]:
            fl = msg.split("\n")[0].lower()
            words.extend([w.strip(".,!?()[]{}:") for w in fl.split()
                          if len(w) > 2 and w not in stop])
        wf = Counter(words).most_common(12)
        if wf:
            wdf = pd.DataFrame(wf, columns=["word","count"])
            fig = px.bar(wdf, x="count", y="word", orientation="h",
                         color="count", color_continuous_scale="Purples",
                         title="Top Commit Keywords")
            fig.update_layout(**PLOT_LAYOUT, height=280,
                              yaxis_title="", coloraxis_showscale=False)
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, width='stretch')

    avg_len = int(msg_lens.mean()) if not msg_lens.empty else 0
    insight(f"<b>Avg commit message length:</b> {avg_len} characters &nbsp;·&nbsp; "
            f"<b>Total commits analysed:</b> {len(commits_df)}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 · LLM INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 LLM Insights":
    st.markdown('<div class="page-title">LLM-Powered Insights</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">Analysing @{username} with <b>{model_a}</b> via {llm_provider.capitalize()}</div>',
                unsafe_allow_html=True)

    TASKS = {
        "🧬 Sentiment Analysis":      ("Developer psychology expert",   "4-class sentiment breakdown of commits"),
        "🗂️ Topic Clustering":         ("Technical recruiter",           "Thematic groups + portfolio gaps"),
        "🛠️ Skill Extraction":         ("Senior tech lead",              "Proficiency tiers by domain"),
        "📝 Documentation Quality":    ("Developer experience expert",   "1–10 ratings across 4 dimensions"),
        "🏷️ Naming Conventions":       ("Code quality consultant",       "Issues + recommendations"),
        "🗺️ Career Narrative":         ("Tech career counselor",         "Timeline, pivots, interview strengths"),
        "💡 Next Project Ideas":       ("Tech mentor",                   "3 skill-builders · 2 stretch · 1 booster"),
    }

    # task cards legend
    st.markdown('<div class="section-head">Choose an Analysis Task</div>', unsafe_allow_html=True)
    pills_html = "".join(
        f'<span class="task-pill">{t} — <span style="color:#64748b">{d}</span></span>'
        for t, (_, d) in TASKS.items()
    )
    st.markdown(f'<div style="margin-bottom:0.8rem">{pills_html}</div>', unsafe_allow_html=True)

    if commits_df.empty:
        st.info(
            "⚠️ No commit data found for this user — **Sentiment Analysis** is unavailable. "
            "All other 6 tasks work fine. "
            "Hit **🔄 Refresh Data** in the sidebar if you think commits exist."
        )

    task = st.selectbox("Select task", list(TASKS.keys()), label_visibility="collapsed")
    persona, task_desc = TASKS[task]
    st.caption(f"**Persona:** {persona}  ·  **Output:** {task_desc}")

    llm = LLMAnalyzer(provider=llm_provider, model=model_a,
                      ollama_url=ollama_url, api_key=groq_api_key or None)

    if st.button(f"▶ Run  {task}", type="primary"):
        # tasks that need commits — show a clear error instead of crashing
        needs_commits = "Sentiment" in task
        if needs_commits and commits_df.empty:
            st.warning(
                f"⚠️ No commits found for **@{username}**. "
                "Sentiment Analysis requires commit history. "
                "Try Topic Clustering, Skill Extraction, or Career Narrative instead."
            )
            st.stop()

        with st.spinner(f"Sending to {model_a}…"):

            if "Sentiment" in task:
                sample = commits_df["message"].str.split("\n").str[0].head(40).tolist()
                prompt = "Analyze sentiment of these commit messages:\n" + \
                         "\n".join(f"{i+1}. {m}" for i,m in enumerate(sample))
                prompt += "\n\nGive: 1) sentiment breakdown %, 2) patterns, 3) developer personality"

            elif "Topic" in task:
                rs = [f"- {r['name']}: {r.get('description','')} ({r.get('language','')})"
                      for r in data["repos"][:25]]
                prompt = "Group these repos into thematic clusters:\n" + "\n".join(rs)
                prompt += "\n\nGive: clusters with repo lists, expertise areas, gaps"

            elif "Skill" in task:
                prompt = f"Extract skills from this GitHub data:\nLanguages: {json.dumps(dict(list(languages_data.items())[:15]))}"
                prompt += "\nRepos: " + ", ".join(r["name"] for r in data["repos"][:20])
                prompt += "\n\nGive: skills with proficiency levels, domain knowledge, next skills to learn"

            elif "Documentation" in task:
                samples = {n: c[:800] for n, c in list(readmes.items())[:4]}
                prompt = f"Evaluate README quality:\n{json.dumps(samples, indent=2)}"
                prompt += "\n\nRate each 1-10 on clarity, setup, examples, completeness."

            elif "Naming" in task:
                prompt = f"Analyze naming conventions:\nRepo names: {json.dumps([r['name'] for r in data['repos'][:25]])}"
                prompt += "\n\nAnalyze: conventions, consistency, descriptiveness, improvements."

            elif "Career" in task:
                timeline = [{"date": r.get("created_at","")[:7], "name": r["name"], "lang": r.get("language","")}
                            for r in sorted(data["repos"], key=lambda x: x.get("created_at",""))][:30]
                prompt = f"Write a career narrative from this timeline:\n{json.dumps(timeline, indent=2)}"
                prompt += "\n\nGive: journey phases, pivots, predicted next move, interview strengths"

            else:
                prompt = f"Suggest next projects based on:\nSkills: {[r.get('language','') for r in data['repos'][:15]]}"
                prompt += f"\nRecent: {[r['name'] for r in data['repos'][:8]]}"
                prompt += "\n\nSuggest: 3 skill-building, 2 stretch, 1 portfolio-booster"

            result = llm.analyze(prompt, f"You are a {persona.lower()}.")

        # ── result display ──
        m1, m2, m3 = st.columns(3)
        m1.metric("⏱ Latency", f"{result.latency_seconds:.1f}s")
        m2.metric("🔢 Tokens", result.completion_tokens)
        m3.metric("⚡ Throughput",
                  f"{result.completion_tokens/max(result.latency_seconds,0.1):.0f} tok/s")

        st.markdown(f'<div class="result-box">{result.response.replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True)

        with st.expander("Raw performance metrics"):
            st.json({
                "model": result.model,
                "latency_seconds": result.latency_seconds,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "tokens_per_sec": round(result.completion_tokens / max(result.latency_seconds, 0.1), 1),
                "cost_usd": result.cost_usd,
            })


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 · ML ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 ML Analysis":
    st.markdown('<div class="page-title">Machine Learning Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">K-Means clustering · TF-IDF · PCA · Linear regression forecasting</div>',
                unsafe_allow_html=True)

    if repos_df.empty:
        st.warning("No repository data available.")
        st.stop()

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression

    # ── K-Means ──
    section("Repository Clustering (K-Means + TF-IDF + PCA)")
    insight("Repos are represented as vectors combining <b>TF-IDF text features</b> "
            "(name + description) and <b>scaled numeric features</b> (stars, forks, size). "
            "PCA reduces to 2D for visualization.")

    if len(repos_df) >= 4:
        k = st.slider("Number of clusters (k)", 2, min(8, len(repos_df)-1), min(4, len(repos_df)-1))

        text     = (repos_df["name"].fillna("") + " " + repos_df["description"].fillna("")).tolist()
        tfidf    = TfidfVectorizer(max_features=20, stop_words="english")
        tf       = tfidf.fit_transform(text).toarray()
        nf       = StandardScaler().fit_transform(
                       repos_df[["stars","forks","size_kb","open_issues"]].fillna(0).values)
        combined = np.hstack([nf, tf])

        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(combined)
        coords = PCA(n_components=2).fit_transform(combined)

        plot_df = pd.DataFrame({
            "PC1": coords[:,0], "PC2": coords[:,1],
            "Cluster": [f"Cluster {l}" for l in labels],
            "Repo": repos_df["name"].values,
            "Language": repos_df["language"].values,
        })
        fig = px.scatter(
            plot_df, x="PC1", y="PC2", color="Cluster",
            hover_data={"Repo": True, "Language": True, "PC1": False, "PC2": False},
            color_discrete_sequence=COLORS,
            title=f"Repository Map — {k} Clusters (PCA projection)",
        )
        fig.update_traces(marker=dict(size=14, line=dict(width=1.5, color="#0f1320")))
        fig.update_layout(**PLOT_LAYOUT, height=460,
                          xaxis_title="Principal Component 1",
                          yaxis_title="Principal Component 2")
        st.plotly_chart(fig, width='stretch')

        cols = st.columns(k)
        for c in range(k):
            members = repos_df.iloc[labels == c]["name"].tolist()
            with cols[c]:
                st.markdown(f"**Cluster {c}** &nbsp; `{len(members)} repos`")
                for m in members:
                    st.markdown(f'<span class="cluster-chip">{m}</span>', unsafe_allow_html=True)
    else:
        st.info("Need at least 4 repositories for clustering.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Forecasting ──
    section("Commit Activity Forecasting (Linear Regression)")
    insight("A linear regression is fitted on monthly commit counts to extract the <b>trend</b> "
            "and project <b>6 months forward</b>. R² measures how well the linear model fits.")

    if not commits_df.empty and len(commits_df) >= 5:
        monthly = commits_df.set_index("date").resample("ME").size().reset_index(name="commits")
        monthly["idx"] = range(len(monthly))

        X, y = monthly["idx"].values.reshape(-1,1), monthly["commits"].values
        lr    = LinearRegression().fit(X, y)
        trend = lr.predict(X)

        fX           = np.arange(len(monthly), len(monthly)+6).reshape(-1,1)
        forecast     = np.maximum(lr.predict(fX), 0)
        future_dates = pd.date_range(
            monthly["date"].max() + pd.DateOffset(months=1), periods=6, freq="ME")

        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly["date"], y=monthly["commits"],
                             name="Actual", marker_color="#6e40c9", opacity=0.7))
        fig.add_trace(go.Scatter(x=monthly["date"], y=trend,
                                 name="Trend", line=dict(dash="dash", color="#f472b6", width=2.5)))
        fig.add_trace(go.Bar(x=future_dates, y=forecast,
                             name="Forecast", marker_color="#34d399", opacity=0.5))
        fig.update_layout(**PLOT_LAYOUT, height=380,
                          barmode="overlay", title="Commit Trend & 6-Month Forecast",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(fig, width='stretch')

        direction = "📈 increasing" if lr.coef_[0] > 0 else "📉 decreasing"
        r2        = lr.score(X, y)
        m1, m2, m3 = st.columns(3)
        m1.metric("Trend Direction", direction.split()[1].capitalize())
        m2.metric("Rate",            f"{lr.coef_[0]:+.2f} commits/month")
        m3.metric("R²",             f"{r2:.3f}")

        if r2 < 0.2:
            insight("<b>Low R²</b> indicates commit frequency is irregular — "
                    "a linear model captures the general direction but not the variance.")
    else:
        st.info("Need at least 5 commits for forecasting.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 · MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Model Comparison":
    st.markdown('<div class="page-title">LLM Model Comparison</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">Side-by-side benchmark — <b>{model_a}</b> vs <b>{model_b}</b></div>',
                unsafe_allow_html=True)

    test_prompt = st.text_area(
        "Custom test prompt (or use the default)",
        value=(
            f"Analyze @{username}'s GitHub profile and give 3 specific, actionable recommendations "
            "to strengthen their open-source presence for job applications."
        ),
        height=100,
    )

    if st.button("⚖️ Run Comparison", type="primary"):
        llm1 = LLMAnalyzer(provider=llm_provider, model=model_a,
                           ollama_url=ollama_url, api_key=groq_api_key or None)
        llm2 = LLMAnalyzer(provider=llm_provider, model=model_b,
                           ollama_url=ollama_url, api_key=groq_api_key or None)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### 🤖 {model_a}")
            with st.spinner(f"Running {model_a}…"):
                r1 = llm1.analyze(test_prompt, "You are a GitHub expert.")
            st.markdown(f'<div class="result-box">{r1.response.replace(chr(10),"<br>")}</div>',
                        unsafe_allow_html=True)

        with col2:
            st.markdown(f"### 🤖 {model_b}")
            with st.spinner(f"Running {model_b}…"):
                r2 = llm2.analyze(test_prompt, "You are a GitHub expert.")
            st.markdown(f'<div class="result-box">{r2.response.replace(chr(10),"<br>")}</div>',
                        unsafe_allow_html=True)

        # ── metrics ──
        section("Performance Comparison")

        def tps(r): return round(r.completion_tokens / max(r.latency_seconds, 0.1), 1)

        faster  = model_a if r1.latency_seconds <= r2.latency_seconds else model_b
        denser  = model_a if r1.completion_tokens >= r2.completion_tokens else model_b

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{model_a} Latency",    f"{r1.latency_seconds:.1f}s",
                  delta=f"{r1.latency_seconds - r2.latency_seconds:+.1f}s vs {model_b}",
                  delta_color="inverse")
        m2.metric(f"{model_b} Latency",    f"{r2.latency_seconds:.1f}s")
        m3.metric(f"{model_a} Throughput", f"{tps(r1)} tok/s")
        m4.metric(f"{model_b} Throughput", f"{tps(r2)} tok/s")

        insight(f"<b>Faster model:</b> {faster} &nbsp;·&nbsp; "
                f"<b>More tokens generated:</b> {denser}")

        metrics_df = pd.DataFrame([
            {"Metric": "Latency (s)",      model_a: r1.latency_seconds,   model_b: r2.latency_seconds},
            {"Metric": "Tokens Generated", model_a: r1.completion_tokens,  model_b: r2.completion_tokens},
            {"Metric": "Tokens / sec",     model_a: tps(r1),               model_b: tps(r2)},
            {"Metric": "Prompt Tokens",    model_a: r1.prompt_tokens,      model_b: r2.prompt_tokens},
            {"Metric": "Cost (USD)",       model_a: r1.cost_usd,           model_b: r2.cost_usd},
        ])
        st.dataframe(metrics_df, width='stretch', hide_index=True)

        fig = go.Figure(data=[
            go.Bar(name=model_a,
                   x=["Latency (s)", "Tokens/sec", "Tokens Generated"],
                   y=[r1.latency_seconds, tps(r1), r1.completion_tokens],
                   marker_color="#6e40c9"),
            go.Bar(name=model_b,
                   x=["Latency (s)", "Tokens/sec", "Tokens Generated"],
                   y=[r2.latency_seconds, tps(r2), r2.completion_tokens],
                   marker_color="#34d399"),
        ])
        fig.update_layout(**PLOT_LAYOUT, barmode="group",
                          height=340, title="Model Performance Benchmark",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(fig, width='stretch')
