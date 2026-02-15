"""
GitHub Data Collector

This is the first script to run in the project. It connects to the GitHub REST API
and downloads all my public profile data into JSON files in the data/ folder.

I chose the GitHub REST API because it doesn't require any special authentication
for public data, and the endpoints are well documented. A personal access token
is optional but recommended because without one, GitHub limits you to 60 requests
per hour, which can be a problem if you have many repos.

The script collects 6 types of data in sequence:
    1. Profile 
    2. Repositories 
    3. Commits 
    4. Languages 
    5. Events 
    6. READMEs + File listings 
Each type is saved as a separate JSON file so the notebook and dashboard
can load only what they need.

Usage:
    python collect_data.py --username YOUR_GITHUB_USERNAME [--token YOUR_TOKEN]

Generate a token at: https://github.com/settings/tokens (no special scopes needed for public data)
"""

import requests
import json
import os
import time
import argparse
from datetime import datetime
from pathlib import Path

# creating the data directory if it doesn't exist yet
# all JSON files will be saved here
Dataset_Path = Path("data")
Dataset_Path.mkdir(exist_ok=True)

# base URL for all GitHub API calls
BASE_URL = "https://api.github.com"


def request_making(url, headers, params=None):
    """
    Wrapper around requests.get that handles GitHub's rate limiting.

    GitHub returns a 403 status code when you've exceeded the rate limit.
    When that happens, the response headers include X-RateLimit-Reset which
    tells us the unix timestamp when the limit resets. This function
    automatically waits until that time and retries the request.

    This way the rest of the code doesn't need to worry about rate limits
    at all - it just calls request_making and gets back a response.
    """
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 403 and "rate limit" in response.text.lower():
        reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
        wait = max(reset_time - int(time.time()), 1)
        print(f"  Rate limited. Waiting {wait}s...")
        time.sleep(wait)
        response = requests.get(url, headers=headers, params=params)
    return response


def collecting_profile_user(username, headers):
    """
    Step 1: Fetch basic profile information.

    This calls the /users/{username} endpoint which returns account-level data
    like display name, bio, location, follower count, and when the account was
    created. This data is used in the dashboard sidebar and in the career
    progression analysis.

    Saves to: data/profile.json
    """
    print(f"[1/6] Collecting profile for {username}...")
    resp = request_making(f"{BASE_URL}/users/{username}", headers)
    if resp.status_code == 200:
        profile = resp.json()
        with open(Dataset_Path / "profile.json", "w") as f:
            json.dump(profile, f, indent=2)
        print(f"  -> Saved profile (created: {profile.get('created_at', 'N/A')})")
        return profile
    else:
        print(f"  ERROR: {resp.status_code} - {resp.text}")
        return None


def collecting_repos(username, headers):
    """
    Step 2: Fetch all public repositories.

    Uses pagination (100 repos per page) to collect every public repo.
    The repos are sorted by last updated so the most active ones come first.
    Each repo object includes metadata like stars, forks, primary language,
    creation date, description, and topics.

    This data feeds into almost every analysis - language distribution,
    topic clustering, skill extraction, K-Means clustering, and the
    repo growth timeline.

    Saves to: data/repos.json
    """
    print(f"[2/6] Collecting repositories...")
    repos = []
    page = 1
    while True:
        resp = request_making(
            f"{BASE_URL}/users/{username}/repos",
            headers,
            params={"per_page": 100, "page": page, "sort": "updated"}
        )
        if resp.status_code != 200:
            break
        batch = resp.json()
        if not batch:
            break
        repos.extend(batch)
        page += 1

    with open(Dataset_Path / "repos.json", "w") as f:
        json.dump(repos, f, indent=2)
    print(f"  -> Saved {len(repos)} repositories")
    return repos


def collecting_commits(username, repos, headers, max_repos=30):
    """
    Step 3: Fetch commit history from the most active repositories.

    Iterates through the top 30 most recently pushed repos and pulls up to
    100 commits per repo, filtered by the username so we only get commits
    authored by the user (not collaborators or merge commits from others).

    Each commit is tagged with a _repo field so we know which repo it came from.
    This is important because the GitHub API returns commits per-repo, but
    our analysis needs all commits in one flat list.

    This data is used for sentiment analysis, commit timeline, coding schedule
    heatmap, message word frequency, and time series forecasting.

    Saves to: data/commits.json
    """
    print(f"[3/6] Collecting commits (up to {max_repos} repos)...")
    all_commits = []
    # sorting repos by pushed_at date so we start with the most recently active
    sorted_repos = sorted(repos, key=lambda r: r.get("pushed_at") or "", reverse=True)

    for repo in sorted_repos[:max_repos]:
        repo_name = repo["full_name"]
        resp = request_making(
            f"{BASE_URL}/repos/{repo_name}/commits",
            headers,
            params={"author": username, "per_page": 100}
        )
        if resp.status_code == 200:
            commits = resp.json()
            # tagging each commit with the repo it came from
            for c in commits:
                c["_repo"] = repo_name
            all_commits.extend(commits)
            if commits:
                print(f"  -> {repo['name']}: {len(commits)} commits")

    with open(Dataset_Path / "commits.json", "w") as f:
        json.dump(all_commits, f, indent=2)
    print(f"  -> Total: {len(all_commits)} commits saved")
    return all_commits


def collecting_languages(repos, headers):
    """
    Step 4: Fetch language breakdown for each repository.

    The /repos/{owner}/{repo}/languages endpoint returns a dictionary mapping
    language names to the number of bytes of code in that language.
    For example: {"Python": 45000, "HTML": 12000, "CSS": 3000}

    This gives a more accurate picture than just the primary language field
    on a repo, because many repos use multiple languages. For instance a
    Python project might have significant amounts of HTML and JavaScript too.

    This data is used for the "Languages by Code Volume" chart in the
    dashboard and for the skill extraction LLM task.

    Saves to: data/languages.json
    """
    print(f"[4/6] Collecting language data...")
    lang_data = {}
    for repo in repos:
        repo_name = repo["full_name"]
        resp = request_making(f"{BASE_URL}/repos/{repo_name}/languages", headers)
        if resp.status_code == 200:
            langs = resp.json()
            if langs:
                lang_data[repo["name"]] = langs

    with open(Dataset_Path / "languages.json", "w") as f:
        json.dump(lang_data, f, indent=2)
    print(f"  -> Language data for {len(lang_data)} repos")
    return lang_data


def collecting_events(username, headers):
    """
    Step 5: Fetch recent public events.

    Events include things like push events, pull request events, issue events,
    and star events. GitHub only keeps the last 90 public events per user
    (3 pages of 30), so this is limited in scope.

    In my case this returned 0 events because my account activity is recent
    and event retention is short. The data is still collected as a fallback
    in case it's useful for future analysis.

    Saves to: data/events.json
    """
    print(f"[5/6] Collecting recent events...")
    events = []
    for page in range(1, 4):
        resp = request_making(
            f"{BASE_URL}/users/{username}/events/public",
            headers,
            params={"per_page": 30, "page": page}
        )
        if resp.status_code == 200:
            batch = resp.json()
            if not batch:
                break
            events.extend(batch)

    with open(Dataset_Path / "events.json", "w") as f:
        json.dump(events, f, indent=2)
    print(f"  -> Saved {len(events)} events")
    return events


def collect_repo_contents(repos, headers, max_repos=15):
    """
    Step 6: Fetch README files and top-level file listings.

    For the top 15 repos (sorted by stars), this function:
    1. Downloads the README file content by decoding it from base64
    2. Lists all top-level files and directories with their names and sizes

    The README content is used for the documentation quality LLM task,
    where the LLM rates each README on clarity, setup instructions,
    usage examples, and completeness.

    The file listings are used for the naming convention analysis,
    where the LLM evaluates file naming patterns across repos.

    Saves to: data/readmes.json, data/file_listings.json
    """
    print(f"[6/6] Collecting README files and repo contents...")
    readmes = {}
    file_listings = {}
    # sorting by stars so we prioritize the most visible repos
    sorted_repos = sorted(repos, key=lambda r: r.get("stargazers_count", 0), reverse=True)

    for repo in sorted_repos[:max_repos]:
        repo_name = repo["full_name"]

        # fetching the README file
        # GitHub returns it as base64 encoded content which we decode to plain text
        resp = request_making(
            f"{BASE_URL}/repos/{repo_name}/readme",
            headers
        )
        if resp.status_code == 200:
            import base64
            readme_data = resp.json()
            try:
                content = base64.b64decode(readme_data.get("content", "")).decode("utf-8", errors="replace")
                readmes[repo["name"]] = content
            except Exception:
                pass

        # fetching the top-level directory listing
        # this gives us file names, types (file vs directory), and sizes
        resp = request_making(
            f"{BASE_URL}/repos/{repo_name}/contents",
            headers
        )
        if resp.status_code == 200:
            files = [
                {"name": f["name"], "type": f["type"], "size": f.get("size", 0)}
                for f in resp.json()
            ]
            file_listings[repo["name"]] = files

    with open(Dataset_Path / "readmes.json", "w") as f:
        json.dump(readmes, f, indent=2)
    with open(Dataset_Path / "file_listings.json", "w") as f:
        json.dump(file_listings, f, indent=2)
    print(f"  -> Saved {len(readmes)} READMEs, {len(file_listings)} file listings")
    return readmes, file_listings


def generate_summary(username):
    """
    Print a summary of all collected data files and their sizes.
    This runs at the end so i can verify everything was saved correctly.
    """
    print("\n" + "="*60)
    print("DATA COLLECTION SUMMARY")
    print("="*60)
    for fname in Dataset_Path.glob("*.json"):
        size = fname.stat().st_size
        print(f"  {fname.name:25s} {size/1024:.1f} KB")
    print("="*60)
    print(f"Data saved to: {Dataset_Path.resolve()}")
    print("Next step: Run the analysis notebook or the Streamlit dashboard.")


def main():
    """
    Entry point. Parses command line arguments and runs all 6 collection
    steps in sequence. The --username flag is required, --token is optional
    but recommended.

    Without a token: 60 API requests per hour (can be slow for many repos)
    With a token: 5000 API requests per hour (more than enough)
    """
    parser = argparse.ArgumentParser(description="Collect GitHub data for self-analysis")
    parser.add_argument("--username", required=True, help="Your GitHub username")
    parser.add_argument("--token", default=None, help="GitHub personal access token (optional)")
    args = parser.parse_args()

    # setting up the headers for all API requests
    # the Accept header tells GitHub we want v3 JSON responses
    headers = {"Accept": "application/vnd.github.v3+json"}
    if args.token:
        headers["Authorization"] = f"token {args.token}"
        print("Using authenticated requests (higher rate limit).\n")
    else:
        print("Running without token (60 req/hr limit). Pass --token for 5000 req/hr.\n")

    # running all 6 collection steps in order
    # each step depends on the previous ones (e.g., commits needs the repo list)
    profile = collecting_profile_user(args.username, headers)
    if not profile:
        print("Could not fetch profile. Check the username and try again.")
        return

    repos = collecting_repos(args.username, headers)
    commits = collecting_commits(args.username, repos, headers)
    languages = collecting_languages(repos, headers)
    events = collecting_events(args.username, headers)
    readmes, file_listings = collect_repo_contents(repos, headers)
    generate_summary(args.username)


if __name__ == "__main__":
    main()