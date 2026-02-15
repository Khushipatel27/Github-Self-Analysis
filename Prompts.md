# Prompts.md — Prompt Engineering Documentation

This document contains the key prompts used in this project along with explanations of design decisions.

---

## Prompt Design Philosophy

All prompts follow these principles:
- **Role assignment**: Each prompt starts with a system prompt that sets the LLM's persona (e.g., "You are a developer psychology expert")
- **Structured input**: Data is formatted clearly with labels and numbered items
- **Explicit output format**: Every prompt specifies exactly what sections to include in the response
- **Bounded scope**: Data is sampled/truncated to fit context windows while remaining representative

---

## Prompt 1: Sentiment Analysis

**Purpose**: Classify the emotional tone of commit messages and identify mood patterns.

**System Prompt**: `You are a developer psychology expert analyzing git commit history.`

**User Prompt**:
```
Analyze the sentiment and tone of these git commit messages.
For each, classify as: positive, neutral, negative, or frustrated.
Then provide an overall summary.

Commit messages:
1. fix: resolve null pointer in auth module
2. feat: add user dashboard with charts
...

Respond with:
1. Overall sentiment breakdown (% positive, neutral, negative, frustrated)
2. Notable patterns or mood shifts
3. What the commit style reveals about the developer
```

**Why it works**: The 4-category classification (positive/neutral/negative/frustrated) captures developer-specific emotions better than generic positive/negative. Asking for percentages forces quantitative output. The "developer personality" question encourages deeper inference.

---

## Prompt 2: Topic Clustering

**Purpose**: Group repositories into thematic clusters to identify areas of expertise.

**System Prompt**: `You are a technical recruiter analyzing a developer's portfolio.`

**User Prompt**:
```
Analyze these repos and group them into thematic clusters.

Repositories:
- repo-name: description (lang: Python, topics: ml, data)
...

Respond with:
1. Identified clusters (name each, list repos)
2. Primary areas of expertise
3. Emerging interests vs established skills
4. Portfolio gaps
```

**Why it works**: The recruiter persona produces practical, hiring-relevant groupings. Including language and topics gives the LLM multiple signals for clustering. Asking for "gaps" produces actionable feedback.

---

## Prompt 3: Skill Extraction

**Purpose**: Build a comprehensive technical skill profile with proficiency levels.

**System Prompt**: `You are a senior tech lead doing a skill assessment.`

**User Prompt**:
```
Extract a comprehensive skill profile from this GitHub data.

Languages: {language_bytes_json}
File types: {file_type_counts}
Repos: - name: description (language)

Provide:
1. Technical skills with proficiency (beginner/intermediate/advanced)
2. Domain knowledge areas
3. Dev practices (CI/CD, testing, docs)
4. Recommended next skills to learn
```

**Why it works**: Providing multiple data signals (language bytes, file types, descriptions) allows triangulation. The 3-tier proficiency scale is simple enough for consistent output. The "next skills" question creates forward-looking value.

---

## Prompt 4: Documentation Quality

**Purpose**: Rate README quality and identify documentation habits.

**System Prompt**: `You are a developer experience expert.`

**User Prompt**:
```
Evaluate documentation quality of these repos based on their READMEs.

{readme_json}

Rate each (1-10) on: clarity, setup instructions, usage examples, completeness.
Then summarize overall documentation habits, strengths, weaknesses, and improvements.
```

**Why it works**: Numeric ratings (1-10) create comparable, structured output. The four dimensions cover the most important README aspects. Truncating READMEs to 1000 chars keeps prompts manageable while capturing the "feel" of each README.

---

## Prompt 5: Career Progression

**Purpose**: Narrate the developer's technical journey from their project timeline.

**System Prompt**: `You are a tech career counselor.`

**User Prompt**:
```
Write a career progression narrative from this GitHub timeline.

Account created: 2020-01
Bio: CS student interested in ML

Timeline:
[{"date": "2020-03", "name": "hello-world", "language": "Python"}, ...]

Write:
1. Journey narrative (phases/chapters)
2. Key turning points or pivots
3. Predicted next career move
4. Strengths to emphasize in interviews
```

**Why it works**: Chronological data naturally lends itself to narrative. The "chapters" framing helps the LLM organize output temporally. Interview strengths make output directly actionable for job seekers.

---

## Prompt 6: Next Project Suggestions

**Purpose**: Generate personalized project recommendations.

**System Prompt**: `You are a tech mentor giving personalized advice.`

**User Prompt**:
```
Suggest what this developer should build next.

Skills: ["Python", "JavaScript", ...]
Recent projects: [{"name": "...", "desc": "..."}]
Topics explored: ["ml", "web", ...]

Suggest:
3 skill-building projects, 2 stretch projects, 1 portfolio-booster.
Include WHY it's a good fit and estimated complexity.
```

**Why it works**: The 3-2-1 structure creates variety. Requiring "WHY" prevents generic suggestions. "Estimated complexity" helps with planning. Providing both skills and recent projects lets the LLM suggest natural extensions.

---

## Tips for Improving Prompts

1. **Add examples**: For classification tasks, include 2-3 labeled examples
2. **Use JSON output**: For structured data, ask the LLM to respond in JSON format
3. **Chain prompts**: Use output from one prompt as input to the next
4. **Temperature tuning**: Use low temperature (0.1-0.3) for factual analysis, higher (0.7) for creative suggestions
5. **Context window management**: Always truncate input data to stay within limits — quality > quantity
