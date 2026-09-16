# Conversation Classification Report

Source: `topic_distribution.csv`  |  **51,571 conversations**  |  topic-only classification (dominant LDA topic per side).

Topic to class map: question Business = `user_topic_1`; answer Business = `agent_topic_1/6/8/10`; answer Neutral = `agent_topic_12`; everything else Resident.

## Mode 1 — Question only
Category taken from the question's dominant topic. Every conversation is categorized (no Neutral on the question side).

| category | conversations | share |
|---|---|---|
| Resident | 38,033 | 73.7% |
| Business | 13,538 | 26.3% |

## Mode 2 — Answer only
Category taken from the answer's dominant topic.

| category | conversations | share |
|---|---|---|
| Resident | 32,037 | 62.1% |
| Business | 8,616 | 16.7% |
| Neutral | 10,918 | 21.2% |

## Mode 3 — Both (question and answer must agree)
A final category is assigned only when the question class and the answer class are the same Resident/Business value; otherwise the conversation is Uncategorized.

| category | conversations | share |
|---|---|---|
| Resident | 25,139 | 48.7% |
| Business | 2,267 | 4.4% |
| Uncategorized | 24,165 | 46.9% |

## Question × Answer agreement
Rows = question class, columns = answer class.

| question \ answer | Business | Neutral | Resident |
|---|---|---|---|
| Business | 2,267 | 4,373 | 6,898 |
| Resident | 6,349 | 6,545 | 25,139 |

Agreement rate (question and answer give the same Resident/Business label): **27,406 / 51,571 = 53.1%**. The remaining 24,165 (46.9%) disagree or have a Neutral answer, so they stay Uncategorized in Mode 3.
