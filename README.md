# FairRec: Fairness-Aware Recommender on KuaiRec

ALS recommender on KuaiRec with a tunable item-exposure fairness re-ranker,
a contextual bandit (simulation), and off-policy evaluation (simulation).

Hackathon problem: Fairness-Aware Recommender with Cold-Start Handling
and Off-Policy Evaluation.

## Team
- Ayushman Gupta: backend (re-ranker, metrics, bandit, OPE, API)
- Mohit Shukla: data pipeline, ALS model, frontend
- Priyanshu Raj: Data processing and Model Training

## Pipeline
KuaiRec big_matrix -> chronological split -> ALS (top-100 candidates per user)
-> fairness re-ranker (lam slider) -> metrics -> results/*.json -> FastAPI -> dashboard

## Data and evaluation setup
- Dataset: KuaiRec 2.0 (big_matrix). Not stored in this repo; downloaded in Colab
  from https://zenodo.org/records/18164998.
- Split: chronological. Train max timestamp < test min timestamp (verified).
- Positive label: a user-video pair is positive if the cumulative watch ratio
  (sum of play_duration / video_duration) in the test period is > 2, the
  definition recommended by the KuaiRec authors.
- Pairs already seen in train are removed from the ground truth, and 335
  user-video pairs with inconsistent video durations are dropped.
- 5,992 users have at least one positive and are evaluated.
- Popularity baseline and ALS use the same users, truth and K=10.

## Results: recommendation quality (K=10, 5,992 users)
| Model | Recall@10 | HitRate@10 | NDCG@10 |
|---|---|---|---|
| Popularity | 0.00053 | 0.0060 | 0.00069 |
| ALS | 0.00099 | 0.0093 | 0.00114 |

ALS is about 1.6x to 1.9x better than popularity on all three metrics.
Absolute values are small: positives are sparse (watch ratio > 2, unseen items only).
With an earlier provisional label (every observed test interaction positive),
popularity beat ALS on HitRate and NDCG, so the ranking depends on the
relevance definition.

## Results: fairness re-ranking (3,000-user sample)
Fairness definition: exposure proportional to item share across popularity
groups (head 10%, mid 40%, tail 50% of items, by train popularity).
This is item-level exposure fairness, not provider or demographic fairness.

| lam | tail share | coverage | Gini | NDCG@10 |
|---|---|---|---|---|
| 0.0 | 0.3% | 45.8% | 0.426 | 0.00183 |
| 0.5 | 1.8% | 47.6% | 0.434 | 0.00177 |
| 1.0 | 3.3% | 48.6% | 0.437 | 0.00202 |

- Tail exposure rose about 10x and coverage rose 2.8 points.
- Exposure Gini did not improve (0.426 to 0.437).
- NDCG changes are within noise at this scale, so no accuracy cost can be claimed
  or ruled out.

## Bandit and off-policy evaluation (SIMULATION)
Run on synthetic users with known propensities, not on KuaiRec.
- LinUCB: 2412 cumulative reward vs 1478 for random (3000 rounds).
- IPS / SNIPS / DR estimates of the random policy: 0.475 / 0.478 / 0.482
  (95% CI about 0.41 to 0.54) vs simulated truth 0.497.
- IPS variance grows as the logging policy becomes more deterministic
  (std 0.019 at eps=0.5 to 0.121 at eps=0.02).

## Setup and run
    git clone https://github.com/Ayushman-codex/fairrec.git
    cd fairrec
    python -m venv venv
    venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    pip install fastapi uvicorn
    $env:RESULTS_DIR="results"
    python -m uvicorn src.api:app --port 8000
    # docs: http://127.0.0.1:8000/docs

Endpoints: /health, /users, /tradeoff, /recommend/{user_id}?lam=0.5,
/bandit, /ope, /baselines.

## Limitations
- Candidate pool (top-100) has only about 0.5% tail items, which limits how much
  the re-ranker can raise tail exposure.
- ALS was trained on all observed interactions with confidence weights, while
  the test label is stricter (watch ratio > 2). This mismatch may depress
  absolute scores.
- Fairness metrics use a 3,000-user sample; recommendation metrics use all
  5,992 users.
- KuaiRec has no logging propensities, so bandit and OPE results are simulations.
- Cold-start handling is not implemented in this version.
- No confidence intervals on the ALS vs popularity comparison yet.

### 1. Dataset Preparation and Exploration
- Set up the KuaiRec 2.0 dataset in Google Colab.
- Explored the original interaction data, including user IDs, video IDs, play duration, video duration, timestamps, and watch ratios.
- Worked with the existing training, validation, and test datasets.

### 2. ALS-Based Recommendation System
- Worked with the existing Alternating Least Squares (ALS) collaborative filtering model.
- Utilized the user-item interaction matrix and user/item index mappings.
- Generated Top-100 recommendations for 6,904 users, including recommendation scores and item popularity groups.

### 3. Popularity-Based Baseline
- Generated a popularity-based recommendation baseline for comparison with ALS.
- Evaluated both approaches using:
  - Recall@10
  - HitRate@10
  - NDCG@10

### 4. Recommendation Outputs and Evaluation
- Generated `candidates.parquet`, containing user IDs, recommended item IDs, recommendation scores, and head/mid/tail group labels.
- Generated an initial `truth.parquet` file containing unique user-item pairs from the test interactions.
- Exported initial evaluation metrics in CSV and JSON formats.
- Packaged the generated outputs into `fairrec_outputs.zip`.

### 5. Temporal Split and Data Validation
- Checked the timestamp ranges of the training and test datasets and confirmed that the training period ends before the test period begins.
- Investigated repeated user-video pairs across the training and test datasets.
- Matched all 1,508,095 test interaction rows against the original KuaiRec `big_matrix.csv`.
- Successfully recovered interaction-level `video_duration` and `watch_ratio` values for the test interactions.

### Current Status

The initial recommendation pipeline, candidate generation, popularity baseline, and evaluation outputs have been generated.

**Ongoing work:**
- Finalizing the positive-interaction definition using watch ratios.
- Investigating inconsistent video-duration values for some user-video pairs.
- Regenerating the ground truth and evaluation metrics using a consistent relevance definition.
- Validating the final results before updating the output bundle.

> **Note:** The current evaluation metrics are preliminary and have not yet been finalized using the watch-ratio-based relevance definition.