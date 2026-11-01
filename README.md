# FairRec: Fairness-Aware Recommender System

Recommender on the KuaiRec dataset with a tunable relevance-fairness
re-ranker, a contextual bandit, and off-policy evaluation.

Hackathon problem: Fairness-Aware Recommender with Cold-Start Handling
and Off-Policy Evaluation.

## Team
- Ayushman Gupta: backend (re-ranker, metrics, bandit, OPE, API)
- Mohit Shukla: data, ALS model, frontend
- Priyanshu Raj: to be added

## What is implemented
- ALS baseline (implicit), trained on KuaiRec
- Fairness re-ranker with a slider `lam` (0 = relevance only, 1 = max fairness)
- Metrics: nDCG@K, exposure Gini, catalog coverage, group (head/mid/tail) exposure
- LinUCB contextual bandit with propensity logging (simulation)
- Off-policy evaluation: IPS, SNIPS, Doubly Robust with bootstrap CIs
- FastAPI layer serving precomputed results as JSON

## Architecture
KuaiRec -> ALS candidates (top-100) -> fairness re-ranker -> top-10 list
-> metrics -> results/*.json -> FastAPI -> frontend dashboard

Bandit and OPE run on a simulated logged-bandit dataset with known propensities.

## Setup
    git clone https://github.com/Ayushman-codex/fairrec.git
    cd fairrec
    python -m venv venv
    venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    pip install fastapi uvicorn

## Run the API
    $env:RESULTS_DIR="results_demo"
    python -m uvicorn src.api:app --reload --port 8000
    # docs at http://127.0.0.1:8000/docs

`results_demo/` holds results from synthetic dummy data.
Real KuaiRec results go in `results/`.

## Data
KuaiRec is not stored in this repo. It is downloaded in Colab from
https://zenodo.org/records/18164998 (KuaiRec.zip).
Dataset and model files are excluded via .gitignore.

## Results
To be filled with real numbers after running on KuaiRec.
Do not report dummy-data numbers as results.

## Limitations
- Bandit and OPE results are simulations, not a real online experiment.
- Fairness is measured as exposure across popularity groups (head/mid/tail),
  not provider or demographic fairness.
- KuaiRec has no logging propensities, so OPE uses a simulated log.