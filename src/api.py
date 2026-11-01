import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

RESULTS = Path(os.getenv("RESULTS_DIR", "results"))

app = FastAPI(title="FairRec API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


def _load(name: str):
    p = RESULTS / name
    if not p.exists():
        raise HTTPException(404, f"{name} not found in {RESULTS}")
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {"status": "ok", "results_dir": str(RESULTS)}


@app.get("/tradeoff")
def tradeoff():
    return _load("tradeoff.json")


@app.get("/users")
def users():
    recs = _load("recs_lam_0.0.json")
    return {"users": [int(u) for u in recs.keys()]}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, lam: float = 0.0):
    recs = _load(f"recs_lam_{float(lam)}.json")
    if str(user_id) not in recs:
        raise HTTPException(404, f"user {user_id} not found")
    return {"user_id": user_id, "lam": lam, "items": recs[str(user_id)]}


@app.get("/bandit")
def bandit():
    return _load("bandit.json")


@app.get("/ope")
def ope():
    return _load("ope.json")