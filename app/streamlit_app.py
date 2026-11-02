import json
from pathlib import Path

import pandas as pd
import streamlit as st

RES = Path(__file__).resolve().parent.parent / "results"
LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]

st.set_page_config(page_title="FairRec", layout="wide")


@st.cache_data
def load(name):
    with open(RES / name, encoding="utf-8") as f:
        return json.load(f)


st.title("FairRec: Fairness-Aware Recommender on KuaiRec")
st.caption("ALS candidates re-ranked for item-exposure fairness. "
           "The bandit and off-policy sections are simulations.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Baselines", "Fairness trade-off", "Recommendations", "Bandit and OPE (simulation)"])

with tab1:
    st.subheader("ALS vs popularity (K=10, 5,992 users)")
    b = pd.DataFrame(load("baselines.json")).set_index("model")
    b = b.rename(columns={"recall10": "Recall@10", "hitrate10": "HitRate@10",
                          "ndcg10": "NDCG@10", "users": "Users"})
    st.dataframe(b.style.format({"Recall@10": "{:.5f}", "HitRate@10": "{:.4f}",
                                 "NDCG@10": "{:.5f}", "Users": "{:.0f}"}),
                 use_container_width=True)
    st.markdown("Positive = cumulative watch ratio > 2 in the test period, "
                "items seen in training removed. Split is chronological. "
                "Absolute values are small because positives are sparse.")

with tab2:
    t = pd.DataFrame(load("tradeoff.json"))
    shares = pd.DataFrame(t["group_share"].tolist()).add_prefix("share_")
    t = pd.concat([t.drop(columns="group_share"), shares], axis=1)

    lam = st.select_slider("Fairness weight (lam): 0 = relevance only, 1 = max fairness",
                           options=LAMS, value=0.0, key="lam_tradeoff")
    row = t[t["lam"] == lam].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tail share of exposure", f"{row['share_tail']:.1%}")
    c2.metric("Catalog coverage", f"{row['coverage']:.1%}")
    c3.metric("Exposure Gini", f"{row['gini']:.3f}")
    c4.metric("NDCG@10", f"{row['ndcg']:.5f}")

    st.markdown("**Exposure share by popularity group, for every lam**")
    st.bar_chart(t.set_index("lam")[["share_head", "share_mid", "share_tail"]])
    left, right = st.columns(2)
    left.markdown("**Catalog coverage**")
    left.line_chart(t.set_index("lam")[["coverage"]])
    right.markdown("**Exposure Gini (lower = more even)**")
    right.line_chart(t.set_index("lam")[["gini"]])
    st.caption("3,000-user sample. Fairness target: exposure proportional to item share "
               "(head 10%, mid 40%, tail 50%). NDCG changes are within noise. "
               "Only about 0.5% of candidates are tail items, which limits the effect.")

with tab3:
    lam2 = st.select_slider("Fairness weight (lam)", options=LAMS, value=0.5, key="lam_recs")
    base = load("recs_lam_0.0.json")
    recs = load(f"recs_lam_{lam2}.json")
    uid = st.selectbox("User", sorted(recs.keys(), key=int))
    base_ids = {r["item_id"] for r in base[uid]}
    df = pd.DataFrame(recs[uid])

    def why(r):
        if r["item_id"] in base_ids:
            return "In the relevance-only top-10"
        return f"Promoted by the fairness re-ranker to raise {r['group']} exposure"

    df["why"] = df.apply(why, axis=1)
    st.dataframe(df[["rank", "item_id", "group", "score", "why"]],
                 use_container_width=True, hide_index=True)
    st.caption("Explanations come from the re-ranking logic (compared with the "
               "lam=0 list). Per-user history explanations are not implemented.")

with tab4:
    st.warning("Simulation on synthetic users with known propensities. "
               "Not a real experiment and not KuaiRec data.")
    bd = load("bandit.json")
    st.markdown("**Cumulative reward: LinUCB vs random**")
    st.line_chart(pd.DataFrame({"LinUCB": bd["linucb"], "Random": bd["random"]},
                               index=bd["round"]))
    o = load("ope.json")
    st.markdown(f"**Off-policy estimates of the random policy** "
                f"(simulated true value {o['true_value']:.3f})")
    st.dataframe(pd.DataFrame(o["estimators"]).set_index("name"), use_container_width=True)
    st.caption("IPS variance grows as the logging policy becomes more deterministic "
               "(std 0.019 at eps=0.5 to 0.121 at eps=0.02).")