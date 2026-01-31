import pandas as pd
import numpy as np


class CandidateGenerator:
    def __init__(self, ials_model, toppop_model=None, default_ials_score=0.0):
        self.ials = ials_model
        self.toppop = toppop_model
        self.default_ials_score = float(default_ials_score)

    @staticmethod
    def _build_seen_map(hist: pd.DataFrame):
        if hist is None or hist.empty:
            return {}
        grp = hist.groupby("user_id")["item_id"].agg(lambda x: set(x.values))
        return grp.to_dict()

    def generate(
        self,
        users,
        k_als=200,
        k_toppop=200,
        hist=None,
        filter_seen=True,
        add_source=True,
        add_scores=True,
    ) -> pd.DataFrame:
        users = list(users)
        seen_map = self._build_seen_map(hist) if (filter_seen and hist is not None) else {}

        rows = []
        for u in users:
            cand = {}

            als_items = self.ials.recommend(u, k=k_als, filter_already_liked_items=False) if self.ials is not None else []
            for it in als_items:
                cand[it] = cand.get(it, set())
                cand[it].add("als")

            if self.toppop is not None:
                tp_items = self.toppop.recommend(u, k=k_toppop)
                for it in tp_items:
                    cand[it] = cand.get(it, set())
                    cand[it].add("toppop")

            if filter_seen:
                seen = seen_map.get(u, set())
            else:
                seen = set()

            for it, srcs in cand.items():
                if it in seen:
                    continue

                if add_scores:
                    ials_score = self.ials.predict(u, it) if self.ials is not None else self.default_ials_score
                    toppop_score = self.toppop.predict_score(u, it) if self.toppop is not None else 0.0
                else:
                    ials_score = np.nan
                    toppop_score = np.nan

                row = {
                    "user_id": u,
                    "item_id": it,
                    "ials_score": float(ials_score),
                    "toppop_score": float(toppop_score),
                }

                if add_source:
                    row["cand_source"] = "+".join(sorted(srcs))

                rows.append(row)

        out = pd.DataFrame(rows)
        if out.empty:
            cols = ["user_id", "item_id", "ials_score", "toppop_score"] + (["cand_source"] if add_source else [])
            return pd.DataFrame(columns=cols)

        subset = ["user_id", "item_id"]
        out = out.sort_values(subset).drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
        return out

