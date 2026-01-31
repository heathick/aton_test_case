import pandas as pd


class TopPop:
    def __init__(self, train_df, window_size=3):
        self.top_items = train_df[
            train_df['last_watch_dt'] == (train_df['last_watch_dt'].max() - pd.Timedelta(days=window_size))
        ]['item_id'].value_counts().index.tolist()[:100]
        self.item_set = set(self.top_items)

    def predict_score(self, user_id, item_id):
        return 1.0 if item_id in self.item_set else 0.0

    def recommend(self, user_id, k=10):
        return self.top_items[:k]

