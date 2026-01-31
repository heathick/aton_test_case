import pandas as pd
import numpy as np


class FeatureBuilder:
    def __init__(self, items: pd.DataFrame, users: pd.DataFrame):
        self.items = items
        self.users = users

        self.item_cols = [c for c in [
            'item_id', 'content_type', 'genres', 'actors', 'studios',
            'age_rating', 'for_kids', 'release_year'
        ] if c in items.columns]

        self.user_cols = [c for c in [
            'user_id', 'age', 'income', 'sex', 'kids_flg'
        ] if c in users.columns]

    def transform(self, hist: pd.DataFrame, today: pd.DataFrame, day: str = None, keep_id_cols: bool = True):
        X = today.copy()
        
        if 'last_watch_dt' not in X.columns:
            if day is None:
                raise ValueError("В today нет last_watch_dt и не передан day. Передай day='YYYY-MM-DD'.")
            X['last_watch_dt'] = day
        elif day is None:
            day = X['last_watch_dt'].iloc[0]

        X = X.merge(self.items[self.item_cols], on='item_id', how='left')
        X = X.merge(self.users[self.user_cols], on='user_id', how='left')

        has_hist = hist is not None and not hist.empty
        global_mean_watched = float(hist['watched_pct'].mean()) if has_hist else 0.0

        user_stats = hist.groupby('user_id').agg(
            mean_user_watched_pct=('watched_pct', 'mean'),
            user_watch_cnt=('item_id', 'count'),
        ) if has_hist else pd.DataFrame(columns=['mean_user_watched_pct', 'user_watch_cnt'])
        
        X = X.merge(user_stats, on='user_id', how='left')
        X['mean_user_watched_pct'] = X['mean_user_watched_pct'].fillna(global_mean_watched)
        X['user_watch_cnt'] = X['user_watch_cnt'].fillna(0)

        if has_hist and 'content_type' in self.items.columns:
            h_ct = hist.merge(self.items[['item_id', 'content_type']], on='item_id', how='left')
            user_movie_share = h_ct.assign(is_movie=(h_ct['content_type'] == 'movie').astype(int)).groupby('user_id')['is_movie'].mean().rename('user_movie_share')
            X['user_movie_share'] = X['user_id'].map(user_movie_share).fillna(0.0)
        else:
            X['user_movie_share'] = 0.0

        item_stats = hist.groupby('item_id').agg(
            item_mean_watched_pct=('watched_pct', 'mean'),
            item_popularity=('user_id', 'count'),
        ) if has_hist else pd.DataFrame(columns=['item_mean_watched_pct', 'item_popularity'])
        
        if has_hist:
            item_stats['log_item_popularity'] = np.log1p(item_stats['item_popularity'])
        else:
            item_stats['log_item_popularity'] = pd.Series(dtype=float)
        
        X = X.merge(item_stats, on='item_id', how='left')
        X['item_mean_watched_pct'] = X['item_mean_watched_pct'].fillna(global_mean_watched)
        X['item_popularity'] = X['item_popularity'].fillna(0)
        X['log_item_popularity'] = X['log_item_popularity'].fillna(0.0)

        if has_hist:
            seen_index = pd.MultiIndex.from_frame(hist[['user_id', 'item_id']].drop_duplicates())
            X['user_seen_item'] = pd.MultiIndex.from_frame(X[['user_id', 'item_id']]).isin(seen_index).astype(int)
        else:
            X['user_seen_item'] = 0

        if has_hist:
            ui_cnt = hist.groupby(['user_id', 'item_id']).size().rename('user_item_watch_cnt')
            X = X.merge(ui_cnt.reset_index(), on=['user_id', 'item_id'], how='left')
            X['user_item_watch_cnt'] = X['user_item_watch_cnt'].fillna(0)
        else:
            X['user_item_watch_cnt'] = 0

        if 'ials_score' in X.columns:
            X['ials_score'] = X['ials_score'].fillna(0.0)
        if 'toppop_score' in X.columns:
            X['toppop_score'] = X['toppop_score'].fillna(0.0)

        drop_cols = {'label', 'target'}
        id_cols = {'user_id', 'item_id', 'last_watch_dt'}
        feature_cols = [c for c in X.columns if c not in drop_cols and c not in id_cols]

        if not keep_id_cols:
            X = X.drop(columns=[c for c in id_cols if c in X.columns], errors='ignore')

        return X, feature_cols

