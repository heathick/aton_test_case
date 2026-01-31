import pandas as pd
from ..models import TopPop, iALS
from ..candidates import CandidateGenerator
from ..features import FeatureBuilder


def inference_pipeline(train, test, items, users, toppop, ials, ranker_model, k=10):
    gen = CandidateGenerator(ials, toppop)
    
    all_users = test['user_id'].unique()
    cand = gen.generate(
        users=all_users,
        k_als=200,
        k_toppop=200,
        hist=train,
        filter_seen=True
    )
    
    cand['last_watch_dt'] = test['last_watch_dt'].iloc[0]
    
    fb = FeatureBuilder(items, users)
    X, feature_cols = fb.transform(train, cand)
    
    scores = ranker_model.predict(X[feature_cols])
    cand['score'] = scores
    
    recommendations = {}
    for user_id in all_users:
        user_cand = cand[cand['user_id'] == user_id].nlargest(k, 'score')
        recommendations[user_id] = user_cand['item_id'].tolist()
    
    return recommendations

