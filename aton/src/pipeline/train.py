import pandas as pd
from ..models import TopPop, iALS
from ..candidates import CandidateGenerator
from ..features import FeatureBuilder


def prepare_data(train_path='train.csv', test_path='test.csv', 
                 items_path='items.csv', users_path='users.csv'):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    items = pd.read_csv(items_path)
    users = pd.read_csv(users_path)
    
    test = test[test['last_watch_dt'] == '2021-08-13']
    
    train['last_watch_dt'] = pd.to_datetime(train['last_watch_dt'])
    test['last_watch_dt'] = pd.to_datetime(test['last_watch_dt'])
    
    train_stage_1 = train[(train['last_watch_dt'] >= '2021-08-05') & (train['last_watch_dt'] < '2021-08-12')]
    valid_stage_1 = train[(train['last_watch_dt'] == '2021-08-12')]
    
    return train_stage_1, valid_stage_1, test, items, users


def train_models(train_stage_1):
    toppop = TopPop(train_stage_1)
    ials = iALS()
    ials.fit(train_stage_1)
    return toppop, ials


def generate_candidates(ials, toppop, valid_users, train_stage_1, k_als=200, k_toppop=200):
    gen = CandidateGenerator(ials, toppop)
    cand = gen.generate(
        users=valid_users,
        k_als=k_als,
        k_toppop=k_toppop,
        hist=train_stage_1,
        filter_seen=True
    )
    return cand


def prepare_features(cand, valid_stage_1, train_stage_1, items, users):
    pos = valid_stage_1[['user_id', 'item_id']].drop_duplicates()
    pos['label'] = 1
    
    train_rank = cand.merge(pos, on=['user_id', 'item_id'], how='left')
    train_rank['label'] = train_rank['label'].fillna(0).astype(int)
    train_rank['last_watch_dt'] = '2021-08-12'
    
    fb = FeatureBuilder(items, users)
    X, feature_cols = fb.transform(train_stage_1, train_rank)
    
    return X, feature_cols

