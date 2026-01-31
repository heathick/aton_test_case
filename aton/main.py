import pandas as pd
from src.pipeline.train import prepare_data, train_models, generate_candidates, prepare_features
from src.utils.metrics import evaluate_ranking_model


if __name__ == '__main__':
    train_stage_1, valid_stage_1, test, items, users = prepare_data()
    
    print("Training models...")
    toppop, ials = train_models(train_stage_1)
    
    print("Generating candidates...")
    cand = generate_candidates(
        ials, toppop, 
        valid_stage_1['user_id'].unique(), 
        train_stage_1
    )
    
    print("Preparing features...")
    X, feature_cols = prepare_features(
        cand, valid_stage_1, train_stage_1, items, users
    )
    
    print(f"Features shape: {X.shape}")
    print(f"Feature columns: {len(feature_cols)}")

