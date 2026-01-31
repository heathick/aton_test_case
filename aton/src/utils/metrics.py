from collections import defaultdict
import numpy as np
from tqdm import tqdm


def evaluate_ranking_model(model, test_df, k=10):
    gt = test_df[test_df['target'] == 1].groupby('user_id')['item_id'].apply(set).to_dict()
    all_users = gt.keys()

    metrics = defaultdict(list)

    for user_id in tqdm(all_users):
        true_items = gt.get(user_id, set())
        if not true_items:
            continue

        pred_items = model.recommend(user_id, k=k)

        hit = any(item in true_items for item in pred_items)
        metrics['HR@K'].append(int(hit))

        precision = sum(1 for item in pred_items if item in true_items) / k
        metrics['Precision@K'].append(precision)

        recall = sum(1 for item in pred_items if item in true_items) / len(true_items)
        metrics['Recall@K'].append(recall)

        dcg = 0.0
        for idx, item in enumerate(pred_items):
            if item in true_items:
                dcg += 1 / np.log2(idx + 2)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_items), k)))
        ndcg = dcg / idcg if idcg > 0 else 0
        metrics['NDCG@K'].append(ndcg)

    print(f"Evaluated on {len(all_users)} users")
    return {m: np.mean(v) for m, v in metrics.items()}

