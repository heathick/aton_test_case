from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix


class iALS:
    def __init__(self, factors=64, regularization=0.01, iterations=20, alpha=40, default_score=0.0):
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations
        )
        self.alpha = alpha
        self.default_score = float(default_score)

        self.user_id_invmap = {}
        self.item_id_invmap = {}
        self.user_id_map = {}
        self.item_id_map = {}
        self.user_items_matrix = None

    def fit(self, df):
        user_ids = df["user_id"].astype("category")
        item_ids = df["item_id"].astype("category")

        self.user_id_invmap = dict(zip(user_ids.cat.categories, range(len(user_ids.cat.categories))))
        self.item_id_invmap = dict(zip(item_ids.cat.categories, range(len(item_ids.cat.categories))))
        self.user_id_map = {inner: raw for raw, inner in self.user_id_invmap.items()}
        self.item_id_map = {inner: raw for raw, inner in self.item_id_invmap.items()}

        rows = user_ids.cat.codes
        cols = item_ids.cat.codes
        values = df["target"].astype(float).values * self.alpha

        self.user_items_matrix = coo_matrix((values, (rows, cols))).tocsr()
        self.model.fit(self.user_items_matrix)

    def predict(self, user_id, item_id):
        if self.model.user_factors is None or self.model.item_factors is None:
            return self.default_score

        u = self.user_id_invmap.get(user_id)
        i = self.item_id_invmap.get(item_id)
        if u is None or i is None:
            return self.default_score

        return float(self.model.user_factors[u].dot(self.model.item_factors[i]))

    def recommend(self, user_id, k=10, filter_already_liked_items=False):
        if self.user_items_matrix is None:
            return []

        u = self.user_id_invmap.get(user_id)
        if u is None:
            return []

        inner_items, _ = self.model.recommend(
            userid=u,
            user_items=self.user_items_matrix,
            N=k,
            filter_already_liked_items=filter_already_liked_items,
        )
        return [self.item_id_map[i] for i in inner_items]

    def recommend_all(self, k=10, filter_already_liked_items=False):
        if self.user_items_matrix is None:
            return {}

        res = {}
        n_users = self.user_items_matrix.shape[0]
        for u in range(n_users):
            inner_items, _ = self.model.recommend(
                userid=u,
                user_items=self.user_items_matrix,
                N=k,
                filter_already_liked_items=filter_already_liked_items,
            )
            raw_user = self.user_id_map[u]
            res[raw_user] = [self.item_id_map[i] for i in inner_items]
        return res

