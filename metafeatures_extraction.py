import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import kurtosis, skew, mode, entropy
from sklearn.metrics.pairwise import pairwise_distances
from collections import defaultdict


def calculate_entropy(array: np.ndarray) -> float:
    """Calculates Shannon entropy of an array."""
    _, counts = np.unique(array, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-9))


def calculate_gini(array: np.ndarray) -> float:
    """Calculates the Gini coefficient of an array."""
    array = array.flatten()
    array = array[~np.isnan(array)]
    if len(array) == 0:
        return np.nan
    sorted_array = np.sort(array)
    n = len(array)
    cumvals = np.cumsum(sorted_array)
    return (2 * np.sum((np.arange(1, n + 1)) * sorted_array) / (n * cumvals[-1])) - (n + 1) / n


def extract_statistical_features(prefix: str, values: np.ndarray) -> dict:
    """Calculates descriptive statistics for a given array of values."""
    values = np.array(values)
    values = values[~np.isnan(values)]

    metrics = ["entropy", "gini", "kurtosis", "max", "mean", "median", "min", "mode", "sd", "skewness"]
    if len(values) == 0:
        return {f"{prefix}_{metric}": np.nan for metric in metrics}

    try:
        mode_val = mode(values.astype(int), keepdims=False).mode
        mode_val = mode_val[0] if isinstance(mode_val, (np.ndarray, list)) else mode_val
    except Exception:
        mode_val = np.nan

    return {
        f"{prefix}_entropy": calculate_entropy(values.astype(int)),
        f"{prefix}_gini": calculate_gini(values),
        f"{prefix}_kurtosis": kurtosis(values),
        f"{prefix}_max": np.max(values),
        f"{prefix}_mean": np.mean(values),
        f"{prefix}_median": np.median(values),
        f"{prefix}_min": np.min(values),
        f"{prefix}_mode": mode_val,
        f"{prefix}_sd": np.std(values),
        f"{prefix}_skewness": skew(values),
    }


def extract_strategy_a(df: pd.DataFrame, dataset_name: str) -> dict:
    """Strategy A: Descriptive statistics on ratings, row/col counts, means, and sums."""
    rating_matrix = csr_matrix((df['rating'].values,
                                (df['user_id'].astype('category').cat.codes,
                                 df['recipe_id'].astype('category').cat.codes)))

    binary_matrix = rating_matrix.copy()
    binary_matrix.data = np.ones_like(binary_matrix.data)

    col_counts = np.asarray(binary_matrix.sum(axis=0)).flatten()
    row_counts = np.asarray(binary_matrix.sum(axis=1)).flatten()

    col_means = np.asarray(rating_matrix.mean(axis=0)).flatten()
    row_means = np.asarray(rating_matrix.mean(axis=1)).flatten()

    col_sums = np.asarray(rating_matrix.sum(axis=0)).flatten()
    row_sums = np.asarray(rating_matrix.sum(axis=1)).flatten()

    metafeatures = {}
    metafeatures.update(extract_statistical_features("colCounts", col_counts))
    metafeatures.update(extract_statistical_features("colMeans", col_means))
    metafeatures.update(extract_statistical_features("colSums", col_sums))
    metafeatures.update(extract_statistical_features("rowCounts", row_counts))
    metafeatures.update(extract_statistical_features("rowMeans", row_means))
    metafeatures.update(extract_statistical_features("rowSums", row_sums))
    metafeatures.update(extract_statistical_features("ratings", df['rating'].values))

    nusers, nitems = rating_matrix.shape
    nratings = rating_matrix.count_nonzero()
    metafeatures["nusers"] = nusers
    metafeatures["nitems"] = nitems
    metafeatures["nratings"] = nratings
    metafeatures["sparsity"] = 1 - (nratings / (nusers * nitems))
    metafeatures["dataset"] = dataset_name
    return metafeatures


def extract_strategy_b(df: pd.DataFrame, dataset_name: str, threshold_user: int = 100) -> dict:
    """Strategy B: User co-rating matrix entropy and gini features."""
    rating_matrix = csr_matrix((df['rating'].values,
                                (df['user_id'].astype('category').cat.codes,
                                 df['recipe_id'].astype('category').cat.codes)))

    all_users = df['user_id'].unique()
    sampled_users = np.random.choice(all_users, min(threshold_user, len(all_users)), replace=False)

    assignments = defaultdict(list)
    assignments_count = defaultdict(int)

    for user in sampled_users:
        items = df[df['user_id'] == user]['recipe_id'].tolist()
        assignments[user] = items
        assignments_count[user] = len(items)

    preference_classes = defaultdict(list)
    for user, count_r in assignments_count.items():
        preference_classes[count_r].append(user)

    max_class = max(preference_classes.keys(), default=0)
    coratings = np.full((max_class + 1, max_class + 1), np.nan)

    for class1 in preference_classes:
        for class2 in preference_classes:
            if np.isnan(coratings[class1, class2]):
                users1 = preference_classes[class1]
                users2 = preference_classes[class2]

                common_counts = [
                    len(set(assignments[u1]) & set(assignments[u2]))
                    for u1 in users1 for u2 in users2
                ]

                if common_counts:
                    mean_common = np.mean(common_counts)
                    coratings[class1, class2] = mean_common
                    coratings[class2, class1] = mean_common

    coratings_df = pd.DataFrame(coratings).dropna(axis=0, how='all').dropna(axis=1, how='all')
    coratings_array = coratings_df.to_numpy().flatten()
    coratings_array = coratings_array[~np.isnan(coratings_array)]

    nusers, nitems = rating_matrix.shape
    nratings = rating_matrix.count_nonzero()

    return {
        "nusers": nusers,
        "nitems": nitems,
        "nratings": nratings,
        "sparsity": 1 - (nratings / (nusers * nitems)),
        "entropy": calculate_entropy(coratings_array),
        "gini": calculate_gini(coratings_array),
        "dataset": dataset_name
    }


def extract_strategy_c(df: pd.DataFrame, dataset_name: str) -> dict:
    """Strategy C: Density, rating variance, user/item count features."""
    rating_matrix = csr_matrix((df['rating'].values,
                                (df['user_id'].astype('category').cat.codes,
                                 df['recipe_id'].astype('category').cat.codes)))

    nusers, nitems = rating_matrix.shape
    nratings = rating_matrix.count_nonzero()

    row_counts = np.asarray((rating_matrix > 0).sum(axis=1)).flatten()
    col_counts = np.asarray((rating_matrix > 0).sum(axis=0)).flatten()

    metafeatures = {
        "nusers": nusers,
        "nitems": nitems,
        "nratings": nratings,
        "density": nratings / (nusers * nitems),
        "variance_ratings": np.var(df['rating'].values),
        "dataset": dataset_name
    }
    metafeatures.update(extract_statistical_features("user.count", row_counts))
    metafeatures.update(extract_statistical_features("item.count", col_counts))
    return metafeatures


def extract_strategy_d(df: pd.DataFrame, dataset_name: str, threshold_user: int = 50) -> dict:
    """Strategy D: Advanced similarity, clustering coefficient, and TF-IDF features."""
    rating_matrix = csr_matrix((df['rating'].values,
                                (df['user_id'].astype('category').cat.codes,
                                 df['recipe_id'].astype('category').cat.codes)))

    metafeatures = {
        "user.count.mean": np.mean((rating_matrix > 0).sum(axis=1)),
        "user.mean.mean": rating_matrix.sum() / rating_matrix.getnnz(),
        "item.count.mean": np.mean((rating_matrix > 0).sum(axis=0)),
        "item.mean.mean": rating_matrix.sum() / rating_matrix.getnnz()
    }

    all_users = df['user_id'].unique()
    users_sampled = np.random.choice(all_users, min(threshold_user, len(all_users)), replace=False)
    df_sample = df[df['user_id'].isin(users_sampled)]

    user_sd = df_sample.groupby('user_id')['rating'].std()
    metafeatures["user.standard_deviation.mean"] = user_sd.mean()

    sample_matrix = csr_matrix((df_sample['rating'].values,
                                (df_sample['user_id'].astype('category').cat.codes,
                                 df_sample['recipe_id'].astype('category').cat.codes)))
    sample_dense = sample_matrix.toarray()

    pearson_sim = 1 - pairwise_distances(sample_dense, metric='correlation')
    np.fill_diagonal(pearson_sim, 0)
    pearson_sim = np.nan_to_num(pearson_sim)

    metafeatures["user.average_similarity.mean"] = np.mean(np.sort(pearson_sim, axis=1)[:, -10:])

    neighbors = (pearson_sim > 0.1).astype(int)
    np.fill_diagonal(neighbors, 0)
    metafeatures["user.number_neighbours.mean"] = neighbors.sum(axis=1).mean()

    clustering = []
    for i in range(neighbors.shape[0]):
        friends = np.where(neighbors[i])[0]
        n = len(friends)
        if n > 1:
            subgraph = neighbors[np.ix_(friends, friends)]
            actual_edges = subgraph.sum() / 2
            clustering.append((2 * actual_edges + n) / (n * n - n))
        else:
            clustering.append(0)
    metafeatures["user.clustering_coefficient.mean"] = np.mean(clustering)

    item_freq = np.asarray((sample_matrix > 0).sum(axis=0)).flatten()
    idf = np.log(1 + (sample_matrix.shape[0] / (item_freq + 1e-10)))
    tfidf = (sample_matrix > 0).astype(int).multiply(idf)
    metafeatures["user.TFIDF.mean"] = tfidf.sum(axis=1).mean()

    user_items = df_sample.groupby('user_id')['recipe_id'].apply(set)
    users = list(user_items.index)
    jaccard_vals = [
        len(user_items[users[i]] & user_items[users[j]]) / len(user_items[users[i]] | user_items[users[j]])
        if len(user_items[users[i]] | user_items[users[j]]) > 0 else 0
        for i in range(len(users)) for j in range(i + 1, len(users))
    ]
    metafeatures["user_coratings.jaccard.mean"] = np.mean(jaccard_vals) if jaccard_vals else 0

    item_ratings = df_sample.groupby('recipe_id')['rating'].apply(lambda x: entropy(np.bincount(x.astype(int))))
    metafeatures["item.entropy.mean"] = item_ratings.mean()
    metafeatures["dataset"] = dataset_name
    return metafeatures


def extract_strategy_e(df: pd.DataFrame, dataset_name: str, threshold_user: int = 200) -> dict:
    """Strategy E: User rating counts, means, and variance."""
    rating_matrix = csr_matrix((df['rating'].values,
                                (df['user_id'].astype('category').cat.codes,
                                 df['recipe_id'].astype('category').cat.codes)))

    all_users = df['user_id'].unique()
    users_sampled = np.random.choice(all_users, min(threshold_user, len(all_users)), replace=False)
    df_sample = df[df['user_id'].isin(users_sampled)]

    user_ratings_dict = df_sample.groupby('user_id')['rating'].apply(list).to_dict()

    row_sums = rating_matrix.sum(axis=1).A1
    row_counts = (rating_matrix > 0).sum(axis=1).A1
    row_means = row_sums / np.where(row_counts == 0, 1, row_counts)

    return {
        "user.count.mean": np.mean((rating_matrix > 0).sum(axis=1).A1),
        "user.mean.mean": np.mean(row_means),
        "user.variance.mean": np.mean([np.var(r) for r in user_ratings_dict.values() if len(r) > 1]),
        "dataset": dataset_name
    }


def process_metafeatures(input_folder: str, output_folder: str, strategy: str = "A"):
    """
    Extracts metafeatures for all CSV files in input_folder using the specified strategy.
    
    Strategies available: 'A', 'B', 'C', 'D', 'E'
    """
    os.makedirs(output_folder, exist_ok=True)
    results = []

    strategy_func_map = {
        "A": extract_strategy_a,
        "B": extract_strategy_b,
        "C": extract_strategy_c,
        "D": extract_strategy_d,
        "E": extract_strategy_e,
    }

    if strategy not in strategy_func_map:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from 'A', 'B', 'C', 'D', 'E'.")

    func = strategy_func_map[strategy]
    entries = [e for e in sorted(os.scandir(input_folder), key=lambda x: x.name) if e.is_file() and e.name.endswith('.csv')]

    for count, entry in enumerate(entries, start=1):
        print(f"[{count}/{len(entries)}] Extracting Metafeatures ({strategy}) from: {entry.name}")
        try:
            df = pd.read_csv(entry.path, encoding='ISO-8859-1')
            meta_dict = func(df, entry.name)
            results.append(meta_dict)

            if count % 20 == 0:
                partial_df = pd.DataFrame(results)
                path = os.path.join(output_folder, f"metafeatures_{strategy}_{count}.csv")
                partial_df.to_csv(path, index=False)
                print(f"Saved partial checkpoint: {path}")

        except Exception as e:
            print(f"Error processing {entry.name}: {e}")

    if results:
        final_df = pd.DataFrame(results)
        final_path = os.path.join(output_folder, f"metafeatures_{strategy}_final.csv")
        final_df.to_csv(final_path, index=False)
        print(f"Finished! Saved all metafeatures to: {final_path}")


if __name__ == "__main__":
    DATASETS_DIR = r"C:\Users\iaras\RS\3pair"
    OUTPUT_DIR = r"C:\Users\iaras\RS\metafeatures_output"

    # Example: Run metafeature extraction strategy A
    process_metafeatures(DATASETS_DIR, OUTPUT_DIR, strategy="A")