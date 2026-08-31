import os
import numpy as np
import pandas as pd
import cornac
from cornac.eval_methods import RatioSplit

SEED = 42
VERBOSE = False


def sample_with_lowest_sparsity(df: pd.DataFrame, sample_frac: float = 0.1, n_samples: int = 1000, seed: int = SEED) -> pd.DataFrame:
    """Samples a subset of the dataframe while minimizing matrix sparsity."""
    np.random.seed(seed)
    best_sample = None
    lowest_sparsity = 1.0

    for _ in range(n_samples):
        sample = df.sample(frac=sample_frac, random_state=np.random.randint(0, int(1e9)))
        n_users = sample['user_id'].nunique()
        n_recipes = sample['recipe_id'].nunique()
        n_interactions = len(sample)

        if n_users == 0 or n_recipes == 0:
            continue

        sparsity = 1 - (n_interactions / (n_users * n_recipes))
        if sparsity < lowest_sparsity:
            lowest_sparsity = sparsity
            best_sample = sample

    return best_sample if best_sample is not None else df


# --- Model Definitions ---

def run_most_pop(ratio_split, metrics) -> pd.DataFrame:
    model = cornac.models.MostPop()
    exp = cornac.Experiment(eval_method=ratio_split, models=[model], metrics=metrics, user_based=True)
    exp.run()
    res = exp.result[0]
    return pd.DataFrame([{"Model": res.model_name, **res.metric_avg_results}])


def run_matrix_factorization(ratio_split, metrics) -> pd.DataFrame:
    model = cornac.models.MF(k=10, max_iter=50, learning_rate=0.01, lambda_reg=0.0,
                            use_bias=False, verbose=VERBOSE, seed=SEED, name="MF(K=10)")
    exp = cornac.Experiment(eval_method=ratio_split, models=[model], metrics=metrics)
    exp.run()
    res = exp.result[0]
    return pd.DataFrame([{"Model": res.model_name, **res.metric_avg_results}])


def run_bivaecf(ratio_split, metrics) -> pd.DataFrame:
    model = cornac.models.BiVAECF(
        k=10, encoder_structure=[20], act_fn='tanh', likelihood='pois',
        n_epochs=20, batch_size=256, learning_rate=0.001, beta_kl=1.0,
        verbose=VERBOSE, use_gpu=True
    )
    exp = cornac.Experiment(eval_method=ratio_split, models=[model], metrics=metrics)
    exp.run()
    res = exp.result[0]
    return pd.DataFrame([{"Model": res.model_name, **res.metric_avg_results}])


def run_vaecf(ratio_split, metrics) -> pd.DataFrame:
    model = cornac.models.VAECF(
        k=10, autoencoder_structure=[10], act_fn="tanh", likelihood="mult",
        n_epochs=20, batch_size=256, learning_rate=0.001, beta=1.0,
        seed=123, use_gpu=True, verbose=VERBOSE
    )
    exp = cornac.Experiment(eval_method=ratio_split, models=[model], metrics=metrics, user_based=True)
    exp.run()
    res = exp.result[0]
    return pd.DataFrame([{"Model": res.model_name, **res.metric_avg_results}])


def run_gmf(ratio_split, metrics) -> pd.DataFrame:
    model = cornac.models.GMF(
        num_factors=10, num_epochs=20, learner="adam", backend="pytorch",
        batch_size=256, lr=0.001, num_neg=50, seed=123
    )
    exp = cornac.Experiment(eval_method=ratio_split, models=[model], metrics=metrics)
    exp.run()
    res = exp.result[0]
    return pd.DataFrame([{"Model": res.model_name, **res.metric_avg_results}])


def run_neumf(ratio_split, metrics) -> pd.DataFrame:
    model = cornac.models.NeuMF(
        num_factors=8, layers=[64, 32, 16, 8], act_fn="tanh", learner="adam",
        backend="pytorch", num_epochs=20, batch_size=256, lr=0.001, num_neg=50, seed=123
    )
    exp = cornac.Experiment(eval_method=ratio_split, models=[model], metrics=metrics)
    exp.run()
    res = exp.result[0]
    return pd.DataFrame([{"Model": res.model_name, **res.metric_avg_results}])


def run_mlp(ratio_split, metrics) -> pd.DataFrame:
    model = cornac.models.MLP(
        layers=[64, 32, 16, 8], act_fn="tanh", learner="adam", backend="pytorch",
        num_epochs=20, batch_size=256, lr=0.001, num_neg=50, seed=123
    )
    exp = cornac.Experiment(eval_method=ratio_split, models=[model], metrics=metrics)
    exp.run()
    res = exp.result[0]
    return pd.DataFrame([{"Model": res.model_name, **res.metric_avg_results}])


def run_coe(ratio_split, metrics) -> pd.DataFrame:
    model = cornac.models.COE(
        k=20, max_iter=20, learning_rate=0.05, lamda=0.001,
        batch_size=256, name="COE", trainable=True, verbose=VERBOSE
    )
    exp = cornac.Experiment(eval_method=ratio_split, models=[model], metrics=metrics, user_based=True)
    exp.run()
    res = exp.result[0]
    return pd.DataFrame([{"Model": res.model_name, **res.metric_avg_results}])


def evaluate_landmarkers(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Evaluates selected models (landmarkers) on the given dataset."""
    metrics = [
        cornac.metrics.MAE(),
        cornac.metrics.AUC(),
        cornac.metrics.Precision(k=10),
        cornac.metrics.Recall(k=10),
        cornac.metrics.NDCG(k=10),
        cornac.metrics.RMSE()
    ]

    ratio_split = RatioSplit(
        data=df.values.tolist(),
        test_size=0.1,
        rating_threshold=1.0,
        exclude_unknowns=False,
        verbose=VERBOSE,
    )

    landmarker_functions = [
        run_most_pop,
        # Add additional landmarkers here as needed (e.g. run_matrix_factorization, run_gmf)
    ]

    results = []
    for func in landmarker_functions:
        try:
            res_df = func(ratio_split, metrics)
            res_df["dataset"] = dataset_name
            results.append(res_df)
        except Exception as e:
            print(f"Error executing landmark function {func.__name__}: {e}")

    if not results:
        return pd.DataFrame()

    final_df = pd.concat(results, ignore_index=True)
    final_df.rename(columns={
        "MAE": "mae",
        "RMSE": "rmse",
        "AUC": "auc",
        "NDCG@10": "ndcg_10",
        "Precision@10": "pre_10",
        "Recall@10": "rec_10"
    }, inplace=True)

    return final_df


def extract_landmarkers_from_folder(datasets_dir: str, output_dir: str):
    """Processes all datasets in folder and evaluates landmarkers."""
    os.makedirs(output_dir, exist_ok=True)
    accumulated_results = []
    files = [f for f in os.listdir(datasets_dir) if f.endswith('.csv')]

    for i, file_name in enumerate(files, start=1):
        file_path = os.path.join(datasets_dir, file_name)
        dataset_name = os.path.splitext(file_name)[0]
        print(f"\n[{i}/{len(files)}] Processing landmarkers for: {dataset_name}")

        try:
            df = pd.read_csv(file_path)
            sample = sample_with_lowest_sparsity(df, sample_frac=0.1)
            result = evaluate_landmarkers(sample, dataset_name)

            if not result.empty:
                accumulated_results.append(result)
                print(f"Successfully evaluated landmarkers for {dataset_name}")

            if i % 5 == 0 and accumulated_results:
                partial_df = pd.concat(accumulated_results, ignore_index=True)
                partial_path = os.path.join(output_dir, f"landmarkers_checkpoint_{i}.csv")
                partial_df.to_csv(partial_path, index=False)
                print(f"Saved partial checkpoint: {partial_path}")

        except Exception as e:
            print(f"Error processing landmarkers for file {file_name}: {e}")

    if accumulated_results:
        final_df = pd.concat(accumulated_results, ignore_index=True)
        final_path = os.path.join(output_dir, "landmarkers_final.csv")
        final_df.to_csv(final_path, index=False)
        print(f"\nSaved final landmark results to: {final_path}")


if __name__ == "__main__":
    DATASETS_DIR = r"C:\Users\iaras\RS\3pair"
    OUTPUT_DIR = r"C:\Users\iaras\RS\output"

    extract_landmarkers_from_folder(DATASETS_DIR, OUTPUT_DIR)