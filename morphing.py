import os
import numpy as np
import pandas as pd


def generate_morphed_datasets(
    dataset_a_path: str,
    dataset_b_path: str,
    output_folder: str,
    swap_ratio: float = 0.1,
    prefix: str = "morphed",
    seed: int = 42
) -> list:
    """
    Generates intermediate synthetic datasets between dataset A and dataset B via
    progressive row swapping.

    MORPHING TRAJECTORY & RANDOMNESS:
    - In each step, `swap_ratio` (default 10%) of the original source rows are
      progressively replaced by unique rows selected from the target dataset.
    - `seed` controls the permutation order of both source and target indices.
    - Changing `seed` yields a COMPLETELY NEW TRAJECTORY of intermediate datasets.

    Parameters:
        dataset_a_path (str): Path to source dataset CSV (user_id, recipe_id, rating).
        dataset_b_path (str): Path to target dataset CSV (user_id, recipe_id, rating).
        output_folder (str): Directory where intermediate datasets will be saved.
        swap_ratio (float): Fraction of rows to swap in each step (e.g., 0.1 = 10%).
        prefix (str): Prefix string for created filenames.
        seed (int): Random seed controlling index permutations and morphing trajectory.

    Returns:
        list: File paths of the generated intermediate datasets.
    """
    os.makedirs(output_folder, exist_ok=True)

    df_source = pd.read_csv(dataset_a_path)
    df_target = pd.read_csv(dataset_b_path)

    # Validate dataset schemas
    required_cols = {'user_id', 'recipe_id', 'rating'}
    if not required_cols.issubset(df_source.columns) or not required_cols.issubset(df_target.columns):
        raise ValueError(f"Both input datasets must contain columns: {required_cols}")

    np.random.seed(seed)
    # Balance row counts if source and target differ in length
    min_length = min(len(df_source), len(df_target))
    df_source = df_source.sample(n=min_length, random_state=seed).reset_index(drop=True)
    df_target = df_target.sample(n=min_length, random_state=seed).reset_index(drop=True)

    total_rows = len(df_source)
    num_rows_to_swap = max(1, int(swap_ratio * total_rows))

    source_indices_permuted = np.random.permutation(total_rows)
    target_indices_used = set()

    df_intermediate = df_source.copy()
    generated_files = []

    save_counter = 1
    pointer = 0

    print(f"--- Morphing Initialized | Seed: {seed} | Total Rows: {total_rows} | Swap Step Size: {num_rows_to_swap} ---")

    while pointer < len(source_indices_permuted):
        end_idx = min(pointer + num_rows_to_swap, len(source_indices_permuted))
        subset_source = source_indices_permuted[pointer:end_idx]

        # Select target rows that have not been used yet
        available_target_indices = list(set(range(total_rows)) - target_indices_used)
        subset_target = np.random.choice(available_target_indices, size=len(subset_source), replace=False)
        target_indices_used.update(subset_target)

        # Shuffle target indices for swap
        subset_target_shuffled = np.random.permutation(subset_target)

        # Execute row replacement
        for src_idx, tgt_idx in zip(subset_source, subset_target_shuffled):
            df_intermediate.iloc[src_idx] = df_target.iloc[tgt_idx].values

        output_filename = f"{prefix}_seed{seed}_step_{save_counter}.csv"
        output_filepath = os.path.join(output_folder, output_filename)

        df_intermediate.to_csv(output_filepath, index=False)
        generated_files.append(output_filepath)

        swapped_pct = (len(target_indices_used) / total_rows) * 100
        print(f"Intermediate dataset saved: {output_filename} ({swapped_pct:.1f}% target data)")

        save_counter += 1
        pointer = end_idx

    return generated_files


if __name__ == "__main__":
    DATASET_A = r"path/to/df_source.csv"
    DATASET_B = r"path/to/df_target.csv"
    MORPH_OUTPUT = r"path/to/output_intermediate"

    generate_morphed_datasets(
        dataset_a_path=DATASET_A,
        dataset_b_path=DATASET_B,
        output_folder=MORPH_OUTPUT,
        swap_ratio=0.1,
        seed=1
    )