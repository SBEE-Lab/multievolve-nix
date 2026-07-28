#!/usr/bin/env python3

"""
Script to propose mutations using trained multievolve models.

Example usage:

conda activate multievolve

p2_propose.py \
--experiment-name multievolve_example \
--protein-name example_protein \
--wt-files apex.fasta \
--training-dataset example_dataset.csv \
--mutation-pool combo_muts.csv \
--min-mutations 3 \
--max-mutations 7 \
--top-muts-per-load 3 \
--max-candidates 100000 \
--export-name multievolve_proposals
"""

import argparse
import json
import os

import matplotlib

matplotlib.use('Agg')

import pandas as pd  # noqa: E402
from Bio import SeqIO  # noqa: E402

from multievolve.featurizers import OneHotFeaturizer  # noqa: E402
from multievolve.predictors import Fcn  # noqa: E402
from multievolve.proposers import CombinatorialProposer  # noqa: E402
from multievolve.splitters import KFoldProteinSplitter  # noqa: E402
from multievolve.utils.data_utils import validate_single_substitution  # noqa: E402
from multievolve.utils.local_sweep import load_sweep_results  # noqa: E402
from multievolve.utils.reproducibility import (  # noqa: E402
    resolve_device,
    runtime_identity,
    seed_everything,
    sha256_file,
    sha256_json,
    stable_seed,
)


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def _seed(value):
    value = int(value)
    if not 0 <= value < 2**32:
        raise argparse.ArgumentTypeError('must satisfy 0 <= seed < 2**32')
    return value


def _mutation_load(value):
    value = int(value)
    if value < 2:
        raise argparse.ArgumentTypeError('must be at least 2')
    return value


def _load_mutation_pool(path, wt_seq):
    pool_df = pd.read_csv(
        path,
        header=None,
        dtype=str,
        keep_default_na=False,
        skip_blank_lines=False,
    )
    if pool_df.shape[1] != 1:
        raise ValueError('mutation pool must be a one-column, no-header CSV')

    mutations = []
    seen = set()
    for row_number, raw_mutation in enumerate(pool_df.iloc[:, 0], start=1):
        try:
            mutation, _, _, _ = validate_single_substitution(raw_mutation, wt_seq)
        except ValueError as exc:
            raise ValueError(
                f'invalid single mutation on pool row {row_number}: {raw_mutation!r}: {exc}'
            ) from exc

        if mutation in seen:
            raise ValueError(f'duplicate mutation on pool row {row_number}: {mutation}')

        seen.add(mutation)
        mutations.append(mutation)

    if not mutations:
        raise ValueError('mutation pool is empty')
    return mutations


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Propose mutations using trained models')
    parser.add_argument(
        '--experiment-name',
        required=True,
        help='Name of experiment'
    )
    parser.add_argument(
        '--protein-name',
        required=True,
        help='Name of protein'
    )
    parser.add_argument(
        '--wt-files',
        required=True,
        help='Comma separated list of paths to the wildtype FASTA files'
    )
    parser.add_argument(
        '--training-dataset',
        required=True,
        help='Path to training dataset CSV'
    )
    parser.add_argument(
        '--mutation-pool',
        required=True,
        help='Path to mutation pool CSV'
    )
    parser.add_argument(
        '--top-muts-per-load',
        type=_positive_int,
        default=3,
        help='Number of top mutations to select per load (default: 3)'
    )
    parser.add_argument(
        '--min-mutations',
        type=_mutation_load,
        default=3,
        help='Minimum number of substitutions per proposed variant (default: 3)'
    )
    parser.add_argument(
        '--max-mutations',
        type=_mutation_load,
        default=10,
        help='Maximum number of substitutions per proposed variant (default: 10)'
    )
    parser.add_argument(
        '--max-candidates',
        type=_positive_int,
        default=100000,
        help='Fail before training if the requested search exceeds this many candidates (default: 100000)'
    )
    parser.add_argument('--seed', type=_seed, default=42)
    parser.add_argument(
        '--split-seed',
        type=_seed,
        default=None,
        help='Fold-assignment seed (default: --seed)',
    )
    parser.add_argument('--ensemble-folds', type=_positive_int, default=10)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument(
        '--export-name',
        required=True,
        help='Name for export files'
    )
    args = parser.parse_args()
    args.wt_files = [f.strip() for f in args.wt_files.split(',')]
    args.split_seed = args.seed if args.split_seed is None else args.split_seed
    if args.ensemble_folds < 2:
        raise SystemExit('error: --ensemble-folds must be at least 2')
    return args


def main():

    """Main function."""

    # Parse command line arguments
    args = parse_args()

    # Define variables from args
    experiment_name = args.experiment_name
    protein_name = args.protein_name
    wt_files = args.wt_files
    training_dataset_fname = args.training_dataset
    mutation_pool_fname = args.mutation_pool
    top_muts_per_load = args.top_muts_per_load
    min_mutations = args.min_mutations
    max_mutations = args.max_mutations
    max_candidates = args.max_candidates
    export_name = args.export_name
    seed_everything(args.seed, deterministic=args.deterministic)
    actual_device = resolve_device(args.device)

    # Validate the search before loading sweep results or training final models.
    if min_mutations > max_mutations:
        raise SystemExit('error: --min-mutations must be less than or equal to --max-mutations')
    wt_seq = "".join([str(SeqIO.read(wt_file, "fasta").seq.upper()) for wt_file in wt_files])
    try:
        mutation_pool = _load_mutation_pool(mutation_pool_fname, wt_seq)
        proposer = CombinatorialProposer(
            start_seq=wt_seq,
            models=None,
            min_mutations=min_mutations,
            max_mutations=max_mutations,
            num_seeds=-1,
            mutation_pool=mutation_pool,
        )
    except ValueError as exc:
        raise SystemExit(f'error: {exc}') from exc

    candidate_counts = proposer.candidate_counts()
    total_candidates = sum(candidate_counts.values())
    print(f'Mutation-pool entries: {len(mutation_pool)}')
    print(f'Distinct mutation positions: {proposer.distinct_positions}')
    print(f'Requested mutational loads: {min_mutations}-{max_mutations}')
    for load, count in candidate_counts.items():
        print(f'  load {load}: {count} candidates')
    print(f'Total candidates: {total_candidates}')
    if total_candidates > max_candidates:
        raise SystemExit(
            f'error: requested search has {total_candidates} candidates, exceeding '
            f'--max-candidates {max_candidates}'
        )

    df = load_sweep_results(experiment_name)
    df['condition'] = (
        df['batch_size'].astype(str)
        + '|'
        + df['learning_rate'].astype(str)
        + '|'
        + df['layer_size'].astype(str)
        + '|'
        + df['num_layers'].astype(str)
        + '|'
        + df['Feature'].astype(str)
    )
    df['rank test loss'] = df.groupby('Split Method')['Test Loss'].rank()
    architecture_scores = (
        df.groupby('condition', as_index=False)[
            ['rank test loss', 'Test Loss', 'Pearson - Test', 'Spearman - Test']
        ]
        .mean()
        .sort_values(by='condition')
        .sort_values(by='rank test loss', kind='stable')
    )
    top_condition = str(architecture_scores.iloc[0]['condition'])
    batch_size, learning_rate, layer_size, num_layers, _ = top_condition.split('|')
    bs = int(batch_size)
    lr = float(learning_rate)
    hidden = int(layer_size)
    layers = int(num_layers)
    print(f'Selected architecture: batch_size={bs}, learning_rate={lr}, layer_size={hidden}, num_layers={layers}')

    config = {
        'layer_size': hidden,
        'num_layers': layers,
        'learning_rate': lr,
        'batch_size': bs,
        'optimizer': 'adam',
        'epochs': 300,
    }

    # Train the final fold ensemble with the selected architecture.
    split = KFoldProteinSplitter(
        protein_name,
        training_dataset_fname,
        wt_files,
        csv_has_header=True,
        use_cache=True,
        random_state=args.split_seed,
        y_scaling=True,
        val_split=0.15,
    )
    splits = split.generate_splits(n_splits=args.ensemble_folds)

    feature = OneHotFeaturizer(protein=protein_name, use_cache=True)
    models = []
    model_seeds = []
    for fold_index, split in enumerate(splits):
        model_seed = stable_seed(args.seed, 'propose', fold_index, top_condition)
        model_seeds.append(model_seed)
        seed_everything(model_seed, deterministic=args.deterministic)
        model_config = {
            **config,
            'seed': model_seed,
            'dataloader_seed': stable_seed(model_seed, 'dataloader'),
            'deterministic': args.deterministic,
            'device': args.device,
        }
        model = Fcn(split, feature, config=model_config, use_cache=True)
        model.run_model()
        models.append(model)

    print("Proposing mutations...")

    proposer.models = models
    proposer.propose(output_df=False)
    proposer.evaluate_proposals()
    proposer.save_proposals(f'{experiment_name}_proposals_all')

    # get top n variants per mutational load
    df = proposer.proposals
    df_ls = []
    for num_mut in range(min_mutations, max_mutations + 1):
        subset = df[df['num_muts'] == num_mut].copy()
        subset.sort_values(by='Mut_string', inplace=True)
        subset.sort_values(by='average', ascending=False, kind='stable', inplace=True)
        top_subset = subset.head(top_muts_per_load).copy()
        df_ls.append(top_subset)
    top_df = pd.concat(df_ls, ignore_index=True)

    # Export results
    print('Saving all proposals...')
    top_df.to_csv(os.path.join(splits[0].file_attrs['dataset_dir'], 'proposers/results', f'{experiment_name}_proposals_top_{top_muts_per_load}.csv'), index=False)
    top_df[['Mut_string']].to_csv(
        os.path.join(splits[0].file_attrs['dataset_dir'], f'{export_name}.csv'),
        index=False,
        header=False,
    )

    # functions for multichain proteins

    def reverse_multichain_mutations(mut_strings, chain_lengths):
        """
        Reverse the position adjustments for mutations in a multi-chain protein.

        Args:
            mut_strings (list): List of mutation strings (e.g. ["A50G/L120M", "R30K"])
            chain_lengths (list): List of lengths for each chain (e.g. [100, 150] for two chains)

        Returns:
            dict: Dictionary mapping original mutation string to dict of chain-specific mutation lists
                e.g. {"A50G/L120M": {0: ["A50G"], 1: ["L20M"]}}
        """
        # Calculate cumulative lengths for each chain
        cumulative_lengths = [sum(chain_lengths[:i]) for i in range(len(chain_lengths))]

        mutation_map = {}

        for mut_string in mut_strings:
            # Split into individual mutations
            mutations = mut_string.split('/')

            # Initialize dictionary with empty lists for each chain
            chain_mutations = {i: [] for i in range(len(chain_lengths))}

            # iterate over each mutation and add to correct chain in chain_mutations
            for mut in mutations:
                position = int(mut[1:-1])  # Extract position number
                wt_aa = mut[0]  # Wild type amino acid
                mut_aa = mut[-1]  # Mutant amino acid

                # Find which chain this mutation belongs to
                for chain_idx, start_pos in enumerate(cumulative_lengths):
                    if position <= cumulative_lengths[chain_idx + 1] if chain_idx + 1 < len(cumulative_lengths) else float('inf'):
                        # Adjust position back to chain-specific numbering
                        chain_pos = position - start_pos
                        # Add mutation to the appropriate chain's list
                        chain_mutations[chain_idx].append(f"{wt_aa}{chain_pos}{mut_aa}")
                        break

            mutation_map[mut_string] = chain_mutations

        return mutation_map

    def mutation_map_to_df(mutation_map):
        """
        Convert mutation map to DataFrame with columns for mut_string and chain-specific mutations

        Args:
            mutation_map (dict): Dictionary mapping mutation strings to chain mutations
                            e.g. {"A50G/L120M": {0: ["A50G"], 1: ["L20M"]}}

        Returns:
            pd.DataFrame: DataFrame with columns ['mut_string', 'chain_1', 'chain_2', ...]
        """
        # Create list of dictionaries for DataFrame
        rows = []
        for mut_string, chain_muts in mutation_map.items():
            row = {'Mut_string': mut_string}
            # Add chain mutations as comma-separated strings if multiple mutations exist
            for chain_idx, mutations in chain_muts.items():
                row[f'chain_{chain_idx + 1}'] = '/'.join(mutations) if mutations else ''
            rows.append(row)

        # Convert to DataFrame
        df = pd.DataFrame(rows)

        # Ensure consistent column ordering
        chain_cols = [col for col in df.columns if col.startswith('chain_')]
        df = df[['Mut_string'] + sorted(chain_cols)]

        return df

    if len(wt_files) > 1:

        mutations = top_df['Mut_string'].values.tolist()
        chain_lens = splits[0].wt_seq_lens
        dict_mutations = reverse_multichain_mutations(mutations, chain_lens)
        df_mutations = mutation_map_to_df(dict_mutations)

        top_df = pd.merge(top_df, df_mutations, on='Mut_string', how='left')
        top_df.to_csv(os.path.join(splits[0].file_attrs['dataset_dir'], 'proposers/results', f'{experiment_name}_proposals_top_{top_muts_per_load}.csv'), index=False)


        for col in df_mutations.columns[1:]:
            mutations = sorted({mutation for mutation in df_mutations[col].tolist() if mutation})
            # Convert mutations to a dataframe for CSV export.
            df_mutations_col = pd.DataFrame({str(col): mutations})
            df_mutations_col.to_csv(
                os.path.join(
                    splits[0].file_attrs['dataset_dir'],
                    f'{export_name}_{col}_mutants.csv',
                ),
                index=False,
                header=False,
            )

    manifest = {
        'schema_version': 2,
        'command': 'propose',
        'experiment_name': experiment_name,
        'seed': args.seed,
        'split_seed': args.split_seed,
        'deterministic': args.deterministic,
        'fold_count': args.ensemble_folds,
        'dataset_sha256': sha256_file(training_dataset_fname),
        'wt_fasta_sha256': [sha256_file(path) for path in wt_files],
        'mutation_pool_sha256': sha256_file(mutation_pool_fname),
        'min_mutations': min_mutations,
        'max_mutations': max_mutations,
        'top_muts_per_load': top_muts_per_load,
        'max_candidates': max_candidates,
        'candidate_count_by_load': candidate_counts,
        'selected_architecture': {
            'batch_size': bs,
            'learning_rate': lr,
            'layer_size': hidden,
            'num_layers': layers,
        },
        'model_seeds': model_seeds,
        'fold_assignment_sha256': sha256_json(
            splits[0].data['fold'].astype(int).tolist()
        ),
        'fold_scalers': [
            {
                'fold_index': fold_index,
                'split_name': fold.splits['split_name'],
                'validation_seed': stable_seed(
                    args.split_seed,
                    'validation',
                    f'kfold-{fold_index}',
                    -1,
                ),
                'data_min': fold.splits['target_scaler'].data_min_.tolist(),
                'data_max': fold.splits['target_scaler'].data_max_.tolist(),
            }
            for fold_index, fold in enumerate(splits)
        ],
        'model_artifact_sha256': [sha256_file(model.model_path) for model in models],
        'prediction_ensemble_size': len(models),
        'runtime': runtime_identity(actual_device),
    }
    manifest_path = os.path.join(
        splits[0].file_attrs['dataset_dir'],
        'proposers',
        'results',
        f'{experiment_name}_proposals_manifest.json',
    )
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')


if __name__ == "__main__":
    main()
