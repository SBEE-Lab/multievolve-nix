# This utils page is for functions that don't belong to any particular category

from typing import List, Tuple
from Bio import SeqIO
import errno
import numpy as np
import os
import scipy.stats as ss
from scipy.spatial.distance import cdist
import logging
import string
import re

# List of amino acids, including stop codon '*'
AAs = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
    '*'
]

aa_dict_3to1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

def load_seqs_file(fnames):
    """
    Load sequences and their corresponding values from files.
    
    Args:
    - fnames (list): List of file names to load from.
    
    Returns:
    - tuple: Lists of sequences and their corresponding values.
    """
    seqs, y = [], []
    for fname in fnames:
        with open(fname) as f:
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) == 0:
                    raise ValueError(f'Input file {fname} has no columns.')
                elif len(fields) == 1:
                    seqs.append(fields[0])
                    y.append(float('nan'))
                elif len(fields) == 2:
                    seqs.append(fields[0])
                    y.append(float(fields[1]))
                else:
                    raise ValueError(f'Input file {fname} has more than two columns.')

                if not seqs[-1].startswith('M'):
                    print(seqs[-1], y[-1])
    return seqs, y

def performance_report(y_true, y_pred):
    """Return metrics in original property units.

    NDCG requires nonnegative relevance labels, so signed properties are shifted
    by their minimum for that metric only. The shift preserves relevance order;
    MSE, correlations, and top-ranked property summaries use the unmodified
    values.
    """
    from sklearn.metrics import mean_squared_error, ndcg_score

    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if not len(y_true) or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be non-empty and have equal length")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("y_true and y_pred must contain only finite values")

    ranked_true = y_true[np.argsort(-y_pred, kind="stable")]
    top_10 = ranked_true[:10]
    top_1_percentile = ranked_true[min(int(len(ranked_true) * 0.01), len(ranked_true) - 1)]
    top_0_1_percentile = ranked_true[
        min(int(len(ranked_true) * 0.001), len(ranked_true) - 1)
    ]

    if len(y_true) < 2 or np.ptp(y_true) == 0 or np.ptp(y_pred) == 0:
        spearman_r = spearman_p = pearson_r = pearson_p = float("nan")
    else:
        spearman = ss.spearmanr(y_true, y_pred)
        pearson = ss.pearsonr(y_true, y_pred)
        spearman_r, spearman_p = spearman.correlation, spearman.pvalue
        pearson_r, pearson_p = pearson.correlation, pearson.pvalue

    ndcg_relevance = y_true - min(float(y_true.min()), 0.0)
    ndcg = (
        float("nan")
        if len(y_true) < 2
        else ndcg_score([ndcg_relevance], [y_pred])
    )
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'Spearman r': spearman_r,
        'Spearman p': spearman_p,
        'Pearson r': pearson_r,
        'Pearson p': pearson_p,
        'NDCG': ndcg,
        'Top 10 Min': top_10.min(),
        'Top 10 Mean': top_10.mean(),
        'Top 10 Max': top_10.max(),
        'Top 1% Min': top_1_percentile,
        'Top 1% Mean': top_1_percentile,
        'Top 1% Max': top_1_percentile,
        'Top 0.1% Min': top_0_1_percentile,
        'Top 0.1% Mean': top_0_1_percentile,
        'Top 0.1% Max': top_0_1_percentile,
    }

def log_results(stats_dict, model_object):
    """No-op: metrics are recorded by the local sweep backend (utils.local_sweep)
    from each run's return value, so per-run W&B logging is not needed."""
    return

def mkdir_p(path):
    """
    Create a directory if it doesn't exist.
    
    Args:
    - path (str): Directory path to create.
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def deep_mutational_scan(sequence, exclude_noop=True):
    """
    Generate all possible single amino acid mutations for a given sequence.
    
    Args:
    - sequence (str): Input protein sequence.
    - exclude_noop (bool): If True, exclude mutations that don't change the amino acid.
    
    Yields:
    - tuple: (position, wild-type amino acid, mutant amino acid)
    """
    for pos, wt in enumerate(sequence):
        for mt in AAs:
            if exclude_noop and wt == mt:
                continue
            yield (pos, wt, mt)

def deep_mutational_scan_seqs(sequence, exclude_stop=True):
    """
    Generate all possible single amino acid mutant sequences for a given sequence.
    
    Args:
    - sequence (str): Input protein sequence.
    - exclude_stop (bool): If True, exclude mutations to stop codons.
    
    Returns:
    - list: List of mutant sequences.
    """
    sequences = []
    for pos, wt in enumerate(sequence):
        for mt in AAs:
            if wt == mt:
                continue
            if exclude_stop and mt == '*':
                continue
            mut_seq = sequence[:pos] + mt + sequence[pos + 1:]
            sequences.append(mut_seq)
    return sequences

def dms_dict(sequence):
    """
    Create a dictionary of all possible single amino acid mutations for each position in a sequence.
    
    Args:
    - sequence (str): Input protein sequence.
    
    Returns:
    - dict: Dictionary with positions as keys and lists of possible mutations as values.
    """
    dms_dict = {}
    for pos, wt in enumerate(sequence):
        pos_mts = []
        for mt in AAs:
            if mt != '*':
                pos_mts.append(wt + str(pos + 1) + mt)
        dms_dict[pos] = pos_mts
    return dms_dict

def mutational_pool_to_dict(mutational_pool, increase_wt=False):
    """
    Convert a list of mutations to a dictionary grouped by position.
    
    Args:
    - mutational_pool (list): List of mutations.
    - increase_wt (bool): If True, increase the count of wild-type mutations.
    
    Returns:
    - dict: Dictionary with positions as keys and lists of mutations as values.
    """
    mutations_dict = {}

    for mutation in mutational_pool:
        number = int(re.search(r'\d+', mutation).group())
        
        if number not in mutations_dict:
            mutations_dict[number] = [mutation]
        else:
            mutations_dict[number].append(mutation)
    
    for key in mutations_dict:
        mut = mutations_dict[key][0]
        wt, pos = mut[0], mut[1:-1]
        wt_value = wt+pos+wt
        
        if wt_value not in mutations_dict[key]:
            mutations_dict[key].append(wt_value)

        if increase_wt:
            total_mutants = len(mutations_dict[key]) - 2
            for i in range(total_mutants):
                mutations_dict[key].append(wt_value)

    return mutations_dict

def wt_only_mutational_pool_to_dict(mutational_pool, wt_seq):
    """
    Create a dictionary of wild-type mutations for each position in a sequence.
    
    Args:
    - mutational_pool (list): List of mutations.
    - wt_seq (str): Wild-type full sequence.
    
    Returns:
    - dict: Dictionary with positions as keys and lists of wild-type mutations as values.
    """
    wt_mutations_dict = {}

    for mutation in mutational_pool:
        number = int(re.search(r'\d+', mutation).group())
        
        if number not in wt_mutations_dict:
            wt = wt_seq[number-1]
            wt_value = wt+str(number)+wt
            wt_mutations_dict[number] = [wt_value]

    return wt_mutations_dict

def mut_pool_searcher(keys, dict):
    """
    Search for mutations in a dictionary based on given keys.
    
    Args:
    - keys (list): List of keys to search for.
    - dict (dict): Dictionary to search in.
    
    Returns:
    - list: List of mutations found for the given keys.
    """
    muts = []
    for key in keys:
        muts.extend(dict[key])
    return muts

def eAUC(y_true, y_pred):
    """
    Calculate the enrichment Area Under the Curve (eAUC).
    
    Args:
    - y_true (array-like): True binary labels.
    - y_pred (array-like): Predicted scores.
    
    Returns:
    - float: eAUC score.
    """
    from sklearn.metrics import auc
    ranked = ss.rankdata(-y_pred)[y_true == 1.]
    n_true = np.array([
        sum(ranked <= i + 1) for i in range(len(y_true))
    ])
    n_consider = np.array([ i + 1 for i in range(len(y_true)) ])
    norm = max(n_consider) * max(n_true)
    return auc(n_consider, n_true) / norm    


def setup_logger(log_file):
    """
    Set up a logger for both file and stream logging.
    
    Args:
    - log_file (str): Path to the log file.
    
    Returns:
    - logging.Logger: Configured logger object.
    """
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))

    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    logger.info('Logger set up.')
    return logger

def close_logger(logger):
    """
    Close all handlers associated with the given logger.
    
    Args:
    - logger (logging.Logger): Logger object to close.
    """
    logger.info('Closing logger.')
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

# MSA utils

# This is an efficient way to delete lowercase characters and insertion characters from a string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

def read_sequence(filename: str) -> Tuple[str, str]:
    """
    Read the first (reference) sequence from a fasta or MSA file.
    
    Args:
    - filename (str): Path to the fasta or MSA file.
    
    Returns:
    - tuple: (sequence description, sequence)
    """
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence: str) -> str:
    """
    Remove any insertions from the sequence. Needed to load aligned sequences in an MSA.
    
    Args:
    - sequence (str): Input sequence.
    
    Returns:
    - str: Sequence with insertions removed.
    """
    return sequence.translate(translation)

def read_msa(filename: str) -> List[Tuple[str, str]]:
    """
    Read sequences from an MSA file, automatically removing insertions.
    
    Args:
    - filename (str): Path to the MSA file.
    
    Returns:
    - list: List of tuples (sequence description, sequence without insertions)
    """
    return [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(filename, "fasta")]

def greedy_select(msa: List[Tuple[str, str]], num_seqs: int, mode: str = "max") -> List[Tuple[str, str]]:
    """
    Select sequences from the MSA to maximize or minimize the hamming distance.
    
    Args:
    - msa (list): List of (description, sequence) tuples.
    - num_seqs (int): Number of sequences to select.
    - mode (str): 'max' to maximize distance, 'min' to minimize.
    
    Returns:
    - list: Selected sequences.
    """
    assert mode in ("max", "min")
    if len(msa) <= num_seqs:
        return msa
    
    array = np.array([list(seq) for _, seq in msa], dtype=np.bytes_).view(np.uint8)

    optfunc = np.argmax if mode == "max" else np.argmin
    all_indices = np.arange(len(msa))
    indices = [0]
    pairwise_distances = np.zeros((0, len(msa)))
    for _ in range(num_seqs - 1):
        dist = cdist(array[indices[-1:]], array, "hamming")
        pairwise_distances = np.concatenate([pairwise_distances, dist])
        shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
        shifted_index = optfunc(shifted_distance)
        index = np.delete(all_indices, indices)[shifted_index]
        indices.append(index)
    indices = sorted(indices)
    return [msa[idx] for idx in indices]

def msa_splicer(msa):
    """
    Splice the MSA to only include positions where the first sequence is not empty.
    
    Args:
    - msa (list): List of (description, sequence) tuples.
    
    Returns:
    - list: Spliced MSA.
    """
    base_sequence = msa[0][1]
    positions = [i for i, _ in enumerate(base_sequence) if base_sequence[i] != '-']

    spliced_msa = []
    for name, seq in msa:
        spliced_seq = [seq[i] for i in positions]
        spliced_msa.append((name, ''.join(spliced_seq)))
    
    return spliced_msa