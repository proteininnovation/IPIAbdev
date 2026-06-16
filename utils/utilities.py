# load DNA to counter and then dataframe
import sys
import os
import json
import gzip as gz
import pandas as pd
import subprocess
import regex as re
import numpy as np
from collections import Counter
from Bio.SeqIO.QualityIO import FastqGeneralIterator
from Bio.Seq import Seq
from Bio import Align
from Bio.Align import substitution_matrices
from time import asctime
from pprint import pprint
from scipy.stats import skewnorm
from sklearn.cluster import AgglomerativeClustering
from random import shuffle


def merge_fastq_pair(name, fastq1, fastq2,FASTP):
    # determine output file path
    folder = '/'.join(fastq1.split("/")[:-1])
    out_fastq = folder + "/" + name + "_merged.fastq.gz"
    json_out = folder + "/" + name + "_merged.json"
    # build command line
    cmd = (
          f"{FASTP}"
          f" --in1 '{fastq1}'" # fastq1
          f" --in2 '{fastq2}'" # fastq2
          f" --merge --merged_out '{out_fastq}'" # merge
          f" --json '{json_out}'" # json
          " --qualified_quality_phred 25 --unqualified_percent_limit 20 --length_required 50"
          " --n_base_limit 5 --correction --overlap_len_require 50 --overlap_diff_limit 5 --thread 8"
          " --disable_adapter_trimming --disable_trim_poly_g")
    # execute
    process = subprocess.Popen(cmd,
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, 
                      shell=True)
    stdout, stderr = process.communicate()
    print(stdout)
    print(stderr)
    # Process some read stats
    data = json.load(open(json_out))
    read_count = {}
    read_count["total"] = data["read1_before_filtering"]["total_reads"]
    read_count["merged"] = data["merged_and_filtered"]["total_reads"]
    read_count["low_quality"] = int(data["filtering_result"]["low_quality_reads"]/2)
    read_count["too_many_N"] = int(data["filtering_result"]["too_many_N_reads"]/2)
    read_count["too_short"] = int(data["filtering_result"]["too_short_reads"]/2)
    read_count["too_long"] = int(data["filtering_result"]["too_long_reads"]/2)
    pprint(read_count)
    return read_count, out_fastq


def load_fastq_to_df(fastq_path: str):
    # open Fastq and store sequences in a counter
    seqs = Counter()
    with gz.open(fastq_path, "rt") as fin:
        for (title, sequence, quality) in FastqGeneralIterator(fin):
            seqs[str(Seq(sequence).reverse_complement())] += 1

    print(asctime(), f"# reads: {sum(seqs.values())}")
    print(asctime(), f"# unique DNA: {len(seqs)}")

    # convert to dataframe
    df = pd.DataFrame(data={"nt": seqs.keys(), "count": seqs.values()})
    # print(df)
    del seqs
    return df.copy()

# find h3

def split_at_delimiters(seq: str, delimiters: list[str], split_downstream=True):
    """Find cut point in a sequence using a list of possible delimiters

    Args:
        seq (str): string to be searched
        delimiters (list[str]): list of substrings to find
        split_downstream (bool, optional): return the end of the match intead of begining. Defaults to True.

    Returns:
        _type_: index of cut point
    """
    for delim in delimiters:
        idx = seq.find(delim)
        if idx != -1:
            if split_downstream:
                return idx + len(delim)
            else:
                return idx
    return -1

def find_hcdr1(df: pd.DataFrame, upseq: str, downseq: str, read_count: dict[str,int]):
    # find upstream cut position
    df.loc[:,"cdr3_beg"] = df["nt"].apply(lambda x: split_at_delimiters(x, upseq, split_downstream=True))

    # find downstream cut position
    df.loc[:, "cdr3_end"] = df["nt"].apply(lambda x: split_at_delimiters(x, downseq, split_downstream=False))

    # drop reads where HCDR3 wasn't found
    n_reads = df["count"].sum()
    n_seqs = len(df)
    df = df.loc[(df["cdr3_beg"] != -1) & (df["cdr3_end"] != -1), :].copy()
    read_count["reads_no_cdr3_edges"] = n_reads - df["count"].sum()
    read_count["seqs_no_cdr3_edges"] = n_seqs - len(df)

    # extract CDR3 sequence
    df.loc[:, "cdr3_nt"] = df[["nt", "cdr3_beg", "cdr3_end"]].apply(lambda r: r["nt"][r["cdr3_beg"]:r["cdr3_end"]] , axis=1)
    df["cdr3_nt"] = df[["nt", "cdr3_beg", "cdr3_end"]].apply(lambda r: r["nt"][r["cdr3_beg"]:r["cdr3_end"]] , axis=1)

    # determine if length is multiple of 3
    df.loc[:, "cdr3_mod3"] = df["cdr3_nt"].apply(lambda x: len(x) % 3)

    # translate, get aa length, check functionality
    df.loc[:, "cdr3_aa"] = df["cdr3_nt"].apply(lambda x: str(Seq(x).translate()) )
    df.loc[:, "cdr3_aa"].fillna("", inplace=True)
    df.loc[:, "cdr3_aa_len"] = df["cdr3_aa"].apply(len)
    df.loc[:, "cdr3_functional"] = df[["cdr3_mod3", "cdr3_aa"]].apply(lambda r: "*" not in r["cdr3_aa"] and r["cdr3_mod3"] == 0, axis=1)
    df.sort_values(by="count", ascending=False, inplace=True)
    return df.copy(), read_count


def find_hcdr3(df: pd.DataFrame, upseq: str, downseq: str, read_count: dict[str,int]):
    # find upstream cut position
    df.loc[:,"cdr3_beg"] = df["nt"].apply(lambda x: split_at_delimiters(x, upseq, split_downstream=True))

    # find downstream cut position
    df.loc[:, "cdr3_end"] = df["nt"].apply(lambda x: split_at_delimiters(x, downseq, split_downstream=False))

    # drop reads where HCDR3 wasn't found
    n_reads = df["count"].sum()
    n_seqs = len(df)
    df = df.loc[(df["cdr3_beg"] != -1) & (df["cdr3_end"] != -1), :].copy()
    read_count["reads_no_cdr3_edges"] = n_reads - df["count"].sum()
    read_count["seqs_no_cdr3_edges"] = n_seqs - len(df)

    # extract CDR3 sequence
    df.loc[:, "cdr3_nt"] = df[["nt", "cdr3_beg", "cdr3_end"]].apply(lambda r: r["nt"][r["cdr3_beg"]:r["cdr3_end"]] , axis=1)
    df["cdr3_nt"] = df[["nt", "cdr3_beg", "cdr3_end"]].apply(lambda r: r["nt"][r["cdr3_beg"]:r["cdr3_end"]] , axis=1)

    # determine if length is multiple of 3
    df.loc[:, "cdr3_mod3"] = df["cdr3_nt"].apply(lambda x: len(x) % 3)

    # translate, get aa length, check functionality
    df.loc[:, "cdr3_aa"] = df["cdr3_nt"].apply(lambda x: str(Seq(x).translate()) )
    df.loc[:, "cdr3_aa"].fillna("", inplace=True)
    df.loc[:, "cdr3_aa_len"] = df["cdr3_aa"].apply(len)
    df.loc[:, "cdr3_functional"] = df[["cdr3_mod3", "cdr3_aa"]].apply(lambda r: "*" not in r["cdr3_aa"] and r["cdr3_mod3"] == 0, axis=1)
    df.sort_values(by="count", ascending=False, inplace=True)
    return df.copy(), read_count

def get_label_from_barcode(seq: str, barcodes: dict[str,str], errors_allowed=1):
    if errors_allowed > 0:
        for label, barcode in barcodes.items():
            x = re.search(f"({barcode}){{e<={errors_allowed}}}", seq)
            if x is not None:
                return label
    if errors_allowed == 0:
        for label, barcode in barcodes.items():
            x = seq.find(barcode)
            if x != -1:
                return label
    return "UNK"

def find_vh_vl(df: pd.DataFrame, vh_barcodes: list[str], vl_barcodes: list[str] = None, vh_errors: int = 1, vl_errors: int = 1):
    df.loc[:, "vh_scaffold"] = df["nt"].apply(
        lambda x: get_label_from_barcode(x[VH_BARCODE_REGION[0]:VH_BARCODE_REGION[1]], 
                                         VH_BARCODES, 
                                         errors_allowed=vh_errors))
    if vl_barcodes is not None:
        df.loc[:, "vl_scaffold"] = df["nt"].apply(
            lambda x: get_label_from_barcode(x[VL_BARCODE_REGION[0]:VL_BARCODE_REGION[1]], 
                                             VL_BARCODES, 
                                             errors_allowed=vl_errors))
    return df.copy()


def consolidate(df: pd.DataFrame):
    df.drop(columns=["nt", "cdr3_beg", "cdr3_end", "cdr3_nt", "cdr3_mod3"], inplace=True)
    df = df.groupby(["cdr3_aa", "cdr3_functional", "vh_scaffold", "vl_scaffold"]).agg({"count": "sum", "cdr3_aa_len": "first"}).sort_values("count", ascending=False).reset_index()
    df["rank"] = range(1, len(df)+1)
    df["freq"] = df["count"] / df["count"].sum()
    return df.copy()

def remove_crap(df: pd.DataFrame, 
                read_count: dict[str,int],
                min_h3_len: int = 1,
                max_h3_len: int = 30,
                functional_only: bool = True,
                keep_vh_unk: bool = True,
                keep_vl_unk: bool = True,
                min_freq: float = 0,
                min_count: int = 1,
                ):

        n_reads = df["count"].sum()
        n_seqs = len(df)

        # drop by length
        df = df.loc[(df["cdr3_aa_len"] >= min_h3_len) &
                    (df["cdr3_aa_len"] <= max_h3_len)
                    , :]
        read_count["reads_with_short_cdr3"] = n_reads - df["count"].sum()
        read_count["seqs_with_short_cdr3"] = n_seqs - len(df)
        n_reads = df["count"].sum()
        n_seqs = len(df)

        # drop non-functional
        if functional_only:
                df = df.loc[df["cdr3_functional"], :]
                read_count["reads_with_nonfunctional_cdr3"] = n_reads - df["count"].sum()
                read_count["seqs_with_nonfunctional_cdr3"] = n_seqs - len(df)
                n_reads = df["count"].sum()
                n_seqs = len(df)

        # drop low frequency
        df = df.loc[(df["freq"] >= min_freq) &
                    (df["count"] >= min_count)
                , :]
        read_count["reads_with_low_frequency"] = n_reads - df["count"].sum()
        read_count["seqs_with_low_frequency"] = n_seqs - len(df)
        n_reads = df["count"].sum()
        n_seqs = len(df)

        # drop sequences with unknown scaffolds 
        if not keep_vh_unk:
            df = df[(df["vh_scaffold"] != "UNK")]
            read_count["reads_with_unk_vh"] = n_reads - df["count"].sum()
            read_count["seqs_with_unk_vh"] = n_seqs - len(df)
            n_reads = df["count"].sum()
            n_seqs = len(df)

        if not keep_vl_unk:
            df = df[(df["vl_scaffold"] != "UNK")]
            read_count["reads_with_unk_vl"] = n_reads - df["count"].sum()
            read_count["seqs_with_unk_vl"] = n_seqs - len(df)
            n_reads = df["count"].sum()
            n_seqs = len(df)

        # drop ambiguous
        df = df[~df["cdr3_aa"].str.contains("X")]
        read_count["reads_ambiguous"] = n_reads - df["count"].sum()
        read_count["seqs_ambiguous"] = n_seqs - len(df)
        n_reads = df["count"].sum()
        n_seqs = len(df)

        df.sort_values("count", ascending=False, inplace=True)
        df["rank"] = range(1, len(df)+1)
        return df.copy(), read_count

# Replace VL1-51 for VK4-1 for the VH1-69 clones
def fix_vl1_51_to_vk4_1(df: pd.DataFrame):
    # df["vl_scaffold"] = df[["vh_scaffold", "vl_scaffold",]].apply(
    #     lambda r: "VK4-1" if ((r["vh_scaffold"] == "VH1-69" or r["vh_scaffold"] == "VH3-23") 
    #                           and r["vl_scaffold"] == "VL1-51") else r["vl_scaffold"] ,
    #     axis=1)
    df["vl_scaffold"] = df[["vh_scaffold", "vl_scaffold",]].apply(
        lambda r: "K4-1_C" if (r["vh_scaffold"] == "H1-69" and 
                               r["vl_scaffold"] == "L1-51") else r["vl_scaffold"] ,
        axis=1)
    # df["vl_scaffold"] = df[["vh_scaffold", "vl_scaffold",]].apply(
    #     lambda r: "K4-1_C" if ((r["vh_scaffold"] == "H1-69" or r["vh_scaffold"] == "H3-23" or r["vh_scaffold"] == "H3-23_A")
    #                            and 
    #                            r["vl_scaffold"] == "L1-51") else r["vl_scaffold"] ,
    #     axis=1)
    return df.copy()



def create_similarity_matrix(sequences):
    """
    Create a similarity pairwise matrix using local alignment with a PAM30 substitution matrix,
    including similarity calculation against the sequence itself.

    Parameters:
        sequences (list): List of sequences.

    Returns:
        numpy.ndarray: Similarity pairwise matrix.
    """
    # initialize alignment tool
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.substitution_matrix = substitution_matrices.load("PAM30")
    aligner.open_gap_score = -9
    aligner.extend_gap_score = -1

    # create matrix
    n_sequences = len(sequences)
    matrix = np.zeros((n_sequences, n_sequences))
    for i in range(n_sequences):
        if i % 100 == 0:
            print(f"# sequences processed {i}")
        for j in range(i, n_sequences):
            score = aligner.score(sequences[i][2:-3], sequences[j][2:-3])
            matrix[i, j] = matrix[j, i] = score
    return matrix

def create_hcdr3_evalue_matrix(seqs: list[str]):
    matrix = create_similarity_matrix(seqs)

    # convert to distance matrix using expect values
    params = [103.30608933,
              7.15378568,
              6.36972762]
    params = [15.98019334,
              4.86832106,
              7.11563926]
    e_matrix = 1 - skewnorm.cdf(matrix, *params)
    # print(e_matrix)
    return e_matrix



def cluster_hcdr3(df: pd.DataFrame, evalue: float, cdr3='cdr3_aa'):
    model = AgglomerativeClustering(n_clusters=None, 
                                    metric="precomputed",
                                    linkage='complete', 
                                    distance_threshold=evalue)
    e_matrix = create_hcdr3_evalue_matrix(df[cdr3].to_list())
    model.fit_predict(e_matrix)
    cluster_counter = Counter(model.labels_).most_common()
    # print(cluster_counter)
    cluster_n = cluster_counter[0][0]
    df["cluster"] = model.labels_
    print(f"# clusters: {len(cluster_counter)}")
    return df.copy()


def consolitate_by_cluster(df: pd.DataFrame):

    # define action for each col
    col_action = {
        'cdr3_aa': ["first"], 
        'vh_scaffold': ["first"],
        'vl_scaffold': ["first"], 
        'count': ["sum", "first"],
        'freq': ["sum", "first"],
        'aux': ["sum"],
        'rank': ["first"],
        }

    # get liability cols if any
    liab_cols = [c for c in df.columns if c[:2] == "l_"]
    for c in liab_cols:
        col_action[c] = "first"

    # group by cluster
    df["aux"] = 1
    dfc = df.groupby("cluster").agg(col_action)
    dfc.columns = ['_'.join(col).strip() for col in dfc.columns.values]
    dfc.rename(columns={"aux_sum": "n_seqs"}, inplace=True)
    dfc = dfc.reset_index()
    dfc.sort_values("freq_sum", ascending=False, inplace=True)
    dfc["rank_cluster"] = range(1,len(dfc)+1)
    return dfc

def shannon_diversity(data):
    """ Given a hash { 'cdr': count } , returns the shannon_diversity

    >>> sdi({'a': 10, 'b': 20, 'c': 30,})
    1.0114042647073518"""

    from math import log as ln

    def p(n, N):
        """ Relative abundance """
        if n is  0:
            return 0
        else:
            return (float(n)/N) * ln(float(n)/N)

    N = sum(data)

    return -sum(p(n, N) for n in data if n is not 0)

def evenness(H,S):
    from math import log as ln
    return H/ln(S)

def simpsons_diversity_index(counts):
    """Calculates Simpson's Diversity Index for a given list of species counts.

    Args:
        counts: A list of integers representing the number of individuals 
                of each species.

    Returns:
        The Simpson's Diversity Index.
    """

    N = sum(counts)
    sum_of_squares = sum(n * (n - 1) for n in counts)

    if N <= 1:
        return 0  # No diversity if only one individual or no individuals
    else:
        return 1 - (sum_of_squares / (N * (N - 1)))

    # Example usage
    #species_counts = [10, 5, 2, 8, 15]
    #diversity = simpsons_diversity_index(species_counts)
    #print("Simpson's Diversity Index:", diversity)

def inverse_simpson_index(counts):
    """
    Calculates the Inverse Simpson's Diversity Index for a given list of counts.    

    Args:
        counts: A list of counts representing the abundance of each species.

    Returns:
        The Inverse Simpson's Diversity Index.
    """

    total_count = sum(counts)
    simpson_index = sum((count / total_count)**2 for count in counts)
    return 1 / simpson_index


def generate_kmers(sequence, k):
    """Generates k-mers from a protein sequence.

    Args:
        sequence (str): The protein sequence.
        k (int): The length of the k-mers.

    Returns:
        list: A list of k-mers.
    """

    kmers = []
    if (len(sequence)>=k):
        for i in range(len(sequence) - k + 1):
            kmers.append(sequence[i:i + k])
    return kmers
