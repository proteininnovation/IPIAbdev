import Levenshtein
import numpy as np
from Bio import Align
from Bio.Align import substitution_matrices
from scipy.stats import skewnorm
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Align import substitution_matrices
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from Bio import pairwise2
def calculate_blosum62_distance_same_len(seq1, seq2):
    """
    Calculate distance between two sequences using BLOSUM62 matrix
    Returns a distance score (lower means more similar)
    """
    # Get BLOSUM62 matrix from BioPython
    blosum62 = substitution_matrices.load('BLOSUM62') 
    
    # Ensure sequences are the same length
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of equal length")
    
    score = 0
    for aa1, aa2 in zip(seq1, seq2):
        # Get score from BLOSUM62 matrix, use reverse pair if not found
        pair = (aa1, aa2)
        if pair not in blosum62:
            pair = (aa2, aa1)
        score += blosum62.get(pair, -4)  # Default to -4 if pair not found
    
    # Convert similarity score to distance (higher score = more similar)
    max_score = len(seq1) * 4  # Maximum possible score (4 is max in BLOSUM62)
    distance = max_score - score
    return distance

def calculate_blosum62_distance(seq1, seq2, gap_open=-10, gap_extend=-1):
    """
    Calculate distance between two sequences of different lengths using pairwise alignment
    Returns a distance score (lower means more similar)
    """
    # Get BLOSUM62 matrix
    blosum62 = substitution_matrices.load('BLOSUM62') 
    
    # Perform global pairwise alignment with BLOSUM62
    alignments = pairwise2.align.globalds(seq1, seq2, blosum62, gap_open, gap_extend)
    
    # Take the best alignment score (first alignment)
    score = alignments[0].score
    
    # Estimate maximum possible score (using shortest sequence length)
    min_length = min(len(seq1), len(seq2))
    max_score = min_length * 4  # 4 is max score in BLOSUM62 for identical residues
    
    # Convert to distance (normalize between 0 and 1, then scale)
    distance = (max_score - score) / max_score
    return distance
def create_distance_matrix_blosum62(sequences):
    """
    Create a distance matrix for all sequence pairs
    """
    n = len(sequences)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = calculate_blosum62_distance(sequences[i], sequences[j])
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist  # Matrix is symmetric
    
    return dist_matrix

def create_clustering_blosum62(sequences, max_distance=None, num_clusters=None):
    """
    Perform hierarchical clustering and return cluster labels
    Either max_distance or num_clusters must be specified
    """
    if max_distance is None and num_clusters is None:
        raise ValueError("Must specify either max_distance or num_clusters")
    
    # Calculate distance matrix
    dist_matrix = create_distance_matrix_blosum62(sequences)
    
    # Perform hierarchical clustering
    linkage_matrix = hierarchy.linkage(dist_matrix, method='average')
    
    # Get cluster labels
    if max_distance is not None:
        # Cluster by maximum distance threshold
        cluster_labels = fcluster(linkage_matrix, max_distance, criterion='distance')
    else:
        # Cluster to get specified number of clusters
        cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    
    # Plot dendrogram
    #plt.figure(figsize=(10, 7))
    #hierarchy.dendrogram(
    #    linkage_matrix,
    #    labels=labels,
    #    leaf_rotation=90,
    #    leaf_font_size=8
    #)
    #plt.title("Hierarchical Clustering with BLOSUM62")
    #plt.xlabel("Sequences")
    #plt.ylabel("Distance")
    #if max_distance is not None:
    #    plt.axhline(y=max_distance, c='r', linestyle='--')
    #plt.tight_layout()
    #plt.show()
    return cluster_labels
    #return cluster_labels, linkage_matrix
# Example usage

def test_blosum():
    # Example protein sequences
    sequences = [
        "MAKPLTDQ",
        "MAKPLTEQ",
        "MVKPLTDQ",
        "LAKPLTDQ"
    ]
    labels = ["Seq1", "Seq2", "Seq3", "Seq4"]
    
    # Or read from a FASTA file
    # sequences = [str(record.seq) for record in SeqIO.parse("your_sequences.fasta", "fasta")]
    # labels = [record.id for record in SeqIO.parse("your_sequences.fasta", "fasta")]
    
    # Perform clustering
    linkage = create_clustering_blosum62(sequences,max_distance=2)




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
    aligner.mode = "global"
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
            if (len(sequences[i])<5):
                sequences[i]='CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC'
            if (len(sequences[j])<5):
                sequences[j]='CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC'
            score = aligner.score(sequences[i][2:-2], sequences[j][2:-2])
            matrix[i, j] = matrix[j, i] = score
    return matrix


def create_hcdr3_evalue_matrix(seqs: list[str]):
    matrix = create_similarity_matrix(seqs)

    # convert to distance matrix using expect values
    # params = [103.30608933,
    #           7.15378568,
    #           6.36972762]
    # params = [15.98019334,
    #           4.86832106,
    #           7.11563926]

    # global params Lib3.1 for seq[2:-2] positions
    params = [2.39163661, 
              -18.46987085,  
              12.44510516]
    e_matrix = 1 - skewnorm.cdf(matrix, *params)
    print(e_matrix)
    return e_matrix

def cluster_hcdr3(df: pd.DataFrame, evalue: float, cdr='cdr3_aa'):
    model = AgglomerativeClustering(n_clusters=None, 
                                    metric="precomputed",
                                    linkage='complete', 
                                    distance_threshold=evalue)
 
    
    e_matrix = create_hcdr3_evalue_matrix(df[cdr].to_list())
    model.fit_predict(e_matrix)
    cluster_counter = Counter(model.labels_).most_common()
    print(cluster_counter)
    cluster_n = cluster_counter[0][0]
    df["cluster"] = model.labels_
    print(f"# clusters: {len(cluster_counter)}")
    return df.copy()

def cluster_hcdr3_2(df: pd.DataFrame, evalue: float, cdr='cdr3_aa'):
    model = AgglomerativeClustering(n_clusters=None, 
                                    metric="precomputed",
                                    linkage='complete', 
                                    distance_threshold=evalue)
 
    e_matrix = create_hcdr3_evalue_matrix(df[cdr].to_list())
    model.fit_predict(e_matrix)
    cluster_counter = Counter(model.labels_).most_common()
    cluster_n = cluster_counter[0][0]
    df["cluster_"+cdr] = model.labels_
    print(f"# clusters: {len(cluster_counter)}")
    return df.copy()




def greedy_clustering_by_levenshtein(seq: list[str], cutoff: float = 0.85) -> list[int]:
    """
    Perform greedy clustering of sequences based on the Levenshtein distance.
    Sequences have to be sorted before calling this function.

    Args:
        seq (list[str]): A list of sequences to be clustered.
        cutoff (float, optional): The similarity cutoff value. Sequences with a Levenshtein distance
            similarity above this cutoff will be considered part of the same cluster. Defaults to 0.85.

    Returns:
        list[int]: A list of cluster labels, where each label corresponds to the cluster index of the
            corresponding sequence in the input list.
    """

    # initialize cluster counter
    n_seqs = len(seq)
    cluster = 0
    clusters = np.zeros(n_seqs, dtype=int)

    # iterate over sequences
    for i in range(n_seqs):
        # only cluster if not already assigned
        if clusters[i] == 0:
            cluster += 1
            clusters[i] = cluster
            for j in range(i + 1, n_seqs):
                if clusters[j] == 0:
                    if Levenshtein.ratio(seq[i], seq[j], score_cutoff=cutoff) > cutoff:
                        clusters[j] = cluster
    return clusters

