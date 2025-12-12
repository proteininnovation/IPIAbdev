import regex as re
import pandas as pd
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint as IP
from Bio.SeqUtils import molecular_weight
from Bio.SeqUtils.ProtParam import ProteinAnalysis
CRITICAL_LIABS = ['l_glyco', 
                   # 'l_aromatic', 
                   # 'l_poly', 
                   "l_RR",
                   "l_WW",
                   'l_cys',
                   'l_hydrophobic',
                   "l_neg_charge",
                   "l_pos_charge",
                   "l_charge_value2",
                   'l_arg',
                   'l_rmd',
                   "l_isoelectricpoint",
                   "l_molecular_weight",
                ]


def unpaired_cys(seq: str):
    if seq.count("C") % 2 == 1:
        return True
    elif re.search("C.C|CC", seq) is not None:
        return True
    else:
        return False
    
def simple_charge(seq: str):
    return (seq.count("R") + seq.count("K")) - (seq.count("D") + seq.count("E"))

def simple_charge2(seq: str):
    return (seq.count("R") + seq.count("K")+ seq.count("H")) - (seq.count("D") + seq.count("E"))

def IsoelectricPoint(seq: str):
    w=0
    try:
        protein=IP(seq)
        w=protein.pi(7.4)
    except:
        pass
    return w


def Molecular_Weight(seq: str):
    w=0
    seq=seq.split('_')[0]
    try:
        w=molecular_weight(seq,"protein")
    except:
        pass
    return w

def charge_value(seq:str):
    
    seq=seq.split('_')[0]
    charge=0
    try:
        analyzed_seq = ProteinAnalysis(seq)
        charge=analyzed_seq.charge_at_pH(7.4)
    except:
        pass
    return charge

def hydrophobicity_value(seq:str):
    
    seq=seq.split('_')[0]
    value=0
    try:
        analyzed_seq = ProteinAnalysis(seq)
        value=analyzed_seq.gravy()
    except:
        pass
    return value

def aromaticity_value(seq:str):
    
    seq=seq.split('_')[0]
    value=0
    try:
        analyzed_seq = ProteinAnalysis(seq)
        value=analyzed_seq.aromaticity()
    except:
        pass
    return value

def instability_value(seq:str):
    
    seq=seq.split('_')[0]
    value=0
    try:
        analyzed_seq = ProteinAnalysis(seq)
        value=analyzed_seq.instability_index()
    except:
        pass
    return value

def len_value(seq:str):
    
    seq=seq.split('_')[0]
    value=0
    try:
        analyzed_seq = ProteinAnalysis(seq)
        value=analyzed_seq.length
    except:
        pass
    return value

def charge_liab(seq: str, max_charge:repr(''), float = 2):
    return simple_charge(seq) > max_charge

def min_kmer_parker(seq, k):
    parker_scale = {
        "W": -10.0,
        "F": -9.2,
        "L": -9.2,
        "I": -8.0,
        "M": -4.2,
        "V": -3.7,
        "Y": -1.9,
        "C": 1.4,
        "A": 2.1,
        "P": 2.1,
        "H": 2.1,
        "R": 4.2,
        "T": 5.2,
        "K": 5.7,
        "G": 5.7,
        "Q": 6.0,
        "S": 6.5,
        "N": 7.0,
        "E": 7.8,
        "D": 10.0,
    }
    return min(parker_hplc_index(seq[i:i+k]) for i in range(0, len(seq) - k + 1))

def parker_hplc_index(seq: str):
  if len(seq)>0:
    parker_scale = {
        "W": -10.0,
        "F": -9.2,
        "L": -9.2,
        "I": -8.0,
        "M": -4.2,
        "V": -3.7,
        "Y": -1.9,
        "C": 1.4,
        "A": 2.1,
        "P": 2.1,
        "H": 2.1,
        "R": 4.2,
        "T": 5.2,
        "K": 5.7,
        "G": 5.7,
        "Q": 6.0,
        "S": 6.5,
        "N": 7.0,
        "E": 7.8,
        "D": 10.0,
    }
    return sum(parker_scale.get(i,0) for i in seq) / len(seq)
  else: 
    return 0

def hydro_liab(seq: str):
    return parker_hplc_index(seq) < -1

def count_aa_liab(seq, aa, max):
    return seq.count(aa) > max

def r_min_d(seq):
    return seq.count("R") - seq.count("D") > 1

def annotate_liabilities(df: pd.DataFrame, cdr3_col: str = "cdr3_aa"):

    # motif liabilities
    name_motif = {
        "l_glyco": re.compile("N[^P][ST]"),
        "l_asp_isomer": re.compile("D[GSD]"),
        "l_asn_deamid": re.compile("N[GSN]"),
        "l_aromatic": re.compile("[FWYH][FWYH][FWYH]"),
        "l_frag": re.compile("DP"),
        "l_poly": re.compile("RR|WW|VV"),
        "l_RR": re.compile("RR"),
        "l_WW": re.compile("WW"),
        "l_VV": re.compile("VV"),
        "l_neg_patch": re.compile("[DE][DE]|[DE].[DE]"),
    }
    # annotate motif liabilities
    for name, motif in name_motif.items():
        try:
            df[name] = df[cdr3_col].apply(lambda x: re.search(motif, x) is not None)
        except:
            pass
   

    # function liabilities
    name_func = {
        "l_cys": unpaired_cys,
        # "l_charge": charge_liab,
        "l_neg_charge": lambda x: simple_charge(x) < -4,
        "l_pos_charge": lambda x: simple_charge(x) > 2,
        "l_charge_value": simple_charge,
        "l_charge_value2": simple_charge2,
        "l_hydrophobic": lambda x: parker_hplc_index(x) < -1,
        "l_hydrophobic_value": parker_hplc_index,
        "l_arg": lambda x: count_aa_liab(x,"R", 2) ,
        "l_trp": lambda x: count_aa_liab(x,"W", 2) ,
        "l_rmd": r_min_d ,
        "l_isoelectricpoint": IsoelectricPoint,
        "l_molecular_weight": Molecular_Weight,
    }



    # annotate function liabilities
    for name, func in name_func.items():
        df[name] = df[cdr3_col].apply(func)

    liabs = [l for l in df.columns if "l_" == l[:2]]
    # for l in liabs:
    #     print(l, df[df[l]]["freq"].sum())

    return df.copy()


def annotate_liabilities_2(df: pd.DataFrame, cdr3_col='cdr3_aa', label=''):

    # motif liabilities
    if (label==''):
        label=cdr3_col
        print(label)

    name_motif = {
        label+"_glyco": re.compile("N[^P][ST]"),
        label+"_asp_isomer": re.compile("D[GSD]"),
        label+"_asn_deamid": re.compile("N[GSN]"),
        label+"_aromatic": re.compile("[FWYH][FWYH][FWYH]"),
        label+"_frag": re.compile("DP"),
        label+"_poly": re.compile("RR|WW|VV"),
        label+"_RR": re.compile("RR"),
        label+"_WW": re.compile("WW"),
        label+"_VV": re.compile("VV"),
        label+"_neg_patch": re.compile("[DE][DE]|[DE].[DE]"),
    }
    # annotate motif liabilities
    for name, motif in name_motif.items():
        df[name] = df[cdr3_col].apply(lambda x: re.search(motif, x) is not None)

    # function liabilities
    name_func = {
        label+"_cys": unpaired_cys,
        # "l_charge": charge_liab,
        label+"_neg_charge": lambda x: simple_charge(x) < -4,
        label+"_pos_charge": lambda x: simple_charge(x) > 2,
        label+"_simple_charge": simple_charge,
        label+"_parker_hplc": lambda x: parker_hplc_index(x) < -1,
        label+"_parker_hplc_index": parker_hplc_index,
        label+"_arg": lambda x: count_aa_liab(x,"R", 2) ,
        label+"_trp": lambda x: count_aa_liab(x,"W", 2) ,
        label+"_rmd": r_min_d ,
        label+"_isoelectricpoint": IsoelectricPoint,
        label+"_charge":charge_value,
        label+"_instability_index":instability_value,
        label+"_aromaticity":aromaticity_value,
        label+"_hydrophobicity":hydrophobicity_value,
        label+"_length":len_value
    }

    # annotate function liabilities
    for name, func in name_func.items():
        df[name] = df[cdr3_col].apply(func)

    liabs = [l for l in df.columns if "l_" == l[:2]]
    # for l in liabs:
    #     print(l, df[df[l]]["freq"].sum())

    return df.copy()

