import itertools
from fuzzywuzzy import process, fuzz
import pandas as pd

def match_missing_fighters(list_a, list_b, match_threshold=0, show_score=True):
    """return best match from list_a to list_b"""

    mapping_dict = {}
    for fighter in list_a:
        match_info = process.extract(fighter, list_b)[0]#, scorer=fuzz.WRatio)[0] # WRatio is the default scorer
        if match_info[1] >= match_threshold:
            if show_score:
                mapping_dict[fighter] = match_info
            else:
                mapping_dict[fighter] = match_info[0]

    return mapping_dict

def overlap_stats(set_a: set, set_b: set) -> float:
    """For input sets a and b, calculate three metrics to compare b to a"""

    intersection_over_union = len(set_a.intersection(set_b)) / len(set_a.union(set_b))
    percent_of_a_in_b = len(set_a.intersection(set_b)) / len(set_a)
    items_missing_from_b = len(set_a - set_b)

    return intersection_over_union, percent_of_a_in_b, items_missing_from_b

def create_set_comparison_df(set_comparisons: dict) -> pd.DataFrame:
    """iterate over the set comparison (which is a list of k=2 permutation sets) and
    run the `overlap stats` function for a->b and b->a"""

    comparisons = list(itertools.permutations(
        list(set_comparisons.keys()), 2
    ))

    x = pd.DataFrame()
    for i, pair in enumerate(comparisons):
        a, b = pair

        set_a = set_comparisons[a]
        set_b = set_comparisons[b]

        x.loc[i, 'A'] = a
        x.loc[i, 'B'] = b

        x.loc[i, 'count_A'] = len(set_a)
        x.loc[i, 'count_B'] = len(set_b)

        x.loc[i, 'iou'], x.loc[i, 'perc_a_in_b'], x.loc[i, 'missing_from_b'] = overlap_stats(set_a, set_b)

        x.sort_values('missing_from_b', inplace=True)
        
    return x.reset_index(drop=True)

def match_missings(set_dict: dict, target_set: str) -> list:
    """Create a list of the union of what is in the list other sets but not set_a"""
    set_a = set_dict[target_set]
    other_sets = {k: v for k, v in set_dict.items() if k != target_set}
    ls = []
    for l in other_sets.values():
        ls += l - set_a

    union_of_missings = list(set().union(ls))
    print('There are ', len(union_of_missings), f' missing values in {target_set}')

    df_match = pd.DataFrame()

    for i, value in enumerate(union_of_missings):
        df_match.loc[i, 'target'] = value

        for set_name, set_ in set_dict.items():
            best_match = process.extractOne(value, list(set_), score_cutoff=80)
            if best_match: # because of cutoff sometime's it's null
                df_match.loc[i, set_name] = best_match[0]

    return df_match