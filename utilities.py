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
    """Calculates overlap metrics between two sets.

    Args:
        set_a: The first set to compare.
        set_b: The second set to compare.

    Returns:
        A tuple containing three metrics:
        1. **Intersection over Union (IoU):** The ratio of the intersection size to the union size.
        2. **Percent of A in B:** The percentage of elements in set A that are also in set B.
        3. **Items Missing from B:** The number of elements in set A that are not in set B.
    """

    intersection_over_union = len(set_a.intersection(set_b)) / len(set_a.union(set_b))
    percent_of_a_in_b = len(set_a.intersection(set_b)) / len(set_a)
    items_missing_from_b = len(set_a - set_b)

    return intersection_over_union, percent_of_a_in_b, items_missing_from_b

def create_set_comparison_df(set_comparisons: dict) -> pd.DataFrame:
    """Creates a DataFrame comparing pairwise overlaps between sets.

    Iterates over pairwise combinations of sets in the provided dictionary and calculates
    overlap metrics using the `overlap_stats` function. The DataFrame includes information
    about the sets being compared, their sizes, and the calculated overlap metrics.

    Args:
        set_comparisons: A dictionary where keys are set names and values are sets.

    Returns:
        A pandas DataFrame containing the following columns:
            - **A:** Name of the first set.
            - **B:** Name of the second set.
            - **count_A:** Number of elements in set A.
            - **count_B:** Number of elements in set B.
            - **iou:** Intersection over Union (IoU) of sets A and B.
            - **perc_a_in_b:** Percentage of elements in A that are also in B.
            - **missing_from_b:** Number of elements in A that are not in B.
    """

    comparisons = list(itertools.permutations(
        list(set_comparisons.keys()), 2
    ))

    df = pd.DataFrame()
    for i, pair in enumerate(comparisons):
        a, b = pair

        set_a = set_comparisons[a]
        set_b = set_comparisons[b]

        df.loc[i, 'A'] = a
        df.loc[i, 'B'] = b

        df.loc[i, 'count_A'] = len(set_a)
        df.loc[i, 'count_B'] = len(set_b)

        df.loc[i, 'iou'], df.loc[i, 'perc_a_in_b'], df.loc[i, 'missing_from_b'] = overlap_stats(set_a, set_b)

        df.sort_values('missing_from_b', inplace=True)
        
    return df.reset_index(drop=True)

def match_missings(set_dict: dict, target_set: str) -> list:
    """Identifies and matches missing values in a target set to other sets.

    This function:
    1. Identifies all elements in the target set that are missing from any other set.
    2. Attempts to find the best match for each missing element in the other sets, using Levenshtein
       distance and the weighted ratio scorer.
    3. Returns a DataFrame containing the missing values and their potential matches for each table.

    Args:
        set_dict: A dictionary where keys are set names and values are sets.
        target_set: The name of the target set to compare against. Must be a key of set_dict

    Returns:
        A pandas DataFrame with columns:
            - **target:** The missing value from the target set, as the index
            - variable other columns, one for each item in set_dict
        Values of the dataframe are the best match to the target value of that row.
        Missing values indicate no match was found above the threshold.
    """

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