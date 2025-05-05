from random import random
from joblib import Parallel, delayed
from .hierarchical_grouping import HierarchicalGrouping
from .utils import segment_iq_data#, load_strategy

def selective_search_one(iq_data, k, sim_strategy, thr_sim=0.3, max_iter=40):
    '''
    Selective Search using single diversification strategy for I/Q data
    Parameters
    ----------
        iq_data : ndarray
            Original I/Q data
        k : int
            Number of segments
        sim_stategy : string
            Combinations of similarity measures

    Returns
    -------
        segments : list
            Segments of the data
        priority: list
            Small priority number indicates higher position in the hierarchy
    '''

    # Generate starting locations
    iq_seg = segment_iq_data(iq_data, k)

    # Initialize hierarchical grouping
    S = HierarchicalGrouping(iq_data, iq_seg, sim_strategy, win_size=k)

    S.build_regions()
    S.build_region_pairs()

    max_sim = max(list(S.s.values()))

    iter = 0
    # Start hierarchical grouping
    while not S.is_empty():
        if(max(list(S.s.values())) <= max_sim * thr_sim):
            break
        if(iter>=max_iter):
            break

        i, j = S.get_highest_similarity()

        S.merge_region(i, j)

        S.remove_similarities(i, j)

        S.calculate_similarity_for_new_region()

        iter+=1

    # convert the order by hierarchical priority
    segments = [x['box'] for x in S.regions.values()][::-1]

    # drop duplicates by maintaining order
    segments = list(dict.fromkeys(segments))

    # generate priority for segments
    priorities = list(range(1, len(segments) + 1))

    return segments, priorities

def selective_search(iq_data, load_strategy, mode='normal', random_sort=False):
    """
    Selective Search in Python for I/Q data
    """

    # load selective search strategy
    strategy, thrd = load_strategy(mode)

    tasks = []
    # vault = []
    for k, sim in strategy:
        thr_sim = thrd[sim][0]
        max_iter = thrd[sim][1]
        task = delayed(selective_search_one)(iq_data, k, sim, thr_sim=thr_sim, max_iter=max_iter)
        tasks.append(task)
        # vault.append( selective_search_one(iq_data, k, sim, thr_sim=thr_sim, max_iter=max_iter))

    vault = Parallel(n_jobs=4, backend='threading')(tasks)

    return vault

def segment_filter(segments, min_size=20, max_ratio=None, topN=None):
    proposal = []

    for segment in segments:
        # Debug: print the segment structure
        # print("Debug: segment =", segment)

        start, end = segment
        length = end - start

        # Filter for size
        if length < min_size:
            continue

        proposal.append(segment)

    if topN:
        if topN <= len(proposal):
            return proposal[:topN]
        else:
            return proposal
    else:
        return proposal
