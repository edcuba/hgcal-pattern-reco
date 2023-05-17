import numpy as np
import awkward as ak
from .data import ARRAYS


def get_lc(tracksters, _eid):
    # this is not an entirely fair comparison:
    # the LC level methods should use sim LCs not only the CLUE3D ones
    # using sim data here is possible, but gets complicated
    x_lc = ak.flatten(tracksters["vertices_x"].array()[_eid])
    y_lc = ak.flatten(tracksters["vertices_y"].array()[_eid])
    z_lc = ak.flatten(tracksters["vertices_z"].array()[_eid])
    return np.array([x_lc, y_lc, z_lc]).T


def get_trackster_map(t_vertices, t_multiplicity, f_min=0):
    """
    Create mapping of vertex_id to an array of tupples:
        (trackster_id, energy_fraction)

    Input:
        t_vertices: array of vertex IDs in dimensions (tracksters, vertices)
        t_multiplicity: array of vertex multiplicities in dims (tracksters, vertices)
        f_min: minimum energy fraction

    Output:
        {vertex: [(trackster_id, energy_fraction)]}
    """
    i2te = {}
    for t_idx in range(len(t_vertices)):
        for i, m in zip(t_vertices[t_idx], t_multiplicity[t_idx]):
            f = 1. / m
            if f > f_min:
                if i not in i2te:
                    i2te[i] = []
                i2te[i].append((t_idx, f))
    return i2te


def remap_arrays_by_label(array, labels):
    h = max(labels) + 1
    rm = []

    for i in range(h):
        rm.append([])

    for l, i in zip(labels, array):
        if l >= 0:
            rm[l] += list(i)

    return ak.Array(rm)

def remap_items_by_label(array, labels):
    h = max(labels) + 1
    rm = []

    for i in range(h):
        rm.append([])

    for l, i in zip(labels, array):
        if l >= 0:
            rm[l].append(i)

    return ak.Array(rm)


def get_merge_map(pair_index, preds, threshold):
    """
        Performs the little-big mapping
        Respects the highest prediction score for each little
    """
    merge_map = {}
    score_map = {}

    # should always be (little, big)
    for (little, big), p in zip(pair_index, preds):
        if p > threshold:
            if not little in score_map or score_map[little] < p:
                merge_map[little] = [big]
                score_map[little] = p

    return merge_map


def get_merge_map_multi(pair_index, preds, threshold):
    """
        Performs the little-big mapping
        Respects the highest prediction score for each little
    """
    merge_map = {}

    # should always be (little, big)
    for (little, big), p in zip(pair_index, preds):
        if p > threshold:
            if not little in merge_map:
                merge_map[little] = []
            merge_map[little].append(big)
    return merge_map


def merge_tracksters(trackster_data, merged_tracksters, eid):
    datalist = list(trackster_data[k][eid] for k in ARRAYS)
    result = {
        k: ak.Array([ak.flatten(datalist[i][list(set(tlist))]) for tlist in merged_tracksters])
        for i, k in enumerate(ARRAYS)
    }
    # recompute barycentres
    tve = result["vertices_energy"]
    for coord in ("x", "y", "z"):
        _bary = [np.average(vx, weights=ve) for vx, ve in zip(result[f"vertices_{coord}"], tve)]
        result[f"barycenter_{coord}"] = ak.Array(_bary)
    return result


def remap_tracksters(trackster_data, pair_index, preds, eid, decision_th=0.5, pileup=False, allow_multiple=False):
    """
        provide a mapping in format (source: target)
    """
    if allow_multiple:
        new_mapping = get_merge_map_multi(pair_index, preds, decision_th)
    else:
        new_mapping = get_merge_map(pair_index, preds, decision_th)

    if pileup:
        # only include right-handed tracksters
        p_list = set(b for _, b in pair_index)
        new_tracksters = [[b] for b in p_list]
    else:
        # include all tracksters
        new_tracksters = [[i] for i in range(len(trackster_data["raw_energy"][eid]))]

    new_idx_map = {o[0]: i for i, o in enumerate(new_tracksters)}

    for l, bigs in new_mapping.items():
        for b in bigs:
            new_b_idx = new_idx_map[b]
            new_l_idx = new_idx_map.get(l, -1)

            if l == b or new_l_idx == new_b_idx:
                # sanity check: same trackster or already merged
                #   otherwise we delete the trackster
                continue

            if new_l_idx == -1:
                # assign to a trackster
                new_tracksters[new_b_idx].append(l)
                new_idx_map[l] = new_b_idx
            else:
                # merge tracksters
                new_tracksters[new_b_idx] += new_tracksters[new_l_idx]
                # forward dictionary references
                for k, v in new_idx_map.items():
                    if v == new_l_idx:
                        new_idx_map[k] = new_b_idx
                # remove the old record
                new_tracksters[new_l_idx] = []

    merged_tracksters = list(t for t in new_tracksters if t)
    return merge_tracksters(trackster_data, merged_tracksters, eid)
