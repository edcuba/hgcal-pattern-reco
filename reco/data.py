import uproot
import numpy as np
import awkward as ak


ARRAYS = [
    "vertices_x",
    "vertices_y",
    "vertices_z",
    "vertices_energy",
    "vertices_multiplicity",
    "vertices_indexes",
]


FEATURE_KEYS = [
    "barycenter_x",
    "barycenter_y",
    "barycenter_z",
    "trackster_barycenter_eta",
    "trackster_barycenter_phi",
    "raw_energy",       # total energy
    "raw_em_energy",    # electro-magnetic energy
    "EV1",              # eigenvalues of 1st principal component
    "EV2",
    "EV3",
    "eVector0_x",       # x of principal component
    "eVector0_y",
    "eVector0_z",
    "sigmaPCA1",        # error in the 1st principal axis
    "sigmaPCA2",
    "sigmaPCA3",
]


def clusters_by_indices(cluster_data, indices, eid):
    clusters_x = cluster_data["position_x"][eid]
    clusters_y = cluster_data["position_y"][eid]
    clusters_z = cluster_data["position_z"][eid]
    clusters_e = cluster_data["energy"][eid]

    t_x = ak.Array([clusters_x[indices] for indices in indices])
    t_y = ak.Array([clusters_y[indices] for indices in indices])
    t_z = ak.Array([clusters_z[indices] for indices in indices])
    t_e = ak.Array([clusters_e[indices] for indices in indices])

    return t_x, t_y, t_z, t_e


def get_data_arrays(clusters, tracksters, simtracksters, associations, collection="SC", pileup=False):
    trackster_data = tracksters.arrays(ARRAYS + FEATURE_KEYS + ['id_probabilities'])
    cluster_data = clusters.arrays([
        "position_x",
        "position_y",
        "position_z",
        "energy",
        "cluster_number_of_hits",
    ])

    p = "" if pileup else f"sts{collection}_"
    simtrackster_data = simtracksters.arrays([
        f"{p}raw_energy",
        f"{p}vertices_indexes",
        f"{p}vertices_energy",
        f"{p}vertices_multiplicity",
        f"{p}barycenter_z"
    ])
    assoc_data = associations.arrays([
        f"tsCLUE3D_recoToSim_{collection}",
        f"tsCLUE3D_recoToSim_{collection}_sharedE",
        f"tsCLUE3D_recoToSim_{collection}_score",
    ])
    return cluster_data, trackster_data, simtrackster_data, assoc_data


def get_bary_data(trackster_data, _eid):
    return np.array([
        trackster_data["barycenter_x"][_eid],
        trackster_data["barycenter_y"][_eid],
        trackster_data["barycenter_z"][_eid],
    ]).T


def get_event_data(source, collection="SC", pileup=False):
    tracksters = uproot.open({source: "ticlNtuplizer/tracksters"})
    simtracksters = uproot.open({source: f"ticlNtuplizer/simtracksters{collection}"})
    associations = uproot.open({source: "ticlNtuplizer/associations"})
    clusters = uproot.open({source: "ticlNtuplizer/clusters"})
    return get_data_arrays(
        clusters,
        tracksters,
        simtracksters,
        associations,
        collection=collection,
        pileup=pileup
    )


def get_lc_data(cluster_data, trackster_data, _eid):
    # this is not an entirely fair comparison:
    # the LC level methods should use sim LCs not only the CLUE3D ones
    # using sim data here is possible, but gets complicated
    indices = trackster_data["vertices_indexes"][_eid]
    t_x, t_y, t_z, _t_e = clusters_by_indices(cluster_data, indices, _eid)
    return np.array([ak.flatten(t_x), ak.flatten(t_y), ak.flatten(t_z)]).T