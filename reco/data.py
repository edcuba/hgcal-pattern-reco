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


def get_data_arrays(clusters, tracksters, simtracksters, associations, collection="SC"):
    trackster_data = tracksters.arrays(ARRAYS + FEATURE_KEYS + ['id_probabilities'])
    cluster_data = clusters.arrays([
        "position_x",
        "position_y",
        "position_z",
        "energy",
        "cluster_number_of_hits",
    ])

    simtrackster_data = simtracksters.arrays([
        f"raw_energy",
        f"vertices_indexes",
        f"vertices_energy",
        f"vertices_multiplicity",
        f"barycenter_z"
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


def get_event_data(source, collection="SC"):
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
    )
