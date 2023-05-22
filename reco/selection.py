import numpy as np


def get_major_PU_tracksters(
    reco2sim_sharedE,
    raw_energy,
    fraction_threshold=0.5,
):
    # assuming only one simtrackster to keep things easy
    big = []

    for recoT_idx, shared_energies in enumerate(reco2sim_sharedE):
        for shared_energy in shared_energies:
            # 2 goals here:
            # - find the trackster with >50% shared energy
            # - find the tracksters with < 0.2 score
            # if score > score_threshold: continue

            t_energy = raw_energy[recoT_idx]
            se_fraction = shared_energy / t_energy

            if se_fraction > fraction_threshold:
                big.append(recoT_idx)

    return big


def get_bigTs(trackster_data, assoc_data, eid, pileup=True, energy_th=10, collection="SC"):
    """
    Select tracksters above a given energy threshold
    """

    if pileup:
        # get associations data
        reco2sim_sharedE = assoc_data[f"tsCLUE3D_recoToSim_{collection}_sharedE"][eid]
        raw_energy = trackster_data["raw_energy"][eid]

        # select only tracksters for which simdata is available
        bigTs = get_major_PU_tracksters(
            reco2sim_sharedE,
            raw_energy,
        )
        return np.array(bigTs)[raw_energy[bigTs] > energy_th].tolist()

    # select tracksters above energy_th
    return np.nonzero(trackster_data["raw_energy"][eid] > energy_th)[0].tolist()
