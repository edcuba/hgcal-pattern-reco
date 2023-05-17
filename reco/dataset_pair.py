import awkward as ak
from os import walk, path
import sys

import torch

import numpy as np
from torch.utils.data import Dataset

from .features import get_min_max_z_points
from .data import get_event_data, FEATURE_KEYS

from .geometry import get_neighborhood
from .selection import get_bigTs


def build_pair_tensor(edge, features):
    a, b = edge
    fa = [f[a] for f in features]
    fb = [f[b] for f in features]
    return fa + fb


def get_event_pairs(
        cluster_data,
        trackster_data,
        assoc_data,
        eid,
        radius,
        pileup=False,
        bigT_e_th=50,
        collection="SC",
    ):

    dataset_X = []
    dataset_Y = []
    pair_index = []

    # get LC info
    clusters_x = cluster_data["position_x"][eid]
    clusters_y = cluster_data["position_y"][eid]
    clusters_z = cluster_data["position_z"][eid]

    # reconstruct trackster LC info
    vertices_indices = trackster_data["vertices_indexes"][eid]
    vertices_x = ak.Array([clusters_x[indices] for indices in vertices_indices])
    vertices_y = ak.Array([clusters_y[indices] for indices in vertices_indices])
    vertices_z = ak.Array([clusters_z[indices] for indices in vertices_indices])

    reco2sim_score = assoc_data[f"tsCLUE3D_recoToSim_{collection}_score"][eid]
    reco2sim_idx = assoc_data[f"tsCLUE3D_recoToSim_{collection}"][eid]

    # add id probabilities
    id_probs = trackster_data["id_probabilities"][eid].tolist()

    bigTs = get_bigTs(
        trackster_data,
        assoc_data,
        eid,
        pileup=pileup,
        energy_th=bigT_e_th,
        collection=collection
    )

    trackster_features = list([
        trackster_data[k][eid] for k in FEATURE_KEYS
    ])

    for bigT in bigTs:

        big_minP, big_maxP = get_min_max_z_points(
            vertices_x[bigT],
            vertices_y[bigT],
            vertices_z[bigT],
        )

        # find index of the best score
        bigT_best_score_idx = np.argmin(reco2sim_score[bigT])
        # get the best score
        bigT_best_score = reco2sim_score[bigT][bigT_best_score_idx]
        # figure out which simtrackster it is
        bigT_simT_idx = reco2sim_idx[bigT][bigT_best_score_idx]

        in_cone = get_neighborhood(trackster_data, vertices_z, eid, radius, bigT)

        for recoTxId, distance in in_cone:

            if recoTxId == bigT:
                # do not connect to itself
                continue

            # do not connect large tracksters
            # if recoTxId in bigTs:
            #     continue

            features = build_pair_tensor((bigT, recoTxId), trackster_features)

            minP, maxP = get_min_max_z_points(
                vertices_x[recoTxId],
                vertices_y[recoTxId],
                vertices_z[recoTxId],
            )

            # add trackster axes
            features += big_minP
            features += big_maxP
            features += minP
            features += maxP
            features += id_probs[bigT]
            features += id_probs[recoTxId]

            features.append(distance)
            features.append(len(vertices_z[bigT]))
            features.append(len(vertices_z[recoTxId]))

            # find out the index of the simpartice we are looking for
            recoTx_bigT_simT_idx = np.nonzero(reco2sim_idx[recoTxId] == bigT_simT_idx)[0][0]
            # get the score for the given simparticle and compute the score
            label = (1 - bigT_best_score) * (1 - reco2sim_score[recoTxId][recoTx_bigT_simT_idx])

            dataset_X.append(features)
            dataset_Y.append(label)
            pair_index.append((recoTxId, bigT))

    return dataset_X, dataset_Y, pair_index


class TracksterPairs(Dataset):
    # output is about 250kb per file

    def __init__(
            self,
            name,
            root_dir,
            raw_data_path,
            transform=None,
            N_FILES=None,
            radius=10,
            score_threshold=0.2,
            pileup=False,
            bigT_e_th=40,
            collection="SC",
        ):
        self.name = name
        self.N_FILES = N_FILES
        self.RADIUS = radius
        self.SCORE_THRESHOLD = score_threshold
        self.raw_data_path = raw_data_path
        self.root_dir = root_dir
        self.transform = transform
        self.pileup = pileup
        self.bigT_e_th = bigT_e_th
        self.collection = collection
        fn = self.processed_paths[0]

        if not path.exists(fn):
            self.process()

        dx, dy = torch.load(fn)
        self.x = torch.tensor(dx).type(torch.float)
        self.y = torch.tensor(dy).type(torch.float)

    @property
    def raw_file_names(self):
        files = []
        for (_, _, filenames) in walk(self.raw_data_path):
            files.extend(filenames)
            break
        full_paths = list([path.join(self.raw_data_path, f) for f in files])

        if self.N_FILES is None:
            self.N_FILES = len(full_paths)

        return full_paths[:self.N_FILES]

    @property
    def processed_file_names(self):
        infos = [
            self.name,
            f"f{self.N_FILES or len(self.raw_file_names)}",
            f"r{self.RADIUS}",
            f"s{self.SCORE_THRESHOLD}",
            f"eth{self.bigT_e_th}"
        ]
        return list([f"TracksterPairs{'PU' if self.pileup else ''}_{'_'.join(infos)}.pt"])

    @property
    def processed_paths(self):
        return [path.join(self.root_dir, fn) for fn in self.processed_file_names]

    def process(self):
        dataset_X = []
        dataset_Y = []

        assert len(self.raw_file_names) == self.N_FILES

        for source in self.raw_file_names:
            print(f"Processing: {source}", file=sys.stderr)
            cluster_data, trackster_data, _, assoc_data = get_event_data(
                source,
                collection=self.collection,
                pileup=self.pileup
            )
            for eid in range(len(trackster_data["barycenter_x"])):
                dX, dY, _ = get_event_pairs(
                    cluster_data,
                    trackster_data,
                    assoc_data,
                    eid,
                    self.RADIUS,
                    pileup=self.pileup,
                    bigT_e_th=self.bigT_e_th,
                    collection=self.collection,
                )
                dataset_X += dX
                dataset_Y += dY

        torch.save((dataset_X, dataset_Y), self.processed_paths[0])

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.y)

    def __repr__(self):
        infos = [
            f"len={len(self)}",
            f"radius={self.RADIUS}",
            f"score_threshold={self.SCORE_THRESHOLD}",
            f"bigT_e_th={self.bigT_e_th}",
        ]
        return f"<TracksterPairs {' '.join(infos)}>"
