import awkward as ak
from os import walk, path
import sys

import torch

import numpy as np
from torch_geometric.data import Data, InMemoryDataset

from .features import get_graph_level_features, get_min_max_z_points
from .graphs import create_graph
from .data import get_event_data, FEATURE_KEYS, get_bary_data

from .geometry import get_neighborhood
from .selection import get_bigTs



def get_event_graph(
        cluster_data,
        trackster_data,
        assoc_data,
        eid,
        radius=10,
        bigT_e_th=10,
        pileup=False,
        collection="SC",
        link_prediction=False,
    ):
    data_list = []

    # get LC info
    clusters_x = cluster_data["position_x"][eid]
    clusters_y = cluster_data["position_y"][eid]
    clusters_z = cluster_data["position_z"][eid]
    clusters_e = cluster_data["energy"][eid]

    # get trackster info
    id_probs = trackster_data["id_probabilities"][eid].tolist()

    # reconstruct trackster LC info
    vertices_indices = trackster_data["vertices_indexes"][eid]
    vertices_x = ak.Array([clusters_x[indices] for indices in vertices_indices])
    vertices_y = ak.Array([clusters_y[indices] for indices in vertices_indices])
    vertices_z = ak.Array([clusters_z[indices] for indices in vertices_indices])
    vertices_e = ak.Array([clusters_e[indices] for indices in vertices_indices])

    bary = get_bary_data(trackster_data, eid)
    raw_energy = trackster_data["raw_energy"][eid]

    # get associations data
    reco2sim_score = assoc_data[f"tsCLUE3D_recoToSim_{collection}_score"][eid]
    reco2sim_idx = assoc_data[f"tsCLUE3D_recoToSim_{collection}"][eid]
    reco2sim_shared_e = assoc_data[f"tsCLUE3D_recoToSim_{collection}_sharedE"][eid]

    bigTs = get_bigTs(
        trackster_data,
        assoc_data,
        eid,
        pileup=pileup,
        energy_th=bigT_e_th,
        collection=collection,
    )

    trackster_features = list([
        trackster_data[k][eid] for k in FEATURE_KEYS
    ])

    node_features = []
    node_labels = []
    node_index = []
    node_pos = []
    node_eng = []
    node_shared_e = []
    node_simT_idx = []
    bigT_index = []
    index_map = {}
    edge_labels = []

    for bigT in bigTs:
        # produce a graph for each bigT
        if not link_prediction:
            node_features = []
            node_labels = []
            node_index = []
            node_pos = []
            node_eng = []
            node_shared_e = []
            node_simT_idx = []

        # find index of the best score
        bigT_best_score_idx = np.argmin(reco2sim_score[bigT])
        # figure out which simtrackster it is
        bigT_simT_idx = reco2sim_idx[bigT][bigT_best_score_idx]

        # get the best score
        bigT_best_score = reco2sim_score[bigT][bigT_best_score_idx]

        in_cone = get_neighborhood(trackster_data, vertices_z, eid, radius, bigT)
        for recoTxId, distance in in_cone:

            # find out the index of the simpartice we are looking for
            recoTx_bigT_simT_idx = np.argwhere(reco2sim_idx[recoTxId] == bigT_simT_idx)[0][0]
            label = (1 - reco2sim_score[recoTxId][recoTx_bigT_simT_idx])

            if link_prediction:
                if bigT != recoTxId:
                    edge_labels.append(label * (1 - bigT_best_score))
                    bigT_index.append((recoTxId, bigT))
                if recoTxId in index_map:
                    # already in the graph - just add an edge
                    continue
                else:
                    index_map[recoTxId] = len(node_labels)

            recoTx_graph = create_graph(
                vertices_x[recoTxId],
                vertices_y[recoTxId],
                vertices_z[recoTxId],
                vertices_e[recoTxId],
            )

            minP, maxP = get_min_max_z_points(
                vertices_x[recoTxId],
                vertices_y[recoTxId],
                vertices_z[recoTxId],
            )

            if link_prediction:
                features = []
            else:
                features = [
                    int(recoTxId == bigT),
                    distance,
                ]

            features.append(len(vertices_z[recoTxId]))
            features += [f[recoTxId] for f in trackster_features]
            features += minP
            features += maxP
            features += id_probs[recoTxId]
            features += get_graph_level_features(recoTx_graph)

            # get the score for the given simparticle and compute the score
            shared_e = reco2sim_shared_e[recoTxId][recoTx_bigT_simT_idx]

            node_features.append(features)
            node_labels.append(label)
            node_index.append(recoTxId)
            node_pos.append(bary[recoTxId].tolist())
            node_eng.append(raw_energy[recoTxId])
            node_shared_e.append(shared_e)
            node_simT_idx.append(bigT_simT_idx)

        if not link_prediction:
            data_list.append(Data(
                x=torch.tensor(node_features, dtype=torch.float),
                e=torch.tensor(node_eng, dtype=torch.float),
                shared_e=torch.tensor(node_shared_e, dtype=torch.float),
                pos=torch.tensor(node_pos, dtype=torch.float),
                y=torch.tensor(node_labels, dtype=torch.float),
                node_index=torch.tensor(node_index, dtype=torch.int),
                simT=torch.tensor(node_simT_idx, dtype=torch.int)
            ))
    if link_prediction:
        if not bigT_index:
            return []
        edge_index = [(index_map[a], index_map[b]) for a, b in bigT_index]
        return [Data(
            x=torch.tensor(node_features, dtype=torch.float),
            e=torch.tensor(node_eng, dtype=torch.float),
            shared_e=torch.tensor(node_shared_e, dtype=torch.float),
            pos=torch.tensor(node_pos, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).T,
            y=torch.tensor(edge_labels, dtype=torch.float),
            node_index=torch.tensor(node_index, dtype=torch.long),
            simT_score=torch.tensor(node_labels, dtype=torch.float),
            simT=torch.tensor(node_simT_idx, dtype=torch.long)
        )]
    return data_list


class TracksterGraph(InMemoryDataset):

    def __init__(
            self,
            name,
            root_dir,
            raw_data_path,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            N_FILES=None,
            radius=10,
            pileup=False,
            bigT_e_th=10,
            collection="SC",
            link_prediction=False,
        ):
        self.name = name
        self.pileup = pileup
        self.N_FILES = N_FILES
        self.raw_data_path = raw_data_path
        self.root_dir = root_dir
        self.RADIUS = radius
        self.bigT_e_th = bigT_e_th
        self.collection = collection
        self.link_prediction = link_prediction
        super(TracksterGraph, self).__init__(root_dir, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        files = []
        for (_, _, filenames) in walk(self.raw_data_path):
            files.extend(filenames)
            break
        full_paths = list([path.join(self.raw_data_path, f) for f in files])
        if self.N_FILES:
            assert len(full_paths) >= self.N_FILES
        return full_paths[:self.N_FILES]

    @property
    def processed_file_names(self):
        infos = [
            self.name,
            f"f{self.N_FILES or len(self.raw_file_names)}",
            f"r{self.RADIUS}",
            f"eth{self.bigT_e_th}"
        ]
        if self.link_prediction:
            infos.append("lp")
        return list([f"TracksterGraph{'PU' if self.pileup else ''}_{'_'.join(infos)}.pt"])

    @property
    def processed_paths(self):
        return [path.join(self.root_dir, fn) for fn in self.processed_file_names]

    def process(self):
        data_list = []
        for source in self.raw_file_names:
            print(source, file=sys.stderr)
            cluster_data, trackster_data, _, assoc_data = get_event_data(
                source,
                collection=self.collection,
            )
            for eid in range(len(trackster_data["barycenter_x"])):
                data_list += get_event_graph(
                    cluster_data,
                    trackster_data,
                    assoc_data,
                    eid,
                    self.RADIUS,
                    pileup=self.pileup,
                    bigT_e_th=self.bigT_e_th,
                    collection=self.collection,
                    link_prediction=self.link_prediction,
                )

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        infos = [
            f"graphs={len(self)}",
            f"nodes={len(self.data.x)}",
            f"radius={self.RADIUS}",
            f"bigT_e_th={self.bigT_e_th}",
        ]
        if self.link_prediction:
            infos.append("lp")
        return f"TracksterGraph{'PU' if self.pileup else ''}({', '.join(infos)})"
