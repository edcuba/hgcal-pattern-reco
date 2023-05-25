import torch
import numpy as np
import awkward as ak


from .energy import get_energy_map
from .event import get_trackster_map, remap_arrays_by_label, remap_tracksters, merge_tracksters

from .dataset_pair import get_event_pairs
from .dataset_graph import get_event_graph

def f_score(precision, recall, beta=1):
    """
        F-Beta score
            when beta = 1, this is equal to F1 score
    """
    return ((1 + beta**2) * precision * recall) / ((precision * beta**2) + recall)


def B(it, jt):
    """
    Consider that it and jt are in the same objects in one clustering
    Are they in the same objects in the other clustering as well?
    """
    # compute the intersection
    fr_inter = 0.
    for it_x, it_f in it:
        for jt_x, jt_f in jt:
            if it_x == jt_x:
                fr_inter += it_f + jt_f

    # compute the intersection over union (sum of fractions must be 2)
    return fr_inter / 2


def get_pairwise_scores(i, V, i2t, te_map):
    """
    Compute the score per trackster/simtrackster
    normalized by the number of pairs or total pair energy
    Inputs:
        i       target LC
        V       L or C - all LCs of the trackster or simtrackster
        i2t     mapping from LC to its trackster or simtracksters on the oposite side
        te_map  LC -> energy in the given trackster
    Output:
        Normalized sum of the pair-wise scores
    """
    e_pairs = 0
    i_trackster_score = 0

    # for all vertices of the trackster
    for j in V:
        # get the pair energy
        e_pair = (te_map[i] * te_map[j])
        e_pairs += e_pair

        # get (sim-)tracksters of i and j on the other side
        if i in i2t and j in i2t:
            pair_score = B(i2t[i], i2t[j])
            # multiply the score by the pair energy
            i_trackster_score += pair_score * e_pair

    return i_trackster_score / e_pairs


def bcubed(vertices, t_vertices, i2a, i2b, e_map):
    """
    Base algorithm for bcubed precision and recall
    Input:
        vertices: all layerclusters in the event
        t_vertices:
            precision: trackster to layercluster mapping
            recall: simtrackster to layercluster mapping
        i2a, i2b:
            precision:
                layercluster to tracksters map
                layercluster to simtracksters map
            recall: reverse order (i2b, i2a)
        e_map:
            precision: LC to energy in a trackster
            recall: LC to energy in a simtrackster
    Returns: precision / recall for the given input
    """
    P = 0
    # for all reco/sim vertices
    for i in vertices:
        # get all tracksters/simtracksters i is in
        i_tracksters = i2a.get(i, [])

        # get score for each trackster/simtrackster i is in
        for i_trackster, _ in i_tracksters:
            V = t_vertices[int(i_trackster)]
            te_map = e_map[int(i_trackster)]

            # normalize by the number of tracksters and add to the outer P sum
            P += te_map[i] * get_pairwise_scores(i, V, i2b, te_map) / len(i_tracksters)

    total_e = sum(sum(te.values()) for te in e_map.values())

    # normalize the result
    return P / total_e


def evaluate(nhits, all_t_indexes, all_st_indexes, t_energy, st_energy, all_v_multi, all_sv_multi, f_min=0, beta=0.5, min_hits=1):
    """
    Core evaluation routine

    1. Select all layerclusters with more than 1 hit
    2. Get mapping between layerclusters and (sim)tracksters
    3. Get the fraction of energy for each layerclusters and trackster
    4. Compute B-CUBED precision and recall
    """

    # prepare RECO indexes
    lc_over_1_hit = ak.Array([nhits[t] > min_hits for t in all_t_indexes])
    t_indexes = all_t_indexes[lc_over_1_hit]
    v_multi = all_v_multi[lc_over_1_hit]
    t_vertices = ak.flatten(t_indexes)

    # prepare SIM indexes
    slc_over_1_hit = ak.Array([nhits[t] > min_hits for t in all_st_indexes])
    st_indexes = all_st_indexes[slc_over_1_hit]
    sv_multi = all_sv_multi[slc_over_1_hit]
    st_vertices = ak.Array(set(ak.flatten(st_indexes)))

    # precompute LC -> Trackster mapping
    i2rt = get_trackster_map(t_indexes, v_multi)
    i2st = get_trackster_map(st_indexes, sv_multi, f_min=f_min)

    # precompute LC -> Energy mapping (same for all tracksters the LC is in)
    te_map = get_energy_map(t_indexes, t_energy, v_multi)
    ste_map = get_energy_map(st_indexes, st_energy, sv_multi)

    precision = bcubed(t_vertices, t_indexes, i2rt, i2st, te_map)
    recall = bcubed(st_vertices, st_indexes, i2st, i2rt, ste_map)

    return precision, recall, f_score(precision, recall, beta=beta)


def eval_graph_lp(trackster_data, eid, dX, model, pileup=False, decision_th=0.5):
    """
    Evaluation routine for graph-based link prediction
    """

    pairs = []
    preds = []
    truths = []

    # for each event sub-graph
    for sample in dX:

        nidx = sample.node_index

        # compute predicted edges, retrieve the ground truth, and compose pairs
        preds += model(sample.x, sample.pos, sample.edge_index).reshape(-1).tolist()
        truths += sample.y.tolist()
        pairs += [(nidx[a].item(), nidx[b].item()) for a, b in sample.edge_index.T]

    # rebuild the event using the pairs
    reco = remap_tracksters(trackster_data, pairs, preds, eid, decision_th=decision_th, pileup=pileup, allow_multiple=True)
    target = remap_tracksters(trackster_data, pairs, truths, eid, decision_th=decision_th, pileup=pileup, allow_multiple=False)
    p_list = list(set(b for _, b in pairs))
    return reco, target, p_list


def eval_graph_fb(trackster_data, eid, dX, model, pileup=False, multiparticle=False, decision_th=0.5):
    """
    Evaluation routine for the node classification

    This is the foreground-background case
    Whatever foregrounds are overlapping by >50%, we join them
    Pick the sample with the highest energy
    """
    max_e_sample_idx = -1
    max_e_sample_e = -1

    p_list = []

    preds = []
    truths = []
    nodes = []

    eng = trackster_data["raw_energy"][eid]

    # compute foreground/background predictions for each event sub-graph
    for s_idx, sample in enumerate(dX):
        bigT_idx = torch.argmax(sample.x[:, 0]).item()
        p_list.append(sample.node_index[bigT_idx].item())

        bigT_e = sample.e[bigT_idx]
        if bigT_e > max_e_sample_e:
            max_e_sample_idx = s_idx
            max_e_sample_e = bigT_e

        preds.append(model(sample.x, sample.pos).detach().cpu()[:,0].reshape(-1))
        truths.append(sample.y.detach().cpu().reshape(-1))
        nodes.append(sample.node_index.detach().cpu().reshape(-1))

    # process the sample
    reco_tracksters = []
    target_tracksters = []

    # merging the foregrounds is sadly not trivial
    if pileup or multiparticle:
        for p, t, n in zip(preds, truths, nodes):
            reco_fg = n[p >= decision_th].tolist()
            target_fg = n[t >= decision_th].tolist()

            # merging
            reco_fg_set = set(reco_fg)
            target_fg_set = set(target_fg)

            for t in reco_tracksters:
                shared = set(t).intersection(reco_fg_set)
                if len(shared) == 0:
                    continue

                t_eng = np.sum(eng[t])
                shared_e = np.sum(eng[shared])
                fg_e = np.sum(eng[reco_fg_set])

                if shared_e >= t_eng * 0.4 or shared_e >= fg_e * 0.4:
                    t += reco_fg_set.difference(shared)
                    break

                if t_eng >= fg_e:
                    reco_fg_set.difference_update(shared)
                else:
                    for sh in shared:
                        t.remove(sh)
            else:
                if len(reco_fg_set):
                    reco_tracksters.append(list(reco_fg_set))

            for t in target_tracksters:
                shared = set(t).intersection(target_fg_set)
                if len(shared) == 0:
                    continue

                t_eng = np.sum(eng[t])
                shared_e = np.sum(eng[shared])
                fg_e = np.sum(eng[target_fg_set])

                if shared_e >= t_eng * 0.4 or shared_e >= fg_e * 0.4:
                    t += target_fg_set.difference(shared)
                    break

                if t_eng >= fg_e:
                    target_fg_set.difference_update(shared)
                else:
                    for sh in shared:
                        t.remove(sh)
            else:
                if len(target_fg_set):
                    target_tracksters.append(list(target_fg_set))

    else:
        # two particles case is easy
        p = preds[max_e_sample_idx]
        t = truths[max_e_sample_idx]
        n = nodes[max_e_sample_idx]

        reco_fg = n[p >= decision_th].tolist()
        reco_bg = n[p < decision_th].tolist()
        if reco_fg: reco_tracksters.append(reco_fg)
        if reco_bg: reco_tracksters.append(reco_bg)

        target_fg = n[t >= decision_th].tolist()
        target_bg = n[t < decision_th].tolist()
        if target_fg: target_tracksters.append(target_fg)
        if target_bg: target_tracksters.append(target_bg)

    reco = merge_tracksters(trackster_data, reco_tracksters, eid)
    target = merge_tracksters(trackster_data, target_tracksters, eid)
    return reco, target, p_list


def model_evaluation(
    cluster_data,
    trackster_data,
    simtrackster_data,
    assoc_data,
    model,
    decision_th=0.5,
    radius=10,
    max_events=100,
    bigT_e_th=50,
    pileup=True,
    collection="SC",
    graph=False,
    reco_eval=True,
    link_prediction=False,
    multiparticle=False,
):
    """
    Evaluation routine used to assess the model performance and compare against the baselines.
    Evaluation must be unbalanced (true distribution)
    """
    model.eval()

    results = {
        "clue3d_to_sim": [],
        "target_to_sim": [],
    }

    if reco_eval:
        results["reco_to_sim"] = []
        results["n_tracksters"] = []

    actual_range = min([len(trackster_data["raw_energy"]), max_events])
    for eid in range(actual_range):
        print(f"Event {eid}:")

        if graph:
            dX = get_event_graph(
                cluster_data,
                trackster_data,
                assoc_data,
                eid,
                radius,
                pileup=pileup,
                bigT_e_th=bigT_e_th,
                collection=collection,
                link_prediction=link_prediction,
            )
        else:
            dX, dY, pair_index = get_event_pairs(
                cluster_data,
                trackster_data,
                assoc_data,
                eid,
                radius,
                pileup=pileup,
                bigT_e_th=bigT_e_th,
                collection=collection
            )

        if len(dX) == 0:
            print("\tNo data")
            continue

        # predict edges
        if graph and link_prediction:
            reco, target, p_list = eval_graph_lp(
                trackster_data,
                eid,
                dX,
                model,
                pileup=pileup,
                decision_th=decision_th
            )
        elif graph and not link_prediction:
            reco, target, p_list = eval_graph_fb(
                trackster_data,
                eid,
                dX,
                model,
                pileup=pileup,
                decision_th=decision_th,
                multiparticle=multiparticle,
            )
        else:
            preds = model(torch.tensor(dX, dtype=torch.float)).detach().cpu().reshape(-1).tolist()
            truth = np.array(dY)

            # rebuild the event
            reco = remap_tracksters(trackster_data, pair_index, preds, eid, decision_th=decision_th, pileup=pileup)
            target = remap_tracksters(trackster_data, pair_index, truth, eid, decision_th=decision_th, pileup=pileup)
            p_list = list(set(b for _, b in pair_index))


        clusters_e = cluster_data["energy"][eid]

        # target
        target_i = target["vertices_indexes"]
        target_m = target["vertices_multiplicity"]
        target_e = ak.Array([clusters_e[indices] for indices in target_i])

        # clue3D
        ci = trackster_data["vertices_indexes"][eid]
        cm = trackster_data["vertices_multiplicity"][eid]
        ce = ak.Array([clusters_e[indices] for indices in ci])

        if pileup:
            # need to filter out all the unrelated stuff
            # keep only the big tracksters (right side)
            ce = ce[p_list]
            ci = ci[p_list]
            cm = cm[p_list]

        # simulation
        p = "" if pileup else f"sts{collection}_"
        si = simtrackster_data[f"{p}vertices_indexes"][eid]
        sm = simtrackster_data[f"{p}vertices_multiplicity"][eid]
        se = ak.Array([clusters_e[indices] for indices in si])

        nhits = cluster_data["cluster_number_of_hits"][eid]

        results["clue3d_to_sim"].append(evaluate(nhits, ci, si, ce, se, cm, sm))
        results["target_to_sim"].append(evaluate(nhits, target_i, si, target_e, se, target_m, sm))

        if reco_eval:
            # reco
            ri = reco["vertices_indexes"]
            rm = reco["vertices_multiplicity"]
            re = ak.Array([clusters_e[indices] for indices in ri])
            results["reco_to_sim"].append(evaluate(nhits, ri, si, re, se, rm, sm))

        for key, values in results.items():
            if key == "n_tracksters":
                continue
            vals = values[-1]
            print(f"\t{key}:\tP: {vals[0]:.3f} R: {vals[1]:.3f} F: {vals[2]:.3f}")

        if reco_eval:
            results["n_tracksters"].append((len(si), len(target_i), len(ri)))
            print(f"\t|Sim| = {len(si)} |Target| = {len(target_i)} |Reco| = {len(ri)}")

    print("-----")
    for key, values in results.items():
        avg_p = np.sum([x[0] for x in values]) / actual_range
        avg_r = np.sum([x[1] for x in values]) / actual_range
        avg_f = np.sum([x[2] for x in values]) / actual_range
        print(f"mean {key}:\tP: {avg_p:.3f} R: {avg_r:.3f} F: {avg_f:.3f}")

    return results
