def get_total_energy(energy, multi):
    total = 0
    for t_idx in range(len(energy)):
        for e, m in zip(energy[t_idx], multi[t_idx]):
            total += e / m
    return total


def get_energy_map(t_vertices, t_energy, t_multi):
    """
    Create mapping of vertex id per trackster to the vertex energy in the trackster
    """
    v2e = {}
    for t_idx in range(len(t_vertices)):
        v2e[t_idx] = {}
        for i, e, m in zip(t_vertices[t_idx], t_energy[t_idx], t_multi[t_idx]):
            v2e[t_idx][i] = e / m
    return v2e