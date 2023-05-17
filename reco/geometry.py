import numpy as np


def get_trackster_representative_points(bx, by, bz, min_z, max_z):
    """
    Take a line (0,0,0), (bx, by, bz) -> any point on the line is t*(bx, by, bz)
    Compute the intersection with the min and max layer
    Beginning of the line: (minx, miny, minz) = t*(bx, by, bz)
    """
    t_min = min_z / bz
    t_max = max_z / bz
    x1 = np.array((t_min * bx, t_min * by, min_z))
    x2 = np.array((t_max * bx, t_max * by, max_z))
    return x1, x2


def get_tracksters_in_cylinder(x1, x2, barycentres, radius=10):
    """
    Get Candidate Tracksters in a cylinder defined two representative points and a radius
    """
    in_cone = []
    for i, x0 in enumerate(barycentres):
        # barycenter between the first and last layer
        if x0[2] > x1[2] - radius and x0[2] < x2[2] + radius:
            # distance from the particle axis less than X cm
            d = np.linalg.norm(np.cross(x0 - x1, x0 - x2)) / np.linalg.norm(x2 - x1)
            if d < radius:
                in_cone.append((i, d))
    return in_cone


def get_neighborhood(trackster_data, vertices_z, eid, radius, bigT):
    """
    Get Candidate Tracksters within the neighborhood of the Selected Trackster
    """

    # get Selected Trackster trackster info
    barycenter_x = trackster_data["barycenter_x"][eid]
    barycenter_y = trackster_data["barycenter_y"][eid]
    barycenter_z = trackster_data["barycenter_z"][eid]

    x1, x2 = get_trackster_representative_points(
        barycenter_x[bigT],
        barycenter_y[bigT],
        barycenter_z[bigT],
        min(vertices_z[bigT]),
        max(vertices_z[bigT])
    )
    barycentres = np.array((barycenter_x, barycenter_y, barycenter_z)).T
    return get_tracksters_in_cylinder(x1, x2, barycentres, radius=radius)
