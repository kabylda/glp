from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp

from .utils import cast
from .periodic import displacement

Graph = namedtuple("Graph", ("positions", "edges", "nodes", "centers", "others", "mask", "total_charge", "num_unpaired_electrons", "idx_i_lr", "idx_j_lr",  "pair_displacements", "cell", "ngrid", "alpha", "frequency"))

def system_to_graph(system, neighbors, pme):
    # neighbors are an *updated* neighborlist
    # question: how do we treat batching?

    positions = system.R
    nodes = system.Z
    edges = jax.vmap(partial(displacement, system.cell))(
        positions[neighbors.centers], positions[neighbors.others]
    )
    pair_displacements = jax.vmap(partial(displacement, system.cell))(
        positions[neighbors.centers], positions[neighbors.others]
    )

    mask = neighbors.centers != positions.shape[0]

    return Graph(positions, edges, nodes, neighbors.centers, neighbors.others, mask, system.total_charge, system.num_unpaired_electrons, neighbors.pair_i, neighbors.pair_j, pair_displacements, system.cell,  pme.ngrid, pme.alpha, pme.frequencies)

