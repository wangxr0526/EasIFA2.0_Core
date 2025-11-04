
import os
import torch
import dgl
from collections.abc import Mapping, Sequence
import logging
from torchdrug import data
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

logger = logging.getLogger(__name__)

from data_loaders.rxn_dataloader import (
    pad_atom_distance_matrix,
)


SITE_TYPE_TO_CLASS = {"BINDING": 0, "ACT_SITE": 1, "SITE": 2}



class ReactionFeatures(object):
    def __init__(self, reaction_features) -> None:

        (self.react_fgraph, self.react_dgraph), (self.prod_fgraph, self.prod_dgraph) = (
            reaction_features
        )

        self.react_dgraph = torch.from_numpy(self.react_dgraph)
        self.prod_dgraph = torch.from_numpy(self.prod_dgraph)

    def __repr__(self) -> str:
        repr_str = f"Reaction(reactants(fgraph:{self.react_fgraph},dgraph:{self.react_dgraph}; products(fgraph:{self.prod_fgraph}, dgraph:{self.prod_dgraph}) )"
        return repr_str

    def to(self, device):
        self.react_fgraph = self.react_fgraph.to(device)
        self.react_dgraph = self.react_dgraph.to(device)
        self.prod_fgraph = self.prod_fgraph.to(device)
        self.prod_dgraph = self.prod_dgraph.to(device)


def collate_rxn_features(batch):
    react_fgraphs = [x.react_fgraph for x in batch]
    react_dgraphs = [x.react_dgraph for x in batch]
    prod_fgraphs = [x.prod_fgraph for x in batch]
    prod_dgraphs = [x.prod_dgraph for x in batch]

    rts_bg = dgl.batch(react_fgraphs)
    pds_bg = dgl.batch(prod_fgraphs)
    rts_bg.set_n_initializer(dgl.init.zero_initializer)
    pds_bg.set_n_initializer(dgl.init.zero_initializer)

    rts_adms = pad_atom_distance_matrix(react_dgraphs)
    pds_adms = pad_atom_distance_matrix(prod_dgraphs)

    rts_node_feats, rts_edge_feats = rts_bg.ndata.pop("h"), rts_bg.edata.pop("e")
    pds_node_feats, pds_edge_feats = pds_bg.ndata.pop("h"), pds_bg.edata.pop("e")

    return {
        "rts_bg": rts_bg,
        "rts_adms": rts_adms,
        "rts_node_feats": rts_node_feats,
        "rts_edge_feats": rts_edge_feats,
        "pds_bg": pds_bg,
        "pds_adms": pds_adms,
        "pds_node_feats": pds_node_feats,
        "pds_edge_feats": pds_edge_feats,
    }


def enzyme_rxn_collate(batch):
    """
    Convert any list of same nested container into a container of tensors.

    For instances of :class:`data.Graph <torchdrug.data.Graph>`, they are collated
    by :meth:`data.Graph.pack <torchdrug.data.Graph.pack>`.

    Parameters:
        batch (list): list of samples with the same nested container
    """
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        shapes = [x.shape for x in batch]
        return torch.cat(batch, 0, out=out), torch.tensor(shapes)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, data.Graph):
        return elem.pack(batch)
    elif isinstance(elem, Mapping):
        return {key: enzyme_rxn_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("Each element in list of batch should be of equal size")
        return [enzyme_rxn_collate(samples) for samples in zip(*batch)]

    elif isinstance(elem, ReactionFeatures):
        return collate_rxn_features(batch)

    raise TypeError("Can't collate data with type `%s`" % type(elem))


def enzyme_rxn_collate_extract(batch):
    batch = [x for x in batch if x]
    if not batch:
        return
    batch_data = enzyme_rxn_collate(batch)
    assert isinstance(batch_data, dict)
    if "targets" in batch_data:
        if isinstance(batch_data["targets"], tuple):
            targets, size = batch_data["targets"]
            batch_data["targets"] = targets
            batch_data["protein_len"] = size.view(-1)
    return batch_data


