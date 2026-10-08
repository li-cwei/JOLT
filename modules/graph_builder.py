from types import SimpleNamespace
import torch
import torch.nn.functional as F
from modules.data_load import (
    load_ACM,
    load_DBLP,
    load_IMDB,
    load_Yelp,
)
from utils import process


def build_subgraphs(data):
    data.graph = {}
    for nt, rels in data.node_rel.items():
        rel_list = []
        for rel in rels:
            edge = data.edge_dict[rel]
            e = edge[0] if isinstance(edge, list) else edge
            e = process.normalize_adj(e)
            e_tuple = process.sparse_to_tuple(e)
            rel_list.append(torch.sparse_coo_tensor(torch.LongTensor(e_tuple[0]), torch.FloatTensor(e_tuple[1]), torch.Size(e_tuple[2])))
        data.graph[nt] = rel_list

def build_distance_graph(args, data):
    """Build a sparse feature-distance matrix using a global quantile."""
    target_feat = data.features[0]
    dist = process.pairwise_distance(target_feat)
    dist = F.normalize(dist)

    kthvalue = torch.kthvalue(dist.view(-1), int(dist.shape[0] * dist.shape[1] * args.distance_quantile))[0]

    mask = (dist > kthvalue).detach().float()
    data.feature_distance = (dist * mask).to_sparse().coalesce()


def build_filters(data):
    """Build normalized meta-path adjacencies and high-pass filters."""
    data.hp_filters = []
    data.rw_adjs = []
    data.sym_adjs = []

    for M_adj in data.meta_adjs:
        M_raw = M_adj.to_dense()
        rowsum_raw = M_raw.sum(dim=1)
        d_inv_sqrt_sym = torch.pow(rowsum_raw, -0.5)
        d_inv_sqrt_sym[torch.isinf(d_inv_sqrt_sym)] = 0.0
        data.sym_adjs.append((M_raw * d_inv_sqrt_sym.unsqueeze(1) * d_inv_sqrt_sym.unsqueeze(0)).to_sparse())

        M_meta = M_raw.clone()
        M_meta.fill_diagonal_(0.0)
        rowsum_meta = M_meta.sum(dim=1)
        d_inv_sqrt_meta = torch.pow(rowsum_meta, -0.5)
        d_inv_sqrt_meta[torch.isinf(d_inv_sqrt_meta)] = 0.0
        A_hat = M_meta * d_inv_sqrt_meta.unsqueeze(1) * d_inv_sqrt_meta.unsqueeze(0)
        F_HP = -A_hat.clone()
        F_HP.fill_diagonal_(2.0)
        data.hp_filters.append(F_HP.to_sparse())

def GraphBuild(args):
    data = SimpleNamespace()
    LOADERS = {
        'ACM': load_ACM,
        'DBLP': load_DBLP,
        'IMDB': load_IMDB,
        'Yelp': load_Yelp,
    }
    loader = LOADERS.get(args.dataset)
    if loader is None:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    (   data.features,
        data.meta_adjs,
        data.node_rel,
        data.edge_dict,
        data.labels,
        data.train_idx,
        data.val_idx,
        data.test_idx,
        data.node_idx,
        data.str_pos,
    ) = loader(args)

    data.nt_rel = data.node_rel
    data.node_cnt = data.node_idx
    data.node_type = list(data.node_idx.keys())
    data.target_nt = data.node_type[0]
    data.num_target = data.node_idx[data.target_nt].shape[0]
    data.nb_classes = data.labels.shape[-1]
    data.node_size = sum(len(idx) for idx in data.node_cnt.values())
    data.feat_map_dims = {nt: feat.shape[-1] for nt, feat in zip(data.node_type, data.features)}

    build_distance_graph(args, data)
    build_subgraphs(data)
    build_filters(data)
    data.adjs=data.sym_adjs

    return data






