import torch
from torch import nn
import torch.nn.functional as F
from layers import FullyConnect, Linear_layer
from layers.GCN import GCN

class HeteroViewEncoder(nn.Module):
    """heterogeneous graph encoder."""
    def __init__(self, args, data,graph):
        super().__init__()
        self.args = args
        self.graph = graph
        self.data = data
        self.nt2idx = {nt: i for i, nt in enumerate(data.node_type)}
        self.nt_rel=data.nt_rel
        self.bnn = nn.ModuleDict()
        self.fc = nn.ModuleDict()
        self.hetero_layers=args.hetero_layers

        for t, rels in self.nt_rel.items():
            self.fc[t] = FullyConnect(args.hid_units2 + data.feat_map_dims[t], args.out_ft)
            for rel in rels:
                neighbor_type = rel.split('-')[-1]
                for layer_idx in range(self.hetero_layers):
                    in_dim, out_dim = self.layer_dims(layer_idx, data.feat_map_dims[neighbor_type])
                    self.bnn[f'{layer_idx}{rel}'] = Linear_layer(in_dim, out_dim, act=nn.ReLU(), isBias=False)

    def layer_dims(self, layer_idx, neighbor_feat_dim):
        is_first = (layer_idx == 0)
        is_last = (layer_idx == self.hetero_layers - 1)
        in_dim = neighbor_feat_dim if is_first else self.args.hid_units
        out_dim = self.args.hid_units2 if is_last else self.args.hid_units
        return in_dim, out_dim

    def aggregate(self, features):
        curr_feat_dict = {nt: features[self.nt2idx[nt]] for nt in self.data.node_type}

        for layer_idx in range(self.hetero_layers):
            next_feat_dict = {}
            for n, rels in self.nt_rel.items():
                vec = []
                for j, rel in enumerate(rels):
                    neighbor_type = rel.split('-')[-1]
                    mean_neighbor = torch.spmm(self.graph[n][j], curr_feat_dict[neighbor_type])
                    v = self.bnn[f'{layer_idx}{rel}'](mean_neighbor)
                    vec.append(v)
                next_feat_dict[n] = torch.mean(torch.stack(vec, 0), dim=0)
            curr_feat_dict = next_feat_dict

        return curr_feat_dict

    def forward(self, features):
        agg_dict = self.aggregate(features)
        embs = torch.zeros((self.data.node_size, self.args.out_ft)).to(self.args.device)
        for n in self.nt_rel.keys():
            v_cat = torch.hstack((agg_dict[n], features[self.nt2idx[n]]))
            embs[self.data.node_cnt[n]] = self.fc[n](v_cat)

        return embs[0:self.data.num_target]


class HomoViewEncoder(nn.Module):
    """homophily encoder"""
    def __init__(self, args,data, adjs):
        super().__init__()
        self.args = args
        self.adjs = adjs
        self.target_type = data.node_type[0]

        self.mp_gcns = nn.ModuleList([GCN(input_dim=data.feat_map_dims[self.target_type], output_dim=self.args.out_ft) for _ in adjs])

    def forward(self, features, distance, str_pos):
        x = features[0]
        embs_hom_list = []
        coe2 = 1.0 / self.args.beta

        for i, adj in enumerate(self.adjs):
            hp = self.mp_gcns[i](x, adj, nlayer=self.args.gcn_layers)
            res = torch.mm(torch.transpose(hp, 0, 1), hp)
            M = torch.eye(hp.shape[1], device=self.args.device) + coe2 * res
            res = torch.linalg.solve(M, res)
            res = coe2 * hp - coe2 * coe2 * torch.mm(hp, res)

            tmp = torch.mm(torch.transpose(hp, 0, 1), res)
            part1 = torch.mm(hp, tmp)
            part2 = (-self.args.alpha / 2) * torch.spmm(distance, res)
            part3 = (self.args.gamma / 2) * torch.spmm(str_pos, res)
            embs_hom_list.append(part1 + part2 + part3)

        return embs_hom_list


class HetViewEncoder(nn.Module):
    """heterophily encoder."""
    def __init__(self, args, data,hp_filters):
        super().__init__()
        self.args = args
        self.hp_filters = hp_filters
        input_dim = data.feat_map_dims[data.node_type[0]]
        output_dim = args.out_ft

        self.dropout = nn.Dropout(args.dropout_rate)
        self.W_hp = nn.ModuleList([nn.Linear(input_dim, output_dim, bias=False) for _ in hp_filters])
        self.ln = nn.ModuleList([nn.LayerNorm(output_dim) for _ in hp_filters])

    def forward(self, features):
        x = features[0]
        embs_het_list = []
        for i, hp_filter in enumerate(self.hp_filters):
            x_drop = self.dropout(x)
            hp_out = torch.sparse.mm(hp_filter, x_drop)
            hp_out = self.W_hp[i](hp_out)
            out = F.relu(hp_out)
            embs_het_list.append(self.ln[i](out))
        return embs_het_list


class MultiViewModeler(nn.Module):
    def __init__(self, args, data,adjs, graph, hp_filters):
        super().__init__()
        self.hetero_encoder = HeteroViewEncoder(args, data,graph)
        self.hom_encoder = HomoViewEncoder(args,data,adjs)
        self.het_encoder = HetViewEncoder(args,data,hp_filters)

    def forward(self, features, distance, str_pos):
        embs_hetero = self.hetero_encoder(features)
        embs_hom_list = self.hom_encoder(features, distance, str_pos)
        embs_het_list = self.het_encoder(features)
        return embs_hetero, embs_hom_list, embs_het_list

    def embed(self, features, distance, str_pos):
        embs_hetero, embs_hom_list, embs_het_list = self.forward(features, distance, str_pos)
        return (
            embs_hetero.detach(),
            [e.detach() for e in embs_hom_list],
            [e.detach() for e in embs_het_list],
        )
