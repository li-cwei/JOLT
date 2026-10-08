from collections import defaultdict
import os.path as osp

import numpy as np
import torch
import scipy.sparse as sp

from utils import process

np.random.seed(0)


def generate_metapath_adj(edge_dict, path_sequence):
    """Construct a meta-path adjacency matrix from relation matrices."""

    def get_mat(rel):
        mat = edge_dict[rel]
        return mat[0] if isinstance(mat, list) else mat

    res_adj = get_mat(path_sequence[0])

    for rel in path_sequence[1:]:
        next_adj = get_mat(rel)
        res_adj = res_adj.dot(next_adj)

    return res_adj

def generate_str_pos(mats, pos_num=50):
    M_size = mats[0].shape[0]
    all_mat = np.zeros((M_size, M_size), dtype=np.float32)

    for mat in mats:
        dense_mat = mat.toarray()
        row_sums = dense_mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-9
        norm_mat = dense_mat / row_sums
        all_mat += norm_mat

    pos_np = np.zeros((M_size, M_size), dtype=np.float32)

    for i in range(M_size):
        one = all_mat[i].nonzero()[0]
        if len(one) > pos_num:
            oo = np.argsort(-all_mat[i, one])
            sele = one[oo[:pos_num]]
            pos_np[i, sele] = 1
        else:
            pos_np[i, one] = 1

    return process.sparse_mx_to_torch_sparse_tensor(sp.csr_matrix(pos_np)).coalesce()

def load_ACM(args):
    path = "./dataset/ACM/"
    P, A, S = 4019, 7167, 60
    args.node_num = P

    feat_p = sp.load_npz(path + "p_feat.npz")
    feat_a = sp.load_npz(path + "a_feat.npz")
    pa = np.loadtxt(path + "pa.txt", dtype=int)
    ps = np.loadtxt(path + "ps.txt", dtype=int)
    pap = sp.load_npz(path + "pap.npz")
    psp = sp.load_npz(path + "psp.npz")

    adjs_raw = [pap, psp]
    adjs = [process.sparse_mx_to_torch_sparse_tensor(adj).coalesce() for adj in adjs_raw]

    edge_dict = defaultdict(list)
    edge_pa = sp.csr_matrix((np.ones(pa.shape[0]), (pa[:, 0], pa[:, 1])), shape=(P, A))
    edge_ps = sp.csr_matrix((np.ones(ps.shape[0]), (ps[:, 0], ps[:, 1])), shape=(P, S))

    edge_dict['p-a'].append(edge_pa)
    edge_dict['a-p'].append(edge_pa.T.tocsr())
    edge_dict['p-s'].append(edge_ps)
    edge_dict['s-p'].append(edge_ps.T.tocsr())


    feat_s = edge_dict['s-p'][0].dot(feat_p).toarray()
    feat_p = process.preprocess_features(feat_p, norm=True)
    feat_a = process.preprocess_features(feat_a, norm=True)
    feat_s = process.preprocess_features(feat_s, norm=True)

    node_rel = {'p': ['p-a', 'p-s'], 'a': ['a-p'], 's': ['s-p']}
    node_idx = {
        'p': torch.LongTensor(range(P)),
        'a': torch.LongTensor(range(P, P + A)),
        's': torch.LongTensor(range(P + A, P + A + S))
    }

    pos_sp = sp.load_npz(path + "pos.npz")
    pos = process.sparse_mx_to_torch_sparse_tensor(pos_sp).coalesce()

    labels = process.encode_onehot(np.load(path + "labels.npy").astype('int32'))
    ratio = [20, 40, 60]
    train_idx = [np.load(osp.join(path, f"train_{i}.npy")) for i in ratio]
    val_idx = [np.load(osp.join(path, f"val_{i}.npy")) for i in ratio]
    test_idx = [np.load(osp.join(path, f"test_{i}.npy")) for i in ratio]

    return [feat_p, feat_a, feat_s], adjs,node_rel, edge_dict, labels, train_idx, val_idx, test_idx, node_idx, pos


def load_IMDB(args):
    """Load the 4,278-movie IMDB benchmark."""
    path = "./dataset/IMDB/"
    M, D, A = 4278, 2081, 5257

    feat_m = np.load(path + "features_movie.npy").astype("float32")
    feat_d = sp.load_npz(path + "features_1.npz").astype("float32")
    feat_a = sp.load_npz(path + "features_2.npz").astype("float32")

    mam = sp.load_npz(path + "mam.npz")
    mdm = sp.load_npz(path + "mdm.npz")
    adjs = [
        process.sparse_mx_to_torch_sparse_tensor(mam).coalesce(),
        process.sparse_mx_to_torch_sparse_tensor(mdm).coalesce(),
    ]

    ma = np.load(path + "nei_a.npy")
    md = np.load(path + "nei_d.npy")
    edge_ma = sp.csr_matrix((np.ones(ma.shape[0]), (ma[:, 0], ma[:, 1])), shape=(M, A))
    edge_md = sp.csr_matrix((np.ones(md.shape[0]), (md[:, 0], md[:, 1])), shape=(M, D))

    edge_dict = defaultdict(list)
    edge_dict["m-a"].append(edge_ma)
    edge_dict["a-m"].append(edge_ma.T.tocsr())
    edge_dict["m-d"].append(edge_md)
    edge_dict["d-m"].append(edge_md.T.tocsr())

    node_rel = defaultdict(list)
    node_rel["m"].extend(["m-a", "m-d"])
    node_rel["d"].append("d-m")
    node_rel["a"].append("a-m")

    node_idx = {
        "m": torch.LongTensor(range(M)),
        "d": torch.LongTensor(range(M, M + D)),
        "a": torch.LongTensor(range(M + D, M + D + A)),
    }

    feat_m = process.preprocess_features(feat_m, norm=True)
    feat_d = process.preprocess_features(feat_d, norm=True)
    feat_a = process.preprocess_features(feat_a, norm=True)

    mam_topo = generate_metapath_adj(edge_dict, ["m-a", "a-m"])
    mdm_topo = generate_metapath_adj(edge_dict, ["m-d", "d-m"])
    pos = generate_str_pos([mam_topo, mdm_topo], pos_num=50)

    labels = np.load(path + "label.npy").astype("int32")
    labels = process.encode_onehot(labels)

    ratio = [20, 40, 60]
    train_idx = [np.load(osp.join(path, f"train_{i}.npy")) for i in ratio]
    val_idx = [np.load(osp.join(path, f"val_{i}.npy")) for i in ratio]
    test_idx = [np.load(osp.join(path, f"test_{i}.npy")) for i in ratio]

    return ([feat_m, feat_d, feat_a],adjs,node_rel,edge_dict,labels,train_idx,val_idx,test_idx,node_idx,pos,)


def load_DBLP(args):
    path = "./dataset/DBLP/"
    A, P, C, T = 4057, 14328, 20, 7723

    feat_a = sp.load_npz(path + "a_feat.npz").toarray()
    feat_p = sp.load_npz(path + "p_feat.npz").toarray()
    feat_c = np.eye(C, dtype=np.float32)
    feat_t = np.eye(T, dtype=np.float32)

    apa = sp.load_npz(path + "apa.npz")
    apcpa = sp.load_npz(path + "apcpa.npz")
    aptpa = sp.load_npz(path + "aptpa.npz")
    adjs_raw = [apa, apcpa, aptpa]
    adjs = []
    for adj in adjs_raw:
        adjs.append(process.sparse_mx_to_torch_sparse_tensor(adj).coalesce())

    pa = np.loadtxt(path + "pa.txt", dtype=int)
    pc = np.loadtxt(path + "pc.txt", dtype=int)
    pt = np.loadtxt(path + "pt.txt", dtype=int)
    edge_pa = sp.csr_matrix((np.ones(pa.shape[0]), (pa[:, 0], pa[:, 1])), shape=(P, A))
    edge_pc = sp.csr_matrix((np.ones(pc.shape[0]), (pc[:, 0], pc[:, 1])), shape=(P, C))
    edge_pt = sp.csr_matrix((np.ones(pt.shape[0]), (pt[:, 0], pt[:, 1])), shape=(P, T))

    edge_dict = defaultdict(list)
    edge_dict['p-a'].append(edge_pa)
    edge_dict['a-p'].append(edge_pa.T.tocsr())
    edge_dict['p-c'].append(edge_pc)
    edge_dict['c-p'].append(edge_pc.T.tocsr())
    edge_dict['p-t'].append(edge_pt)
    edge_dict['t-p'].append(edge_pt.T.tocsr())
    feat_a = process.preprocess_features_rownorm(feat_a, norm=True)
    feat_p = process.preprocess_features_rownorm(feat_p, norm=True)
    feat_c = process.preprocess_features_rownorm(feat_c, norm=True)
    feat_t = process.preprocess_features_rownorm(feat_t, norm=True)

    node_rel = defaultdict(list)
    node_rel['a'].append('a-p')
    node_rel['p'].extend(['p-a', 'p-c'])
    node_rel['c'].append('c-p')

    node_idx = {
        'a': torch.LongTensor(range(A)),
        'p': torch.LongTensor(range(A, A + P)),
        'c': torch.LongTensor(range(A + P, A + P + C))
    }
    pos_sp = sp.load_npz(path + "pos.npz")
    pos = torch.FloatTensor(pos_sp.toarray())

    labels = np.load(path + "labels.npy").astype('int32')
    labels = process.encode_onehot(labels)

    ratio = [20, 40, 60]
    train_idx = [np.load(osp.join(path, f"train_{i}.npy")) for i in ratio]
    val_idx = [np.load(osp.join(path, f"val_{i}.npy")) for i in ratio]
    test_idx = [np.load(osp.join(path, f"test_{i}.npy")) for i in ratio]

    return [feat_a, feat_p, feat_c, feat_t],adjs, node_rel, edge_dict, labels, train_idx, val_idx, test_idx, node_idx,pos

def load_Yelp(args):
    path = "./dataset/Yelp/"

    feat_b = sp.load_npz(path + "features_0.npz").astype("float32")
    feat_u = sp.load_npz(path + "features_1.npz").astype("float32")
    feat_s = sp.load_npz(path + "features_2.npz").astype("float32")
    feat_l = sp.load_npz(path + "features_3.npz").astype("float32")

    B, U, S, L = feat_b.shape[0], feat_u.shape[0], feat_s.shape[0], feat_l.shape[0]
    args.node_num = B

    bub = sp.csr_matrix(np.load(path + "bub.npy").astype("float32"))
    bsb = sp.csr_matrix(np.load(path + "bsb.npy").astype("float32"))
    blb = sp.csr_matrix(np.load(path + "blb.npy").astype("float32"))

    adjs_raw = [bub, bsb, blb]
    adjs = []
    for adj in adjs_raw:
        adjs.append(process.sparse_mx_to_torch_sparse_tensor(adj).coalesce())

    nei_u = np.load(path + "nei_u.npy", allow_pickle=True)
    nei_s = np.load(path + "nei_s.npy", allow_pickle=True)
    nei_l = np.load(path + "nei_l.npy", allow_pickle=True)

    edge_bu = sp.csr_matrix((np.ones(len(nei_u[:,0])), (nei_u[:,0],nei_u[:,1])), shape=(B, U))
    edge_bs = sp.csr_matrix((np.ones(len(nei_s[:,0])), (nei_s[:,0],nei_s[:,1])), shape=(B, S))
    edge_bl = sp.csr_matrix((np.ones(len(nei_l[:,0])), (nei_l[:,0],nei_l[:,1])), shape=(B, L))

    edge_dict = defaultdict(list)
    edge_dict['b-u'] = edge_bu
    edge_dict['u-b'] = edge_bu.T.tocsr()
    edge_dict['b-s'] = edge_bs
    edge_dict['s-b'] = edge_bs.T.tocsr()
    edge_dict['b-l'] = edge_bl
    edge_dict['l-b'] = edge_bl.T.tocsr()

    feat_b = process.preprocess_features_rownorm(feat_b, norm=True)
    feat_u = process.preprocess_features_rownorm(feat_u, norm=True)
    feat_s = process.preprocess_features_rownorm(feat_s, norm=True)
    feat_l = process.preprocess_features_rownorm(feat_l, norm=True)

    node_rel = {
        'b': ['b-u', 'b-s', 'b-l'],
        'u': ['u-b'],
        's': ['s-b'],
        'l': ['l-b']
    }
    node_idx = {
        'b': torch.LongTensor(range(B)),
        'u': torch.LongTensor(range(B, B + U)),
        's': torch.LongTensor(range(B + U, B + U + S)),
        'l': torch.LongTensor(range(B + U + S, B + U + S + L))
    }
    bub_topo = generate_metapath_adj(edge_dict, ['b-u', 'u-b'])
    bsb_topo = generate_metapath_adj(edge_dict, ['b-s', 's-b'])
    blb_topo = generate_metapath_adj(edge_dict, ['b-l', 'l-b'])

    adjs_topo_raw = [bub_topo, bsb_topo, blb_topo]
    pos = generate_str_pos(adjs_topo_raw, pos_num=50)
    labels = process.encode_onehot(np.load(path + "labels.npy").astype('int32'))
    ratio = [20, 40, 60]
    train_idx = [np.load(osp.join(path, f"train_{i}.npy")) for i in ratio]
    val_idx = [np.load(osp.join(path, f"val_{i}.npy")) for i in ratio]
    test_idx = [np.load(osp.join(path, f"test_{i}.npy")) for i in ratio]

    return [feat_b, feat_u, feat_s, feat_l], adjs, node_rel, edge_dict, labels, train_idx, val_idx, test_idx, node_idx, pos

