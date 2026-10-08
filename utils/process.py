from sklearn.preprocessing import normalize, OneHotEncoder
import scipy.sparse as sp
import torch as th
import torch
import os
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F

def get_activation(name):
    """Return an activation module by name."""
    if name == 'relu':
        activation = nn.ReLU()
    elif name == 'elu':
        activation = nn.ELU()
    elif name == 'leaky_relu':
        activation = nn.LeakyReLU()
    elif name == 'prelu':
        activation = nn.PReLU()
    elif name == 'gelu':
        activation = nn.GELU()
    elif name == 'selu':
        activation = nn.SELU()
    elif name == 'tanh':
        activation = nn.Tanh()
    elif name == 'sigmoid':
        activation = nn.Sigmoid()
    elif name == 'none':
        activation = nn.Identity()
    else:
        raise ValueError(f"Unsupported activation: {name}")
    return activation

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj_symmetric(adj):
    """Symmetrically normalize an adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def sparse_to_tuple(mx):
#     mx = normalize_adj(mx)
    if not sp.isspmatrix_coo(mx):
        mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col))
    values = mx.data
    shape = mx.shape
    return coords, values, shape


def preprocess_features(features, norm=True):
    """Row-normalize feature matrix and convert to tuple representation"""
    if sp.issparse(features):
        features = np.array(features.todense())
    if norm:
        features[features>0] = 1
        # rowsum = np.array(features.sum(1))
        # r_inv = np.power(rowsum, -1.0).flatten()
        # r_inv[np.isinf(r_inv)] = 0.
        # r_mat_inv = sp.diags(r_inv)
        # features = r_mat_inv.dot(features)
    return th.FloatTensor(features)

def preprocess_features_rownorm(features, norm=True):
    """Row-normalize feature matrix and convert to tuple representation"""
    if sp.issparse(features):
        features = features.toarray()
    if norm:
        features[features>0] = 1
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1.0).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
    return th.FloatTensor(features)


def normalize_mx(mx, diagonal=True):
    if diagonal:
        size = mx.shape[0]
        return normalize(mx+sp.eye(size), norm='l1', axis=1)
    else:
        return normalize(mx, norm='l1', axis=1)



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def pairwise_distance(x, y=None):
    x = x.unsqueeze(0).permute(0, 2, 1)
    if y is None:
        y = x
    y = y.permute(0, 2, 1) # [B, N, f]
    A = -2 * th.bmm(y, x) # [B, N, N]
    A += th.sum(y**2, dim=2, keepdim=True) # [B, N, 1]
    A += th.sum(x**2, dim=1, keepdim=True) # [B, 1, N]
    return A.squeeze()

def setup_seed(seed=0):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def random_aug(x, feat_drop_rate):
    feat = drop_feature(x, feat_drop_rate)

    return feat

def drop_feature(x, drop_prob):
    drop_mask = th.empty((x.size(1),), dtype=th.float32, device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def mask_edge(graph, mask_prob):
    E = graph.number_of_edges()

    mask_rates = th.FloatTensor(np.ones(E) * mask_prob)
    masks = th.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx

def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


