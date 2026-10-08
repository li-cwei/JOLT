import os
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import warnings
from sklearn.metrics import f1_score, roc_auc_score, normalized_mutual_info_score, adjusted_rand_score, pairwise
from torch import nn, softmax

warnings.filterwarnings("ignore")


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0.0)

    def forward(self, seq):
        return self.fc(seq)

def evaluate_probing(embeds, train_idx, val_idx, test_idx, labels, nb_classes, device, ratio,logger=None):
    if torch.is_tensor(embeds):
        embeds = embeds.detach().clone().float()
    else:
        embeds = torch.FloatTensor(embeds)

    if not torch.is_tensor(labels):
        labels = torch.LongTensor(labels)

    embeds = embeds.to(device)
    labels = labels.to(device)
    if len(labels.shape) > 1:
        if labels.shape[1] > 1:
            labels = torch.argmax(labels, dim=1)
        else:
            labels = labels.squeeze()

    labels = labels.long()
    hid_units = embeds.shape[1]
    xent = nn.CrossEntropyLoss()

    train_embs, train_lbls = embeds[train_idx], labels[train_idx]
    val_embs, val_lbls = embeds[val_idx], labels[val_idx]
    test_embs, test_lbls = embeds[test_idx], labels[test_idx]

    micro_f1s, macro_f1s, auc_score_list = [], [], []

    for _ in range(5):
        log = LogReg(hid_units, nb_classes).to(device)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        val_macro_f1s = []
        test_micro_f1s, test_macro_f1s = [], []
        logits_list = []

        for iter_ in range(250):
            log.train()
            opt.zero_grad()
            logits = log(train_embs)
            loss = xent(logits, train_lbls)
            loss.backward()
            opt.step()

            log.eval()
            with torch.no_grad():
                val_logits = log(val_embs)
                val_preds = torch.argmax(val_logits, dim=1)
                val_macro_f1s.append(f1_score(val_lbls.cpu(), val_preds.cpu(), average='macro'))
                test_logits = log(test_embs)
                test_preds = torch.argmax(test_logits, dim=1)
                test_macro_f1s.append(f1_score(test_lbls.cpu(), test_preds.cpu(), average='macro'))
                test_micro_f1s.append(f1_score(test_lbls.cpu(), test_preds.cpu(), average='micro'))
                logits_list.append(test_logits)

        max_iter = val_macro_f1s.index(max(val_macro_f1s))

        macro_f1s.append(test_macro_f1s[max_iter] * 100)
        micro_f1s.append(test_micro_f1s[max_iter] * 100)

        best_logits = logits_list[max_iter]
        best_proba = softmax(best_logits, dim=1)
        auc_score_list.append(roc_auc_score(y_true=test_lbls.cpu().numpy(), y_score=best_proba.cpu().numpy(), multi_class='ovr') * 100)

    mac_mean = float(np.mean(macro_f1s))
    mic_mean = float(np.mean(micro_f1s))
    auc_mean = float(np.mean(auc_score_list))
    mac_std = float(np.std(macro_f1s))
    mic_std = float(np.std(micro_f1s))
    auc_std = float(np.std(auc_score_list))
    logger.info(f"\t[Ratio {ratio:2d}] Macro-F1: {mac_mean:.2f} ± {mac_std:.2f}  |  Micro-F1: {mic_mean:.2f} ± {mic_std:.2f}  |  AUC: {auc_mean:.2f} ± {auc_std:.2f}")



def run_kmeans(x, y, k, dataset, logger):
    """Run KMeans ten times and return mean NMI and ARI scores."""
    NMI_list = []
    ARI_list = []
    for random_state in range(10):
        estimator = KMeans(n_clusters=k,random_state=random_state)
        estimator.fit(x)
        y_pred = estimator.labels_

        n1 = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        a1 = adjusted_rand_score(y, y_pred)
        NMI_list.append(n1)
        ARI_list.append(a1)

    nmi = sum(NMI_list) / len(NMI_list)
    ari = sum(ARI_list) / len(ARI_list)

    nmi_score = np.round(nmi * 100, 2)
    ari_score = np.round(ari * 100, 2)

    logger.info('\t[Clustering] NMI: {:.2f}   ARI: {:.2f}'.format(nmi_score, ari_score))
    return nmi_score, ari_score


def run_similarity_search(test_embs, test_lbls, logger=None):
    """Compute Recall@5 and Recall@10 for nearest-neighbor labels."""

    cos_sim = pairwise.cosine_similarity(test_embs)

    np.fill_diagonal(cos_sim, -1.0)

    results = []
    for k in [5, 10]:
        topk_indices = np.argsort(cos_sim, axis=1)[:, -k:]

        topk_labels = test_lbls[topk_indices]  # shape [N, k]

        anchor_labels = test_lbls[:, np.newaxis]

        matches = np.sum(topk_labels == anchor_labels, axis=1)
        recall_k = np.mean(matches / k)
        results.append(recall_k)

    result_str = f"Recall@5: {results[0]:.4f} | Recall@10: {results[1]:.4f}"
    if logger:
        logger.info(f"\t[Similarity Search] {result_str}")
    else:
        print(f"\t[Similarity Search] {result_str}")

    return results
