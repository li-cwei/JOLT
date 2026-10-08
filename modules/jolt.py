import itertools
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from modules import evaluate_probing, run_kmeans, run_similarity_search

from modules import Contrast, generate_pos_mask
from modules import MultiViewModeler
from utils.logger import setup_logger
from utils.process import get_activation

class JOLT:
    def __init__(self, args, data, logger=None):
        self.args = args
        self.data = data

        if logger is None:
            self.logger = setup_logger(dataset_name=self.args.dataset)
        else:
            self.logger = logger

        self.features = [feat.to(self.args.device) for feat in data.features]
        self.graph = {t: [m.to(self.args.device) for m in ms] for t, ms in data.graph.items()}
        self.adjs = [adj.to(self.args.device) for adj in data.adjs]
        self.feature_distance = data.feature_distance.to(self.args.device).detach()
        self.str_pos = data.str_pos.to(self.args.device)
        self.hp_filters = [f.to(self.args.device) for f in data.hp_filters]

        self.model = MultiViewModeler(self.args,self.data,self.adjs,self.graph,self.hp_filters).to(self.args.device)

        self.g_hom = nn.ModuleList([self.build_proj() for _ in range(len(self.adjs))]).to(self.args.device)
        self.g_hp = nn.ModuleList([self.build_proj() for _ in range(len(self.adjs))]).to(self.args.device)

        self.contrast = Contrast(self.args.tau).to(self.args.device)

        self._ckpt_path = f'checkpoints/{self.args.dataset}/best.pkl'
        os.makedirs(os.path.dirname(self._ckpt_path) or ".", exist_ok=True)

    def build_proj(self):
        act = get_activation(self.args.act)
        if self.args.dataset == 'ACM':
            return nn.Sequential(nn.Linear(self.args.out_ft, self.args.g_dim, bias=False), act)
        return nn.Sequential(nn.Linear(self.args.out_ft, self.args.g_dim, bias=False), nn.BatchNorm1d(self.args.g_dim), act, nn.Linear(self.args.g_dim, self.args.g_dim, bias=False))

    def set_opti(self):
        params = [
            self.model.parameters(),
        ]
        if self.args.dataset in ['ACM','DBLP']:
            params.append(self.g_hom.parameters())
            params.append(self.g_hp.parameters())
        return itertools.chain(*params)

    def train(self):
        cnt_wait = 0
        best = 1e9
        optimizer = torch.optim.AdamW(self.set_opti(), lr=self.args.lr)
        nb_epochs = int(self.args.nb_epochs)
        pbar = tqdm(range(nb_epochs))
        for epoch in pbar:
            self.model.train()
            optimizer.zero_grad()

            embs_hetero, embs_hom_list, embs_het_list = self.model(self.features, self.feature_distance, self.str_pos)
            loss_hom = 0.0
            loss_het = 0.0

            for i in range(len(self.adjs)):
                pos_mask = generate_pos_mask(embs_hom_list[i], k=self.args.K)
                z_hetero_hom = self.g_hom[i](embs_hetero)
                z_hom = self.g_hom[i](embs_hom_list[i])
                loss_1 = self.contrast(z_hom, z_hetero_hom, mask=pos_mask)
                loss_hom += loss_1

                z_hetero_het = self.g_hp[i](embs_hetero)
                z_het = self.g_hp[i](embs_het_list[i])
                loss_2 = self.contrast(z_het, z_hetero_het, mask=pos_mask)
                loss_het += loss_2

            total_loss = self.args.theta * loss_hom + self.args.eta * loss_het
            train_loss = total_loss.item()
            total_loss.backward()
            optimizer.step()

            if train_loss < best:
                best = train_loss
                cnt_wait = 0
                torch.save(self.model.state_dict(), self._ckpt_path)
            else:
                cnt_wait += 1
                if cnt_wait == self.args.patience:
                    tqdm.write("Early stopping triggered.")
                    break

        return self.evaluate(self._ckpt_path)

    def evaluate(self,path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

        with torch.no_grad():
            embs_hetero, _, _ = self.model.embed(self.features, self.feature_distance, self.str_pos)
            h_concat = embs_hetero
            test_out = h_concat.detach().cpu().numpy()

        ratios = [20, 40, 60]
        for i, ratio in enumerate(ratios):
            evaluate_probing(test_out, self.data.train_idx[i], self.data.val_idx[i], self.data.test_idx[i], self.data.labels, self.data.nb_classes, self.args.device, ratio, logger=self.logger)

        label = np.argmax(self.data.labels, axis=-1)
        run_kmeans(x=test_out, y=label, k=self.data.nb_classes, dataset=self.args.dataset, logger=self.logger)
