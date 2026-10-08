import argparse
import os
from datetime import datetime

import torch
from ruamel.yaml import YAML

from modules.graph_builder import GraphBuild
from modules.jolt import JOLT
from utils import process
from utils.logger import setup_logger


def get_args(model_name, dataset, custom_key="Node") -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=model_name)
    parser.add_argument("--custom_key", type=str, default=custom_key)
    parser.add_argument("--dataset", type=str, default=dataset, choices=["ACM", "DBLP", "IMDB", "Yelp"])
    parser.add_argument('--gpu_num', nargs='?', default='0')
    parser.add_argument('--load_parameter', action='store_true',default=False)

    parser.add_argument('--lr', type=float, default=0.0005, help="Learning rate")
    parser.add_argument('--patience', type=int, default=30, help="Early-stopping patience.")
    parser.add_argument('--nb_epochs', type=int, default=500, help="Maximum training epochs.")
    parser.add_argument('--hid_units', type=int, default=256, help="Hidden aggregation dimension.")
    parser.add_argument('--hid_units2', type=int, default=512, help="Final aggregation dimension.")
    parser.add_argument('--out_ft', type=int, default=512, help="Representation dimension.")
    parser.add_argument('--g_dim', type=int, default=256, help="Projection-head dimension.")
    parser.add_argument('--distance_quantile', type=float, default=0.9, help="Quantile used to retain large feature distances.")
    parser.add_argument('--gamma', type=float, default=0.1, help="Structural-prior weight.")
    parser.add_argument("--alpha", default=1.0, type=float, help="Feature-distance penalty weight.")
    parser.add_argument("--beta", default=1.0, type=float, help="Self-expressive regularization weight.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout rate for the heterophily encoder.")
    parser.add_argument("--het_layers", type=int, default=1, help="High-pass propagation layers.")
    parser.add_argument("--hetero_layers", type=int, default=2, help="Relation-specific message-passing layers.")
    parser.add_argument("--gcn_layers", type=int, default=1, help="Homophily encoder propagation layers.")
    parser.add_argument("--tau", default=0.5, type=float, help="Temperature parameter.")
    parser.add_argument('--K', type=float, default=0, help="Number of additional positive neighbors.")
    parser.add_argument('--theta', type=float, default=1, help="Homophily contrastive-loss weight.")
    parser.add_argument('--eta', type=float, default=0.5, help="Heterophily contrastive-loss weight.")
    parser.add_argument('--act', type=str, default='elu', help="Projection-head activation.")


    args = parser.parse_args()
    config_path = "config/args.yaml"

    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = YAML().load(f)
        config_key = f"{args.model_name}-{args.dataset}-{args.custom_key}"
        if config_key in config_dict:
            model_config = config_dict[config_key]
            for key, value in model_config.items():
                setattr(args, key, value)
    return args



def main():
    args = get_args(model_name="JOLT",
                    dataset="IMDB", #ACM、DBLP、IMDB、Yelp
                    custom_key="Node")
    process.setup_seed(args.seed)
    now = datetime.now()
    print(f"Seed: {args.seed},Time: {now:%Y-%m-%d %H:%M:%S}")

    args.gpu_num_ = int(args.gpu_num)
    if args.gpu_num_ == -1:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f"cuda:{args.gpu_num_}" if torch.cuda.is_available() else "cpu")

    logger = setup_logger(args.dataset)
    logger.info(f"Arguments: {args}")

    data = GraphBuild(args)
    trainer = JOLT(args, data, logger=logger)
    if not args.load_parameter:
        trainer.train()
    else:
        trainer.evaluate(f'best/{args.dataset}/best.pkl')

if __name__ == '__main__':
    main()
