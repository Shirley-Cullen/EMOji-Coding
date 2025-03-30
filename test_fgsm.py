import argparse, os
from sklearn.metrics import r2_score
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
from mosei_dataset import Mosei_Dataset
from meld_dataset import Meld_Dataset
from model_LA import Model_LA
from model_LAV import Model_LAV
from train import evaluate
import numpy as np

import json


def test_fgsm(net, test_loader, args, perturb=0.1):
    from utils.compute_args import LossFuncClass

    all_preds = []
    all_ans = []
    loss_fn = LossFuncClass[args.loss_fn]
    for step, (id, x, y, z, ans) in enumerate(test_loader):
        net.train()
        x, y, z, ans = x.cuda(), y.cuda(), z.cuda(), ans.cuda()
        y.requires_grad = True
        z.requires_grad = True
        pred = net(x, y, z)

        ans = ans.float() if args.proj == "regression" else ans.long()
        loss = loss_fn(pred, ans)
        loss.backward()

        net.eval()
        y_ = y.detach().clone()
        y.requires_grad = False
        y += y.grad.sign() * perturb
        y[y_ == 0] = 0

        if z.grad is not None:
            z_ = z.detach().clone()
            z.requires_grad = False
            z += z.grad.sign() * perturb
            z[z_ == 0] = 0
        pred = net(x, y, z)

        all_preds += pred.detach().cpu().tolist()
        all_ans += ans.cpu().tolist()

    breakpoint()
    if all_preds:
        all_preds = np.array(all_preds)
        all_ans = np.array(all_ans)

        # MAE
        all_preds = all_preds.round().clip(0, 6).astype(int)
        mae = np.mean(np.abs(all_preds - all_ans))

        r2 = r2_score(all_ans, all_preds)

        # Pearson correlation coefficient.
        # Flatten arrays in case predictions/answers have extra dimensions.
        pearson_corr = np.corrcoef(all_preds.flatten(), all_ans.flatten())[0, 1]

        accuracy = (all_preds == np.array(all_ans)).mean() * 100

        metrics = {
            "MAE": mae,
            "R2": r2,
            "Pearson": pearson_corr,
            "Accuracy": accuracy,
        }
        return metrics


if __name__ == "__main__":
    # Load the args from the json file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir", type=str, default="exp/LA_reg_head_4_layer_4_bs_256_lr_0.0001"
    )
    args_ = parser.parse_args()
    config_path = os.path.join(args_.exp_dir, "config.json")
    with open(config_path, "r") as f:
        args = json.load(f)
    args = SimpleNamespace(exp_dir=args_.exp_dir, **args)

    # DataLoader
    token_to_ix = ""
    token_to_ix = json.load(open("data/MOSEI/token_to_ix.json"))
    test_dset = eval(args.dataloader)("test", args, token_to_ix)
    test_loader = DataLoader(test_dset, args.batch_size, num_workers=8, pin_memory=True)

    # Net
    pretrained_emb = np.load("data/MOSEI/pretrained_emb.npy")
    net = eval(args.model)(args, args.vocab_size, pretrained_emb).cuda()
    net.load_state_dict(
        torch.load(os.path.join(args.exp_dir, "checkpoints", "best.pkl"))["state_dict"]
    )

    # Run test
    test_metric = test_fgsm(net, test_loader, args)

    # Save the results
    test_dir = os.path.join(args.exp_dir, "test")
    os.makedirs(test_dir, exist_ok=True)
    # Save as json
    with open(os.path.join(test_dir, "test_fgsm_result.json"), "w") as f:
        json.dump(test_metric, f)
