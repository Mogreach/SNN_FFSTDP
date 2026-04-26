"""
====================================================================
File          : ff-snn.py
Description   : SNN-FF training entry point
Author        : Morgreach
Version       : 1.0.0
Date          : 2025-04-18
contact       : 1245598043@qq.com
License       : MIT
====================================================================
"""
from config import ConfigParser
from src.experiment_runner import run_experiment

def compute_test_acc(net, test_data_loader, device):
    test_acc = 0
    test_count = 0
    with torch.no_grad():
        for x_te, y_te in test_data_loader:
            test_count += 1
            x_te, y_te = x_te.to(device), y_te.to(device)
            test_acc += net.predict_winner(x_te).eq(y_te).cpu().float().mean().item()
    if test_count == 0:
        return 0.0
    return 100 * test_acc / test_count

def _mean_last_epoch(metric_per_layer_list):
    values = []
    for layer_list in metric_per_layer_list:
        if not layer_list:
            continue
        v = layer_list[-1]
        if torch.is_tensor(v):
            if v.numel() == 1:
                v = v.detach().cpu().item()
            else:
                v = v.detach().cpu().mean().item()
        elif hasattr(v, "item"):
            v = v.item()
        values.append(float(v))
    if not values:
        return None
    return float(np.mean(values))

def main():
    args = ConfigParser().parse()
    run_experiment(args)


if __name__ == "__main__":
    main()
