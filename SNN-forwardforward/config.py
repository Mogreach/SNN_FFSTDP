"""
====================================================================
File          : config.py
Description   : SNN-FF训练参数设置
Author        : Morgreach
Version       : 1.0.0
Date          : 2025-04-18
contact       : 1245598043@qq.com
License       : MIT
====================================================================
"""
import argparse
class ConfigParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="CNN/Fashion/tdLN/no IF node/delta loss")
        # argparse
        self.parser.add_argument(
            "-model",
            type=str,
            default="MLP",
            choices=["CNN", "MLP", "VGG6", "VGG8", "VGG11", "ResNet"],
            help="Network architecture type"
        )
        self.parser.add_argument(
            "-dataset",
            default="MNIST",
            type=str,
            choices=["MNIST", "N-MNIST", "NMNIST", "FashionMNIST", "CIFAR10", "DVS128Gesture"],
            help="Train dataset",
        )
        self.parser.add_argument(
            "-conv_cfg",
            default=[
                # in_ch, out_ch, k, s, p
                (1, 16, 3, 1, 1),
                (16, 32, 3, 1, 1),
                (32, 64, 3, 1, 1),
            ],
            help="configuration of convolutional layers for the generic CNN model",
            type=eval,
        )
        self.parser.add_argument(
            "-dims",
            default=[784, 256, 10],
            help="dimension of the MLP network",
            type=int,
            nargs="+",
        )
        self.parser.add_argument(
            "-T", default=8, type=int, help="simulating time-steps"
        )
        self.parser.add_argument("-device", default="cuda:0", help="device")
        self.parser.add_argument("-b", default=500,type=int, help="batch size")
        self.parser.add_argument(
            "-epochs",
            default=200,
            type=int,
            metavar="N",
            help="number of total epochs to run",
        )
        self.parser.add_argument(
            "-j",
            default=8,
            type=int,
            metavar="N",
            help="number of data loading workers (default: 4)",
        )
        self.parser.add_argument(
            "-data-dir", default="./SNN-forwardforward/data", type=str, help="root dir of MNIST dataset"
        )
        self.parser.add_argument(
            "-out-dir",
            type=str,
            default="./SNN-forwardforward/logs",
            help="root dir for saving logs and checkpoint",
        )
        self.parser.add_argument(
            "-resume", type=str, help="resume from the checkpoint path"
        )
        self.parser.add_argument(
            "-amp", action="store_true", help="automatic mixed precision training"
        )
        self.parser.add_argument(
            "-opt",
            type=str,
            choices=["sgd", "adam"],
            default="adam",
            help="use which optimizer",
        )
        self.parser.add_argument(
            "-momentum", default=0.9, type=float, help="momentum for SGD"
        )
        self.parser.add_argument(
            "-lr", default=0.015625, type=float, help="learning rate"
        )
        self.parser.add_argument(
            "-tau", default=2.0, type=float, help="parameter tau of LIF neuron"
        )
        self.parser.add_argument(
            "-v_threshold", default=1.2, type=float, help="V_threshold of LIF neuron"
        )
        self.parser.add_argument(
            "-v_threshold_neg", default=-1.0, type=float, help="V_threshold of LIF neuron"
        )
        self.parser.add_argument(
            "-loss_threshold",
            default=0.25,
            type=float,
            help="threshold of loss function. orignal loss threshold is 0.25. delta loss threshold is 8",
        )
        self.parser.add_argument(
            "-learning_mode",
            default="unsupervised",
            type=str,
            choices=["unsupervised", "supervised"],
            help="Learning mode: supervised or unsupervised.",
        )
        self.parser.add_argument(
            "-hidden_layer_update_mode",
            default="autograd",
            type=str,
            choices=["autograd", "manual"],
            help="Update rule used by hidden layers in both supervised and unsupervised FF-SNN training.",
        )
        self.parser.add_argument(
            "-capture_manual_grad_metrics",
            dest="capture_manual_grad_metrics",
            action="store_true",
            help="Capture manual-gradient profiling metrics.",
        )
        self.parser.add_argument(
            "-no-capture_manual_grad_metrics",
            dest="capture_manual_grad_metrics",
            action="store_false",
            help="Disable manual-gradient profiling metrics.",
        )
        self.parser.add_argument(
            "-capture_autograd_comparison",
            dest="capture_autograd_comparison",
            action="store_true",
            help="Capture autograd comparison metrics for unsupervised hidden layers.",
        )
        self.parser.add_argument(
            "-no-capture_autograd_comparison",
            dest="capture_autograd_comparison",
            action="store_false",
            help="Disable autograd comparison metrics.",
        )
        self.parser.set_defaults(
            capture_manual_grad_metrics=True,
            capture_autograd_comparison=True,
        )
        self.parser.add_argument(
            "-save-model", action="store_true", help="save the model or not"
        )

    def parse(self):
        return self.parser.parse_args()


# 示例用法
# if __name__ == "__main__":
#     config = ConfigParser()
#     args = config.parse()
#     print(args)
