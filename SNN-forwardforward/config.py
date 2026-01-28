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
            default="CNN",
            choices=["CNN", "MLP"],
            help="Network architecture type"
        )
        self.parser.add_argument(
            "-dataset", default="FashionMNIST", type=str, choices=["MNIST","FashionMNIST","CIFAR10"],help="Train dataset"
        )
        if self.parser.parse_known_args()[0].model == "CNN":
            self.parser.add_argument(
                "-conv_cfg",
                default = [
                            # in_ch, out_ch, k, s, p
                            (1,  16, 3, 1, 1),
                            (16, 32, 3, 1, 1),
                            (32, 64, 3, 1, 1),
                        ],
                help="configuration of convolutional layers: (in_channels, out_channels, kernel_size, stride, padding)",
                type=eval,
            )
        elif self.parser.parse_known_args()[0].model == "MLP":
            self.parser.add_argument(
                "-dims",
                default=[784,512,512,10],
                help="dimension of the MLP network",
                type=int,
                nargs="+",
            )
        self.parser.add_argument(
            "-T", default=16, type=int, help="simulating time-steps"
        )
        self.parser.add_argument("-device", default="cuda:0", help="device")
        self.parser.add_argument("-b", default=512,type=int, help="batch size")
        self.parser.add_argument(
            "-epochs",
            default=20,
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
            "-lr", default=0.015625/4, type=float, help="learning rate"
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
            default=4,
            type=float,
            help="threshold of loss function. orignal loss threshold is 0.25. delta loss threshold is 8",
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
