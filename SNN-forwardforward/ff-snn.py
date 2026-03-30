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


def main():
    args = ConfigParser().parse()
    run_experiment(args)


if __name__ == "__main__":
    main()
