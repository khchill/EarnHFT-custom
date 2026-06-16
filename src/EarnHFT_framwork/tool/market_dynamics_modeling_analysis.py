import pandas as pd
import numpy as np
import os
import argparse

class MarketDynamicsModelingAnalysis(object):
    def __init__(self, data_path, key_indicator):
        self.data_path = data_path
        self.key_indicator = key_indicator
        # lấy phần mở rộng của file
        if self.data_path:
            self.file_extension = os.path.splitext(self.data_path)[1][1:]

    def run_analysis(self, data_path):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--key_indicator", type=str, default="bid1_price")
    args = parser.parse_args()
    MarketDynamicsModelingAnalysis(args.data_path, args.key_indicator).run_analysis(args.data_path)
