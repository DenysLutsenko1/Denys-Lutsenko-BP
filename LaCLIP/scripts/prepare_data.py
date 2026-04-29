import sys
import os
sys.path.append(os.path.expanduser("~/work/python_libs"))

import argparse, pandas as pd, json


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_csv', type=str, required=True)
    ap.add_argument('--text_col', type=str, default='text')
    ap.add_argument('--label_col', type=str, default='label')
    ap.add_argument('--out_jsonl', type=str, required=True)