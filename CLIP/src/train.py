from __future__ import annotations

import sys
import os
sys.path.append(os.path.expanduser("~/work/python_libs"))

import os, argparse, yaml, joblib, sys
from rich import print
from sklearn.metrics import classification_report, roc_auc_score, f1_score

from src.datasets import load_jsonl, texts_and_labels
from src.encoders import TextEncoder
from src.models import make_clf
from src.utils import set_seed, get_device, ensure_dir

def main(cfg):
    try:
        print("[bold cyan]=== Start training (text-only) ===[/bold cyan]")
        print(f"[bold]Working dir:[/bold] {os.getcwd()}")
        print(f"[bold]Config:[/bold] {cfg}")

        set_seed(cfg.get('random_seed', 42))
        device = get_device(cfg['encoder'].get('device', 'auto'))
        print(f"[bold green]Device:[/bold green] {device}")

        # --- Load data
        print("[yellow]Loading data...[/yellow]")
        train_path = cfg['paths']['train']
        val_path   = cfg['paths']['val']
        print(f" train: {train_path}")
        print(f" val  : {val_path}")
        train = load_jsonl(train_path)
        val   = load_jsonl(val_path)
        print(f" loaded: train={len(train)} records, val={len(val)} records")
        if len(train) == 0 or len(val) == 0:
            raise RuntimeError("Empty train/val data — проверь файлы JSONL")

        Xtr_texts, ytr = texts_and_labels(train)
        Xva_texts, yva = texts_and_labels(val)

        # --- Encode
        print("[yellow]Loading encoder and encoding texts...[/yellow]")
        enc = TextEncoder(
            cfg['encoder']['name'], device,
            max_length=cfg['encoder']['max_length'],
            pool=cfg['features'].get('pool','mean')
        )
        Xtr = enc.encode(Xtr_texts, batch_size=cfg['encoder']['batch_size'])
        Xva = enc.encode(Xva_texts, batch_size=cfg['encoder']['batch_size'])
        print(f" embeddings: Xtr={Xtr.shape}, Xva={Xva.shape}")

        # --- Train classifier
        print("[yellow]Training classifier...[/yellow]")
        clf = make_clf(
            cfg['train']['clf_type'],
            normalize=cfg['features']['normalize'],
            lr=cfg['train']['lr'],
            weight_decay=cfg['train']['weight_decay'],
            epochs=cfg['train']['epochs']
        )
        clf.fit(Xtr, ytr)

        # --- Validate
        print("[yellow]Validating...[/yellow]")
        prob = clf.predict_proba(Xva)[:,1]
        pred = (prob >= 0.5).astype(int)
        try:
            auc = roc_auc_score(yva, prob)
        except Exception as e:
            print(f"[red]AUC error:[/red] {e}")
            auc = None
        f1 = f1_score(yva, pred)
        print(f"AUC: {auc}")
        print(f"F1 : {f1}")
        print(classification_report(yva, pred, digits=4))

        # --- Save
        outdir = cfg['paths']['outdir']
        ensure_dir(outdir)
        abs_outdir = os.path.abspath(outdir)
        outpath = os.path.join(abs_outdir, 'text_clf.joblib')
        bundle = {
            'pipeline': clf,
            'encoder_name': cfg['encoder']['name'],
            'max_length': cfg['encoder']['max_length'],
            'pool': cfg['features'].get('pool','mean')
        }
        joblib.dump(bundle, outpath)
        print(f"[bold cyan]Saved ->[/bold cyan] {outpath}")

        print("[bold green]=== Done ===[/bold green]")
    except Exception as e:
        print(f"[bold red]Training failed:[/bold red] {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    main(cfg)
