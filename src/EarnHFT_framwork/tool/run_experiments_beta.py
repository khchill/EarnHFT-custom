import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Chay thu nghiem voi 4 he so beta: -90, -10, 30, 100")
    parser.add_argument("--epochs", type=int, default=50, help="So epoch huan luyen")
    parser.add_argument("--dataset", type=str, default="BTCUSDT", help="Ten dataset")
    parser.add_argument("--quick", action="store_true", help="Chay nhanh 1 epoch")
    args = parser.parse_args()

    betas = [-90.0, -10.0, 30.0, 100.0]
    epochs_to_run = 1 if args.quick else args.epochs

    print("=" * 60)
    print(f"chay test may con beta nha: {betas}")
    print(f"[*] So epoch: {epochs_to_run} | Cặp: {args.dataset}")
    print("=" * 60)

    for beta in betas:
        model_save_path = f"result_risk/{args.dataset}/beta_{beta}/seed_12345/model.pth"
        
        print(f"\nhuan luyen voi Beta = {beta}")
        print(f"Luu tai: {model_save_path}")
        
        cmd = [
            ".venv/bin/python",
            "src/EarnHFT_framwork/RL/agent/low_level/ddqn_pes_risk_aware.py",
            "--beta", str(beta),
            "--epochs", str(epochs_to_run),
            "--model_save_path", model_save_path
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"xong beta = {beta} roi")
        except subprocess.CalledProcessError as e:
            print(f"[x] Loi khi chay Beta = {beta}: {e}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("xong test cac beta ")
    print("=" * 60)

if __name__ == "__main__":
    main()
