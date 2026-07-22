"""Prepare datasets: verify structure, parse protocols, generate manifests.

Usage:
    python scripts/prepare_datasets.py --config configs/base_config.yaml
    python scripts/prepare_datasets.py --verify
"""
import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config


def verify_asvspoof2019(root_dir: str) -> bool:
    """Verify ASVspoof 2019 LA directory structure."""
    root = Path(root_dir)
    required = [
        root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt",
        root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt",
        root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt",
        root / "ASVspoof2019_LA_train" / "flac",
        root / "ASVspoof2019_LA_dev" / "flac",
        root / "ASVspoof2019_LA_eval" / "flac",
    ]

    all_ok = True
    for path in required:
        exists = path.exists()
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status} {path}")
        if not exists:
            all_ok = False

    return all_ok


def verify_asvspoof2021_la(root_dir: str) -> bool:
    """Verify ASVspoof 2021 LA directory structure."""
    root = Path(root_dir)
    required_patterns = [
        root / "ASVspoof2021_LA_eval" / "flac",
        root / "keys",
    ]

    all_ok = True
    for path in required_patterns:
        exists = path.exists()
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status} {path}")
        if not exists:
            all_ok = False

    return all_ok


def verify_wavefake(wavefake_dir: str, ljspeech_dir: str) -> bool:
    """Verify WaveFake directory structure."""
    wf = Path(wavefake_dir)
    lj = Path(ljspeech_dir)

    all_ok = True
    if wf.exists():
        subdirs = [d for d in wf.iterdir() if d.is_dir()]
        print(f"  [OK] WaveFake dir: {wf} ({len(subdirs)} synthesis methods)")
        for sd in subdirs[:5]:
            wav_count = len(list(sd.glob("*.wav")))
            print(f"    - {sd.name}: {wav_count} files")
    else:
        print(f"  [MISSING] WaveFake dir: {wf}")
        all_ok = False

    if (lj / "wavs").exists():
        wav_count = len(list((lj / "wavs").glob("*.wav")))
        print(f"  [OK] LJSpeech wavs: {wav_count} files")
    else:
        print(f"  [MISSING] LJSpeech dir: {lj / 'wavs'}")
        all_ok = False

    return all_ok


def verify_musan(musan_dir: str) -> bool:
    """Verify MUSAN noise directory."""
    musan = Path(musan_dir)
    noise_dir = musan / "noise"

    if noise_dir.exists():
        wav_files = list(noise_dir.rglob("*.wav"))
        print(f"  [OK] MUSAN noise: {len(wav_files)} files")
        return True
    else:
        print(f"  [MISSING] MUSAN noise dir: {noise_dir}")
        return False


def generate_manifest(
    dataset_name: str,
    samples: List[Dict],
    output_path: str,
) -> None:
    """Generate a CSV manifest file for a dataset."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        if samples:
            writer = csv.DictWriter(f, fieldnames=samples[0].keys())
            writer.writeheader()
            writer.writerows(samples)

    print(f"  Manifest saved: {output_path} ({len(samples)} samples)")


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--verify", action="store_true",
                        help="Only verify directory structure")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config.get("paths", {})

    print("=" * 60)
    print("DATASET PREPARATION")
    print("=" * 60)

    # Verify all datasets
    print("\n[1/4] ASVspoof 2019 LA:")
    ok_2019 = verify_asvspoof2019(paths.get("asvspoof2019_root", "./data/ASVspoof2019_LA"))

    print("\n[2/4] ASVspoof 2021 LA:")
    ok_2021_la = verify_asvspoof2021_la(paths.get("asvspoof2021_la_root", "./data/ASVspoof2021_LA"))

    print("\n[3/4] WaveFake + LJSpeech:")
    ok_wf = verify_wavefake(
        paths.get("wavefake_root", "./data/WaveFake"),
        paths.get("ljspeech_root", "./data/LJSpeech-1.1"),
    )

    print("\n[4/4] MUSAN (for perturbations):")
    ok_musan = verify_musan(paths.get("musan_root", "./data/musan"))

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  ASVspoof 2019 LA: {'[READY]' if ok_2019 else '[MISSING]'}")
    print(f"  ASVspoof 2021 LA: {'[READY]' if ok_2021_la else '[MISSING]'}")
    print(f"  WaveFake:         {'[READY]' if ok_wf else '[MISSING]'}")
    print(f"  MUSAN:            {'[READY]' if ok_musan else '[MISSING]'}")

    if not all([ok_2019, ok_2021_la, ok_wf, ok_musan]):
        print("\n[WARNING] Some datasets are missing. See datasets/download_instructions.md")

    if args.verify:
        return

    # Generate manifests if datasets exist
    if ok_2019:
        print("\nGenerating ASVspoof 2019 manifests...")
        from datasets.asvspoof2019 import ASVspoof2019LA
        for split in ["train", "dev", "eval"]:
            try:
                ds = ASVspoof2019LA(paths["asvspoof2019_root"], split=split)
                generate_manifest(
                    f"asvspoof2019_{split}",
                    ds.samples,
                    os.path.join(paths.get("data_root", "./data"), "manifests",
                                 f"asvspoof2019_{split}.csv"),
                )
                counts = ds.get_label_counts()
                print(f"    {split}: {counts}")
            except Exception as e:
                print(f"    Error for {split}: {e}")


if __name__ == "__main__":
    main()
