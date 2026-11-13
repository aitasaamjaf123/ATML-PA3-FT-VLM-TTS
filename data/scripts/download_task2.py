"""Download the raw data needed by Task 2.

  python data/scripts/download_task2.py annotations
      -> COCO 2014 captions/instances (train+val) and VQAv2 questions/annotations/complementary pairs.
         Only the needed JSON members are extracted; the archives are deleted afterwards.
         (COCO ships all annotation files in one 241 MB zip, so it is downloaded whole.)

  python data/scripts/download_task2.py images --splits-dir data/splits/task2
      -> ONLY the COCO images listed in the manifest (a few thousand, not all 123K).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

COCO_ANN_ZIP = "https://images.cocodataset.org/annotations/annotations_trainval2014.zip"
COCO_MEMBERS = ["captions_train2014.json", "captions_val2014.json", "instances_train2014.json", "instances_val2014.json"]
VQA_BASE = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/"
VQA_ZIPS = [
    "v2_Questions_Train_mscoco.zip", "v2_Questions_Val_mscoco.zip",
    "v2_Annotations_Train_mscoco.zip", "v2_Annotations_Val_mscoco.zip",
    "v2_Complementary_Pairs_Train_mscoco.zip", "v2_Complementary_Pairs_Val_mscoco.zip",
]
VQA_EXPECTED = [
    "v2_OpenEnded_mscoco_train2014_questions.json", "v2_OpenEnded_mscoco_val2014_questions.json",
    "v2_mscoco_train2014_annotations.json", "v2_mscoco_val2014_annotations.json",
    "v2_mscoco_train2014_complementary_pairs.json", "v2_mscoco_val2014_complementary_pairs.json",
]


def http_download(url: str, dest: Path, retries: int = 4, timeout: int = 60) -> None:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = None
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
            tmp.replace(dest)
            return
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to download {url}: {last}")


def extract_members(zip_path: Path, wanted_names: list[str] | None, out_dir: Path) -> list[Path]:
    """Extract members whose file name is in `wanted_names` (None: every .json member)."""
    out = []
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            base = Path(name).name
            keep = base.endswith(".json") if wanted_names is None else base in wanted_names
            if keep:
                target = out_dir / base
                with z.open(name) as src, open(target, "wb") as dst:
                    while True:
                        buf = src.read(1 << 20)
                        if not buf:
                            break
                        dst.write(buf)
                out.append(target)
    return out


def download_annotations(raw_dir: Path) -> None:
    ann_dir, vqa_dir, tmp = raw_dir / "coco" / "annotations", raw_dir / "vqa", raw_dir / "_tmp"
    if all((ann_dir / m).exists() for m in COCO_MEMBERS):
        print("COCO annotations already present")
    else:
        z = tmp / "annotations_trainval2014.zip"
        print("Downloading COCO annotations zip ...")
        http_download(COCO_ANN_ZIP, z)
        got = extract_members(z, COCO_MEMBERS, ann_dir)
        assert len(got) == len(COCO_MEMBERS), f"missing COCO members, got {[g.name for g in got]}"
        z.unlink()
    for zname in VQA_ZIPS:
        expected = [e for e in VQA_EXPECTED if _vqa_zip_matches(zname, e)]
        if all((vqa_dir / e).exists() for e in expected):
            print(f"{zname}: already extracted")
            continue
        z = tmp / zname
        print(f"Downloading {zname} ...")
        http_download(VQA_BASE + zname, z)
        got = extract_members(z, expected, vqa_dir)
        if len(got) != len(expected):  # archive uses different member names: keep every JSON inside it
            got = extract_members(z, None, vqa_dir)
        assert got, f"{zname}: no JSON members extracted"
        z.unlink()
    for p in (tmp,):
        if p.exists():
            try:
                p.rmdir()
            except OSError:
                pass
    print("Annotations ready under", raw_dir)


def _vqa_zip_matches(zname: str, expected: str) -> bool:
    split = "train" if "Train" in zname else "val"
    if f"{split}2014" not in expected:
        return False
    if "Questions" in zname:
        return "questions" in expected
    if "Annotations" in zname:
        return "annotations" in expected
    return "complementary" in expected


def download_images(raw_dir: Path, splits_dir: Path, workers: int = 16) -> None:
    from task2_vlm_grounding.src.data_coco import coco_image_path, coco_image_url

    images = json.loads((splits_dir / "images.json").read_text())
    todo = []
    for split, ids in images.items():
        for i in ids:
            p = coco_image_path(raw_dir, split, i)
            if not p.exists() or p.stat().st_size == 0:
                todo.append((split, i, p))
    total = sum(len(v) for v in images.values())
    print(f"{total} images in manifest, {len(todo)} to download")

    def fetch(job):
        split, i, p = job
        http_download(coco_image_url(split, i), p, retries=5, timeout=30)
        return p

    done, failed = 0, []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch, j): j for j in todo}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                failed.append((futs[fut][1], str(e)))
            done += 1
            if done % 500 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} downloaded ({len(failed)} failed)")
    if failed:
        raise RuntimeError(f"{len(failed)} image downloads failed, e.g. {failed[:3]}")
    print("Images ready under", raw_dir / "coco")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["annotations", "images"])
    ap.add_argument("--raw-dir", default=str(REPO_ROOT / "data" / "raw"))
    ap.add_argument("--splits-dir", default=str(REPO_ROOT / "data" / "splits" / "task2"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    raw, splits = Path(args.raw_dir), Path(args.splits_dir)
    if args.what == "annotations":
        download_annotations(raw)
    else:
        download_images(raw, splits, args.workers)


if __name__ == "__main__":
    main()
