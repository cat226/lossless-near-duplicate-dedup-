"""
Distractor-pool audit with DINOv2 ViT-S/14 (docs/phase3_dataset_plan.md sections 8.3-8.4).

The audit embedding is deliberately neither SSCD nor pHash. It is used ONLY to look for hidden
near-duplicates; it is never a candidate generator or comparator.

Rules declared BEFORE any embedding was computed (and recorded in the manifest):
  * pool        : the 10,000 accepted distractors + the 157 Copydays ORIGINALS (as contamination
                  probes; the strong attacks are derived from those originals and are not added)
  * embedding   : DINOv2 ViT-S/14 CLS token (384-d), L2-normalised; each image is resized whole
                  to 224x224 (bicubic, no crop) and normalised with the ImageNet mean/std
  * search      : exact, all pairs, cosine similarity
  * flag rule   : cosine >= 0.90 is flagged for review; counts are also reported at
                  0.80 / 0.85 / 0.90 / 0.95 / 0.98 (descriptive)
  * manual check: the 30 closest pairs overall and the 10 closest distractor-Copydays pairs are
                  rendered to contact sheets, whatever their similarity
  * no file is deleted or edited; flagged pairs go to a separate manifest and the exclusion
    decision is deferred to the Phase 3 design freeze

    python phase3/audit_distractors.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MANIFESTS = ROOT / "manifests"
ARTIFACTS = ROOT / "artifacts" / "audit"

MODEL = "dinov2_vits14"
HUB_REPO = "facebookresearch/dinov2"
WEIGHTS_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
INPUT = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FLAG_THRESHOLD = 0.90
REPORT_THRESHOLDS = (0.80, 0.85, 0.90, 0.95, 0.98)
PAIR_MANIFEST_MIN = 0.85
COLLECT_FLOOR = 0.60  # pairs below this are not kept; chosen so the 30 closest pairs always exist
N_SHEET_OVERALL, N_SHEET_COPYDAYS = 30, 10
BATCH = 64


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_pool() -> tuple[list[str], list[str], list[Path]]:
    names, kinds, paths = [], [], []
    for line in open(MANIFESTS / "distractors.jsonl"):
        r = json.loads(line)
        if r["accepted"]:
            name = Path(r["key"]).name
            names.append(name)
            kinds.append("distractor")
            paths.append(DATA / "distractors" / name)
    for line in open(MANIFESTS / "copydays.jsonl"):
        r = json.loads(line)
        if r["accepted"] and r["set"] == "original":
            stem = Path(r["member"]).stem
            names.append("copydays_" + stem)
            kinds.append("copydays_original")
            paths.append(DATA / "copydays" / "original" / f"{stem}.png")
    return names, kinds, paths


def preprocess(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB").resize((INPUT, INPUT), Image.BICUBIC), dtype=np.float32) / 255.0
    return ((arr - MEAN) / STD).transpose(2, 0, 1)


def embed(paths: list[Path]) -> tuple[np.ndarray, str, str]:
    import torch

    torch.set_num_threads(10)
    model = torch.hub.load(HUB_REPO, MODEL, trust_repo=True)
    model.eval()
    ckpt = Path(torch.hub.get_dir()) / "checkpoints" / "dinov2_vits14_pretrain.pth"
    weights_sha = sha256_file(ckpt)
    out = np.empty((len(paths), 384), dtype=np.float32)
    with ThreadPoolExecutor(8) as pool, torch.no_grad():
        for s in range(0, len(paths), BATCH):
            batch = np.stack(list(pool.map(preprocess, paths[s : s + BATCH])))
            out[s : s + BATCH] = model(torch.from_numpy(batch)).numpy()
            if (s // BATCH) % 20 == 0:
                print(f"  embedded {min(s + BATCH, len(paths))}/{len(paths)}", flush=True)
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out, weights_sha, torch.__version__


def pairs_above(E: np.ndarray, floor: float) -> list[tuple[int, int, float]]:
    n = E.shape[0]
    found: list[tuple[int, int, float]] = []
    nn1 = np.empty(n, dtype=np.float32)
    for s in range(0, n, 1024):
        sims = E[s : s + 1024] @ E.T
        for r in range(sims.shape[0]):
            sims[r, s + r] = -1.0  # exclude self
        nn1[s : s + 1024] = sims.max(axis=1)
        rows, cols = np.nonzero(sims >= floor)
        for r, c in zip(rows, cols):
            i, j = s + int(r), int(c)
            if i < j:
                found.append((i, j, float(sims[r, c])))
    found.sort(key=lambda t: -t[2])
    return found, nn1


def contact_sheet(pairs, names, paths, title: str, out: Path) -> None:
    h = 170
    rows = []
    for i, j, sim in pairs:
        ims = []
        for k in (i, j):
            with Image.open(paths[k]) as im:
                im = im.convert("RGB")
                ims.append(im.resize((max(1, round(im.width * h / im.height)), h)))
        w = ims[0].width + ims[1].width + 20
        row = Image.new("RGB", (max(w, 420), h + 22), "white")
        row.paste(ims[0], (0, 22))
        row.paste(ims[1], (ims[0].width + 20, 22))
        ImageDraw.Draw(row).text((2, 4), f"cos={sim:.4f}  {names[i]}  |  {names[j]}", fill="black")
        rows.append(row)
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height + 6 for r in rows) + 20), "white")
    ImageDraw.Draw(sheet).text((2, 2), title, fill="black")
    y = 20
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 6
    sheet.save(out)


def main():
    names, kinds, paths = load_pool()
    assert len(names) == 10157, len(names)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    print(f"pool: {len(names)} images ({kinds.count('distractor')} distractors, "
          f"{kinds.count('copydays_original')} Copydays originals)", flush=True)

    E, weights_sha, torch_version = embed(paths)
    np.save(ARTIFACTS / "embeddings.npy", E)
    (ARTIFACTS / "names.json").write_text(json.dumps(names))

    found, nn1 = pairs_above(E, COLLECT_FLOOR)
    counts = {str(t): sum(1 for _, _, s in found if s >= t) for t in REPORT_THRESHOLDS}
    kind_of = lambda a, b: "distractor-distractor" if kinds[a] == kinds[b] == "distractor" else (
        "copydays-copydays" if kinds[a] == kinds[b] else "distractor-copydays")
    counts_by_kind = {}
    for i, j, s in found:
        if s >= FLAG_THRESHOLD:
            counts_by_kind[kind_of(i, j)] = counts_by_kind.get(kind_of(i, j), 0) + 1

    rows = [{"a": names[i], "b": names[j], "cosine": round(s, 6), "kind": kind_of(i, j),
             "flagged": s >= FLAG_THRESHOLD}
            for i, j, s in found if s >= PAIR_MANIFEST_MIN]
    (MANIFESTS / "distractor_audit_pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    top = found[:N_SHEET_OVERALL]
    mixed = [t for t in found if kind_of(t[0], t[1]) == "distractor-copydays"][:N_SHEET_COPYDAYS]
    for k in range(0, len(top), 10):
        contact_sheet(top[k : k + 10], names, paths, f"closest pairs overall, ranks {k + 1}-{k + len(top[k:k + 10])}",
                      ARTIFACTS / f"sheet_overall_{k // 10 + 1}.png")
    if mixed:
        contact_sheet(mixed, names, paths, "closest distractor-Copydays pairs", ARTIFACTS / "sheet_copydays.png")

    q = np.quantile(nn1, [0.0, 0.01, 0.5, 0.99, 1.0]).tolist()
    audit = {
        "purpose": "hidden near-duplicate audit of the Phase 3 distractor pool (docs/phase3_dataset_plan.md 8.4)",
        "embedding_model": {
            "name": "DINOv2 ViT-S/14 (CLS token, 384-d)",
            "loaded_via": f"torch.hub.load('{HUB_REPO}', '{MODEL}')",
            "weights_url": WEIGHTS_URL,
            "weights_sha256": weights_sha,
            "weights_bytes_expected": 88283115,
            "torch_version": torch_version,
            "weights_licence": {
                "name": "Apache License 2.0",
                "evidence": [
                    "README 'License' section of github.com/facebookresearch/dinov2: "
                    "'DINOv2 code and model weights are released under the Apache License 2.0. "
                    "See LICENSE for additional details.'",
                    "LICENSE file of the same repository is the Apache License, Version 2.0, January 2004",
                ],
                "checked": "2026-10-12",
                "scope_note": "other models in that repository (XRay-DINO weights, Cell-DINO code) "
                              "carry different terms and are not used",
            },
            "not_sscd_not_phash": True,
        },
        "preprocessing": f"whole image resized to {INPUT}x{INPUT} bicubic, ImageNet mean/std, no crop",
        "pool": {"distractors": kinds.count("distractor"), "copydays_originals": kinds.count("copydays_original"),
                 "total": len(names)},
        "declared_rules": {"flag_threshold_cosine": FLAG_THRESHOLD, "report_thresholds": list(REPORT_THRESHOLDS),
                           "pair_manifest_min_cosine": PAIR_MANIFEST_MIN,
                           "manual_check": f"{N_SHEET_OVERALL} closest overall + {N_SHEET_COPYDAYS} closest distractor-Copydays pairs",
                           "no_files_deleted": True},
        "results": {"pairs_at_or_above": counts, "flagged_pairs_by_kind": counts_by_kind,
                    "nearest_neighbour_cosine_min_p1_median_p99_max": q,
                    "pairs_in_manifest": len(rows)},
        "outputs": {"pairs_manifest": "phase3/manifests/distractor_audit_pairs.jsonl",
                    "contact_sheets_and_embeddings": "phase3/artifacts/audit/ (gitignored)"},
    }
    (MANIFESTS / "distractor_audit.json").write_text(json.dumps(audit, indent=1))
    print(json.dumps(audit["results"], indent=1))
    print("top 5 pairs:", [(names[i], names[j], round(s, 4)) for i, j, s in found[:5]])


def main_final():
    """Re-audit the curated pool (phase3/manifests/distractor_pool.jsonl) with the SAME declared
    rules. Embeddings already computed in the initial audit are reused; only new images are embedded."""
    rows = [json.loads(l) for l in open(MANIFESTS / "distractor_pool.jsonl")]
    names, kinds, paths = [], [], []
    for r in rows:
        names.append(r["name"]); kinds.append("distractor"); paths.append(DATA / "distractors" / r["name"])
    for line in open(MANIFESTS / "copydays.jsonl"):
        r = json.loads(line)
        if r["accepted"] and r["set"] == "original":
            stem = Path(r["member"]).stem
            names.append("copydays_" + stem); kinds.append("copydays_original")
            paths.append(DATA / "copydays" / "original" / f"{stem}.png")
    assert len(names) == 10157, len(names)
    assert len({sha256_file(p) for p, k in zip(paths, kinds) if k == "distractor"}) == 10000, "duplicate bytes in pool"

    old_names = json.loads((ARTIFACTS / "names.json").read_text())
    old_E = np.load(ARTIFACTS / "embeddings.npy")
    where = {n: i for i, n in enumerate(old_names)}
    E = np.empty((len(names), 384), dtype=np.float32)
    missing = []
    for i, n in enumerate(names):
        if n in where:
            E[i] = old_E[where[n]]
        else:
            missing.append(i)
    print(f"final pool: {len(names)} images; reused {len(names) - len(missing)} embeddings, embedding {len(missing)} new", flush=True)
    if missing:
        new_E, _, _ = embed([paths[i] for i in missing])
        E[missing] = new_E
    np.save(ARTIFACTS / "embeddings_final.npy", E)
    (ARTIFACTS / "names_final.json").write_text(json.dumps(names))

    found, nn1 = pairs_above(E, COLLECT_FLOOR)
    counts = {str(t): sum(1 for _, _, s in found if s >= t) for t in REPORT_THRESHOLDS}
    kind_of = lambda a, b: "distractor-distractor" if kinds[a] == kinds[b] == "distractor" else (
        "copydays-copydays" if kinds[a] == kinds[b] else "distractor-copydays")
    flagged_by_kind = {}
    for i, j, s in found:
        if s >= FLAG_THRESHOLD:
            flagged_by_kind[kind_of(i, j)] = flagged_by_kind.get(kind_of(i, j), 0) + 1
    manifest_rows = [{"a": names[i], "b": names[j], "cosine": round(s, 6), "kind": kind_of(i, j), "flagged": s >= FLAG_THRESHOLD}
                     for i, j, s in found if s >= PAIR_MANIFEST_MIN]
    (MANIFESTS / "distractor_pool_audit_pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest_rows))

    top = found[:N_SHEET_OVERALL]
    for k in range(0, len(top), 10):
        contact_sheet(top[k : k + 10], names, paths, f"FINAL pool: closest pairs, ranks {k + 1}-{k + len(top[k:k + 10])}",
                      ARTIFACTS / f"final_sheet_overall_{k // 10 + 1}.png")
    mixed = [t for t in found if kind_of(t[0], t[1]) == "distractor-copydays"][:N_SHEET_COPYDAYS]
    if mixed:
        contact_sheet(mixed, names, paths, "FINAL pool: closest distractor-Copydays pairs", ARTIFACTS / "final_sheet_copydays.png")
    band = [t for t in found if 0.85 <= t[2] < 0.90]
    sample = []
    if band:
        pick = np.sort(np.random.default_rng(12345).choice(len(band), size=min(24, len(band)), replace=False))
        sample = [band[k] for k in pick]
        for k in range(0, len(sample), 12):
            contact_sheet(sample[k : k + 12], names, paths, f"FINAL pool: seeded sample of the 0.85-0.90 band ({len(band)} pairs), part {k // 12 + 1}",
                          ARTIFACTS / f"final_sheet_band_{k // 12 + 1}.png")

    old = json.loads((MANIFESTS / "distractor_audit.json").read_text())
    if "initial_pool_audit" not in old:
        old = {
            "purpose": old["purpose"], "embedding_model": old["embedding_model"],
            "preprocessing": old["preprocessing"], "declared_rules": old["declared_rules"],
            "initial_pool_audit": {k: old[k] for k in ("pool", "results", "outputs", "manual_checks_done", "post_hoc_findings")},
        }
    old["final_pool_audit"] = {
        "pool_manifest": "phase3/manifests/distractor_pool.jsonl (10,000 images after the content-only exclusion; see distractor_pool.summary.json)",
        "pool": {"distractors": kinds.count("distractor"), "copydays_originals": kinds.count("copydays_original"), "total": len(names)},
        "sha256_distinct_distractors": 10000,
        "embeddings": {"reused_from_initial_audit": len(names) - len(missing), "newly_embedded": len(missing)},
        "results": {"pairs_at_or_above": counts, "flagged_pairs_by_kind": flagged_by_kind,
                    "nearest_neighbour_cosine_min_p1_median_p99_max": np.quantile(nn1, [0.0, 0.01, 0.5, 0.99, 1.0]).tolist(),
                    "pairs_in_manifest": len(manifest_rows),
                    "band_0.85_0.90_pairs": len(band), "band_sample_size": len(sample)},
        "outputs": {"pairs_manifest": "phase3/manifests/distractor_pool_audit_pairs.jsonl",
                    "contact_sheets": "phase3/artifacts/audit/final_sheet_*.png (gitignored)"},
    }
    (MANIFESTS / "distractor_audit.json").write_text(json.dumps(old, indent=1))
    print(json.dumps(old["final_pool_audit"]["results"], indent=1))
    print("top 5:", [(names[i][:10], names[j][:10], round(s, 4)) for i, j, s in found[:5]])


if __name__ == "__main__":
    sys.exit(main_final() if "--final" in sys.argv else main())
