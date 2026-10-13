"""Build the hands-on teaching subset from the public M-A island dataset.

Instructor-only script. Run it once before the session; participants only
download the resulting zip (~30 MB instead of the 876 MB original).

Source
------
Ackermann, Iren, Wesselmecking, Krupp, Shetty (2022). "M-A island segmentation
of bainite microstructures in SEM image data". figshare, MIT licence.
https://doi.org/10.6084/m9.figshare.19232523.v2
Paper: Materials Characterization 191 (2022) 112091,
https://doi.org/10.1016/j.matchar.2022.112091

What it does
------------
1. Reads the 40 reference-size SEM images (2048 x 1536) of the three cooling
   strategies (sets i, ii, iii) and their COCO polygon labels.
2. Rasterises the polygons into binary masks (M-A island = 255).
3. Downsamples images and masks by ``--factor`` (default 2 -> 1024 x 768).
4. Writes PNGs plus ``metadata.csv`` (one row per image, fields parsed from the
   file name), ``DATASET_CARD.md``, ``LICENSE``, ``checksums.sha256`` and
   ``provenance.json`` -- i.e. the dataset documents itself.
5. Zips everything to ``<out>.zip``.

Usage
-----
    python make_teaching_data.py --source ~/data/public/ackermann2022_ma/dataset.zip
    python make_teaching_data.py --source ~/data/public/ackermann2022_ma/dataset --out data/ma_bainite_teaching
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import platform
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SCRIPT_VERSION = "1.0.0"
SOURCE_DOI = "10.6084/m9.figshare.19232523.v2"
PAPER_DOI = "10.1016/j.matchar.2022.112091"
SOURCE_URL = "https://ndownloader.figshare.com/files/34171098"
AUTHORS = "Marc Ackermann, Deniz Iren, Sebastian Wesselmecking, Ulrich Krupp, Deekshith Shetty"
SETS = ("i", "ii", "iii")

# e.g. 10-400-01-H-12mm-4000x14  /  10_500-V-6mm-4000x08
NAME_RE = re.compile(
    r"^(?P<series>.+?)-(?P<section>[HV])-(?P<pos>\d+)mm-(?P<mag>\d+)x(?P<field>\d+)$"
)

MIT_TEXT = """MIT License

Copyright (c) 2022 {authors}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

(Upstream dataset licence as declared on figshare: MIT,
https://doi.org/{doi}. This teaching subset is a downsampled derivative.)
"""


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def find_reference_root(source: Path, workdir: Path) -> tuple[Path, str | None]:
    """Return the ``1536x`` directory (extracting the zip if needed) and the zip's SHA-256."""
    zip_sha = None
    if source.suffix == ".zip":
        zip_sha = sha256(source)
        with zipfile.ZipFile(source) as zf:
            members = [m for m in zf.namelist() if "/1536x/" in f"/{m}"]
            zf.extractall(workdir, members=members)
        source = workdir
    hits = sorted(p for p in source.rglob("1536x") if p.is_dir())
    if not hits:
        raise FileNotFoundError(f"no '1536x' folder below {source}")
    return hits[0], zip_sha


def rasterise(size: tuple[int, int], polygons: list[list[float]]) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        if len(poly) >= 6:
            draw.polygon(poly, fill=255)
    return mask


def downsample(img: Image.Image, mask: Image.Image, factor: int) -> tuple[Image.Image, Image.Image]:
    if factor == 1:
        return img, mask
    size = (img.width // factor, img.height // factor)
    img_s = img.resize(size, Image.BOX)
    # area-average the mask, then re-binarise: a pixel is M-A if >= half its area was M-A
    m = np.asarray(mask.resize(size, Image.BOX), dtype=np.uint8)
    return img_s, Image.fromarray(np.where(m >= 128, 255, 0).astype(np.uint8))


def build(source: Path, out: Path, factor: int) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        ref, zip_sha = find_reference_root(source, Path(tmp))
        rows = []
        for set_name in SETS:
            for split in ("train", "val"):
                coco = json.loads((ref / set_name / "label" / f"{split}.json").read_text())
                polys: dict[int, list[list[float]]] = {}
                for ann in coco["annotations"]:
                    polys.setdefault(ann["image_id"], []).extend(ann["segmentation"])
                for info in coco["images"]:
                    src = ref / set_name / f"im_{split}" / info["file_name"]
                    img = Image.open(src).convert("L")
                    mask = rasterise(img.size, polys.get(info["id"], []))
                    img_s, mask_s = downsample(img, mask, factor)

                    stem = Path(info["file_name"]).stem
                    image_id = f"{set_name}__{stem}"
                    img_s.save(out / "images" / f"{image_id}.png", optimize=True)
                    mask_s.save(out / "masks" / f"{image_id}.png", optimize=True)

                    m = NAME_RE.match(stem)
                    if m is None:
                        raise ValueError(f"unexpected file name pattern: {stem}")
                    rows.append(
                        {
                            "image_id": image_id,
                            "image_file": f"images/{image_id}.png",
                            "mask_file": f"masks/{image_id}.png",
                            "cooling_strategy": set_name,
                            "sample_series": m["series"],
                            "section": m["section"],
                            "position_mm": int(m["pos"]),
                            "magnification": int(m["mag"]),
                            "field": int(m["field"]),
                            "original_split": split,
                            "source_file": f"1536x/{set_name}/im_{split}/{info['file_name']}",
                            "width_px": img_s.width,
                            "height_px": img_s.height,
                            "downsample_factor": factor,
                            "n_annotated_islands": len(polys.get(info["id"], [])),
                            "ma_area_fraction": round(float(np.asarray(mask_s).mean() / 255), 4),
                            "pixel_size_nm": "",  # not documented upstream -- see DATASET_CARD.md
                        }
                    )

    rows.sort(key=lambda r: r["image_id"])
    with (out / "metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (out / "LICENSE").write_text(MIT_TEXT.format(authors=AUTHORS, doi=SOURCE_DOI))
    (out / "DATASET_CARD.md").write_text(dataset_card(rows, factor))

    checksums = [
        f"{sha256(p)}  {p.relative_to(out).as_posix()}"
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name not in {"checksums.sha256", "provenance.json"}
    ]
    (out / "checksums.sha256").write_text("\n".join(checksums) + "\n")

    provenance = {
        "dataset": out.name,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "created_by_script": f"make_teaching_data.py v{SCRIPT_VERSION}",
        "parameters": {"downsample_factor": factor, "sets": list(SETS)},
        "source": {
            "doi": SOURCE_DOI,
            "download_url": SOURCE_URL,
            "zip_sha256": zip_sha,
            "licence": "MIT",
            "paper_doi": PAPER_DOI,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": Image.__version__,
        },
        "n_images": len(rows),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    archive = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    return Path(archive)


def dataset_card(rows: list[dict], factor: int) -> str:
    by_set: dict[str, int] = {}
    by_series: dict[str, int] = {}
    for r in rows:
        by_set[r["cooling_strategy"]] = by_set.get(r["cooling_strategy"], 0) + 1
        by_series[r["sample_series"]] = by_series.get(r["sample_series"], 0) + 1
    w, h = rows[0]["width_px"], rows[0]["height_px"]
    set_lines = "\n".join(f"| {k} | {v} |" for k, v in sorted(by_set.items()))
    series_lines = "\n".join(f"| {k} | {v} |" for k, v in sorted(by_series.items()))
    return f"""# Dataset card: M-A islands in bainitic steel (teaching subset)

## Summary

{len(rows)} secondary-electron SEM micrographs of a bainitic steel at nominal 4000x,
with dense binary masks of martensite-austenite (M-A) islands. Downsampled
{factor}x from 2048 x 1536 to {w} x {h} px for a CPU-sized hands-on session.

## Source and licence

- Upstream: Ackermann, Iren, Wesselmecking, Krupp, Shetty (2022), figshare,
  https://doi.org/{SOURCE_DOI}, licence **MIT** (see `LICENSE`).
- Paper: Materials Characterization 191 (2022) 112091, https://doi.org/{PAPER_DOI}
- This subset: built by `make_teaching_data.py` v{SCRIPT_VERSION}
  (see `provenance.json`; file integrity in `checksums.sha256`).
- **Cite the upstream dataset and paper** if you use these data.

## Contents

```
images/<image_id>.png   8-bit grayscale micrograph
masks/<image_id>.png    0 = matrix, 255 = M-A island
metadata.csv            one row per image (see below)
```

Images per cooling strategy (upstream sets; set i merges two strategies):

| cooling_strategy | images |
|---|---|
{set_lines}

Images per sample series (prefix of the file name):

| sample_series | images |
|---|---|
{series_lines}

## metadata.csv columns

| column | meaning | source |
|---|---|---|
| image_id | `<set>__<original file stem>` | derived |
| cooling_strategy | upstream set i / ii / iii | upstream description |
| sample_series | file-name prefix, e.g. `10-500-03` | file name |
| section | `H` or `V` | file name; **meaning not documented upstream** |
| position_mm | 6 / 9 / 12 | file name; **meaning not documented upstream** |
| magnification | nominal magnification (4000) | file name |
| field | field-of-view index | file name |
| original_split | upstream train / val | upstream folder |
| ma_area_fraction | fraction of mask pixels labelled M-A | computed |
| pixel_size_nm | **empty: not stated upstream** | none |

## Known gaps (deliberately kept, they are part of the lesson)

- **Pixel size is not documented.** Island sizes in nm cannot be computed
  reliably; only area fractions are scale-free.
- The meaning of `H`/`V` and of the `mm` value in file names is not documented.
- No information on annotator(s), annotation protocol or inter-annotator
  agreement is included in the dataset itself (see the paper).
- Detector, voltage, working distance and etching are not in per-image
  metadata.

## Intended use

Teaching: semantic segmentation of a two-phase microstructure, data-split
design (leakage across samples), and documenting an analysis FAIRly.
Not a benchmark: the subset is downsampled and small.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True, help="dataset.zip or extracted folder")
    ap.add_argument("--out", type=Path, default=Path("data/ma_bainite_teaching"))
    ap.add_argument("--factor", type=int, default=2, help="downsampling factor (default 2)")
    args = ap.parse_args()
    archive = build(args.source.expanduser(), args.out, args.factor)
    size_mb = archive.stat().st_size / 1e6
    print(f"wrote {args.out}/ and {archive} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
