# Machine learning for materials data: hands-on

Hands-on session of the NFDI4ING Caden Summer School 2026, *Practical Research Data
Management in Materials Science*, IMD-2, Forschungszentrum Jülich, 14 October 2026.
Bashir Kazimi, Materials Data Science and Informatics (IAS-9), Forschungszentrum Jülich.

You train a small neural network (a U-Net) to find martensite-austenite (M-A) islands in
SEM images of a bainitic steel, measure the M-A area fraction, test how much the result
depends on how you split the data, and save the analysis so that someone else can find,
understand and reproduce it.

## Notebooks

| Notebook | Content | Time | Open |
|---|---|---|---|
| `01_ma_segmentation.ipynb` | main exercise: data and metadata, threshold baseline, U-Net training, data splits, FAIR run folder | ~25 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bashirkazimi/caden2026-ml-handson/blob/main/notebooks/01_ma_segmentation.ipynb) |
| `02_bonus_spectra.ipynb` | bonus: the same ideas on XRD patterns, phase identification and automatic quality control of measurements | ~15 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bashirkazimi/caden2026-ml-handson/blob/main/notebooks/02_bonus_spectra.ipynb) |

## Run on Google Colab (recommended, nothing to install)

1. Click **Open in Colab** above and sign in with a Google account.
2. Notebook 01 asks Colab for a GPU. If its first code cell prints `device: cpu`, choose
   *Runtime → Change runtime type → T4 GPU* and run the cell again. Without a GPU it
   still works; training then takes about a minute instead of a few seconds.
3. Run the cells from top to bottom with **Shift + Enter** (or *Runtime → Run all*).
   The data (22 MB) are downloaded automatically.
4. At the end, notebook 01 zips your run folder and downloads it to your computer.
   Colab deletes files when the session ends.

When Colab shows *"Warning: This notebook was not authored by Google"*, click
*Run anyway*.

## Run on your own computer

```bash
git clone https://github.com/bashirkazimi/caden2026-ml-handson.git
cd caden2026-ml-handson
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab notebooks/01_ma_segmentation.ipynb
```

Python 3.10 or newer. The data zip is part of the repository, so no download is needed.

## If something goes wrong

- *"download failed ... using a SYNTHETIC stand-in dataset"*: the data could not be
  downloaded (no internet connection). The notebook still runs on synthetic look-alike
  images; the workflow is the same, the numbers are not.
- *No module named torch* (local installation only): `pip install torch`.

## Data

`data/ma_bainite_teaching.zip` is a teaching subset (40 images, downsampled 2x) of

> Ackermann, Iren, Wesselmecking, Krupp, Shetty (2022). *M-A island segmentation of
> bainite microstructures in SEM image data.* figshare.
> [doi:10.6084/m9.figshare.19232523.v2](https://doi.org/10.6084/m9.figshare.19232523.v2),
> MIT licence.
>
> Ackermann et al. (2022). *Automated segmentation of martensite-austenite islands in
> bainitic steel.* Materials Characterization 191, 112091.
> [doi:10.1016/j.matchar.2022.112091](https://doi.org/10.1016/j.matchar.2022.112091)

Please cite both if you use the data. The zip contains a dataset card, the licence,
SHA-256 checksums and a provenance record; `scripts/make_teaching_data.py` rebuilds it
from the original figshare download.

## Licence

Code and notebooks: MIT (see `LICENSE`). Data: MIT, see the licence inside the data zip.

## Contact

Bashir Kazimi, IAS-9, Forschungszentrum Jülich, b.kazimi@fz-juelich.de
