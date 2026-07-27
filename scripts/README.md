# IU X-Ray Analysis

Consistency checks between the IU Chest X-ray dataset variants:

| Directory | Content |
| :--- | :--- |
| `dataset/iu-xray/iu_xray_original/` | Original release: `reports/*.xml`, flat `images/*.png` |
| `dataset/iu-xray/iu_xray_r2gen/` | R2Gen split: `annotation.json`, `images/<ID>/{0,1}.png` |
| `dataset/iu-xray/iu_xray_r2gen_labeled/` | Manually labeled copy: `images/<ID>/{frontal,lateral}.png` |
| `dataset/iu-xray/iu_xray_r2gen_final/` | Cleaned dataset the 8 fine-tuning versions are built from |
| `dataset/iu-xray/iu_xray_translated.csv` | Reports with Italian translation (`findings_it`, `impression_it`) |

Requirements: Python 3, `numpy`, `pillow` (`pip install numpy pillow`).
The analysis scripts are read-only on the datasets — they only write to their own
output directories. The preprocessing and labeling scripts never modify their
input either: they always write a new dataset to `--out`.

## Analysis scripts

Run from the project root. Every script prints a summary table and writes a
Markdown copy of it to `scripts/analysis/out/<name>_summary.md`.

To run them all with their defaults (`_table.py` is a helper module, not a
script, so it is skipped):

```bash
for s in scripts/analysis/*.py; do
    [ "$(basename "$s")" = "_table.py" ] && continue
    echo "══════ $(basename "$s")"
    python3 "$s" || echo "  FAILED"
done
```

Scripts 4, 6 and 7 need `iu_xray_r2gen_labeled`; pass `--dataset DIR` (`--b DIR`
for script 4) if it does not sit under `dataset/iu-xray/`.

### 1. Findings: original XML reports vs R2Gen annotation

Matches by ID (`CXR<ID>_IM-xxxx` → `reports/<ID>.xml`) and compares the
`FINDINGS` section against the `report` field.

```bash
python3 scripts/analysis/compare_findings_original_to_r2gen.py \
    --csv scripts/analysis/out/findings_original_to_r2gen.csv
```

### 2. Findings: R2Gen annotation vs translated CSV

Matches the annotation ID against the `uid` column of the CSV.

```bash
python3 scripts/analysis/compare_findings_r2gen_to_translated.py \
    --csv-out scripts/analysis/out/findings_r2gen_to_translated.csv
```

### 3. Images: original vs R2Gen (pixel by pixel)

Matches images by the ID in the R2Gen folder name and in the original file
names, then compares decoded pixel arrays (shape + mode + md5 of the raw bytes).

```bash
python3 scripts/analysis/compare_images_original_to_r2gen.py \
    --csv-out scripts/analysis/out/images_original_to_r2gen.csv --workers 8
```

### 4. Images: two annotations compared sample by sample

Compares **only** the images listed in `image_path`, position by position
(`image_path[0]` vs `image_path[0]`, `[1]` vs `[1]`). Used to prove that manual
labeling changed file names only, never pixels.

```bash
python3 scripts/analysis/compare_annotation_images_r2gen_to_r2gen-labeled.py \
    --csv-out scripts/analysis/out/annotation_images_r2gen_to_labeled.csv
```

### 5. Reports missing the IMPRESSION section

For every annotation ID, opens the matching original XML and checks
`<AbstractText Label="IMPRESSION">`. Reports samples where the tag is absent
(`MISSING_TAG`), empty (`EMPTY`) or a pure placeholder such as `None.` / `XXXX`
(`PLACEHOLDER`).

```bash
python3 scripts/analysis/check_missing_impression_r2gen_to_original.py \
    --csv-out scripts/analysis/out/missing_impression_r2gen_to_original.csv \
    --list-out scripts/analysis/out/missing_impression_ids.txt
```

Extra option: `--strict` treats placeholders as present, flagging only genuinely
missing or empty sections.

### 6. Annotation views: exactly 1 frontal + 1 lateral

Checks that each sample's `image_path` lists exactly one image with `frontal` in
the name and one with `lateral`. Violations are typed: `SAME_VIEW`,
`MISSING_FRONTAL`, `MISSING_LATERAL`, `DUPLICATE_VIEW`, `WRONG_COUNT`,
`UNCLASSIFIED`.

```bash
python3 scripts/analysis/check_annotation_diff_views.py \
    --dataset dataset/iu-xray/iu_xray_r2gen_labeled \
    --csv-out scripts/analysis/out/annotation_views.csv \
    --list-out scripts/analysis/out/annotation_views_bad_ids.txt
```

### 7. Image folders holding two views of the same kind

Looks at what is actually on disk in `images/<ID>/` and reports the samples whose
folder holds **exactly two** images that are both frontal or both lateral —
i.e. no other view is available for that sample.

```bash
python3 scripts/analysis/check_same_view_pairs.py \
    --dataset dataset/iu-xray/iu_xray_r2gen_labeled \
    --csv-out scripts/analysis/out/same_view_pairs.csv \
    --list-out scripts/analysis/out/same_view_pairs_ids.txt
```

Extra option: `--scan-dirs` ignores `annotation.json` and walks every folder
under `images/` instead.

### Common options

| Option | Meaning |
| :--- | :--- |
| `--dataset DIR` | Dataset root — `dataset/iu-xray` for scripts 1–3 and 5, a single dataset directory for 6–7; script 4 uses `--a DIR` / `--b DIR` instead |
| `--csv` / `--csv-out FILE` | Write the per-item detail as CSV (script 1 uses `--csv`, the others `--csv-out`) |
| `--list-out FILE` | Write just the flagged IDs, one per line (scripts 5–7) |
| `--table-out FILE` | Where to write the Markdown summary; pass `''` to skip it |
| `--limit N` | Only process the first N records — useful for a quick check |
| `--show N` | How many examples to print; `0` prints them all (scripts 5–7) |
| `--workers N` | Parallel processes, image scripts only (default 8) |

## Preprocessing pipeline

Turns `iu_xray_r2gen_labeled` into `iu_xray_r2gen_final`, the dataset
`shield.data.build` reads. Each step takes `--dataset` and writes a **new**
dataset to `--out`, so any step can be redone without losing the previous one.
Each reuses the detection logic of the matching analysis script — imported, not
reimplemented.

```bash
D=dataset/iu-xray

# 1. drop the studies whose original report has no IMPRESSION      2955 → 2949
python3 scripts/preprocessing/drop_missing_impression.py \
    --dataset $D/iu_xray_r2gen_labeled --out $D/step1_impression

# 2. fix by hand the same-view pairs that have extra images on disk (GUI, 48)
python3 scripts/labeling/fix_annotation_views.py \
    --dataset $D/step1_impression --out $D/step2_views

# 3. drop the studies whose folder holds only two same-view images    → 2939
python3 scripts/preprocessing/drop_same_view_pairs.py \
    --dataset $D/step2_views --out $D/step3_pairs

# 4. normalise the order: frontal first, lateral second
python3 scripts/preprocessing/order_annotation_views.py \
    --dataset $D/step3_pairs --out $D/iu_xray_r2gen_final
```

Common options: `--images copy|symlink|none` (`symlink` avoids duplicating ~1 GB
between steps), `--dry-run`, `--overwrite`, `--show N`. Steps 1 and 3 refuse to
remove a study they could not verify (missing XML, missing image folder) unless
given `--drop-unverifiable` / `--drop-missing-dir`.

Then verify and build:

```bash
python3 scripts/analysis/check_annotation_diff_views.py --dataset $D/iu_xray_r2gen_final
python3 scripts/analysis/check_same_view_pairs.py       --dataset $D/iu_xray_r2gen_final
uv run python -m shield.data.build --all
```

`build` runs its own pre-flight check and refuses to write anything if the
annotation is not usable by all 8 versions.

## Labeling scripts

### `check_labeling.py` — flat frontal/lateral copies for visual review

Copies every image of a dataset into two folders, renamed `<folderID>__<file>`,
so all frontal (or lateral) images can be browsed at once and mislabeled ones
spotted quickly. Files containing `frontal` in the name go to `frontal/`, files
containing `lateral` go to `lateral/`.

```bash
# preview only, nothing is written
python3 scripts/labeling/check_labeling.py --dry-run

# write the copies outside the dataset (~1.1 GB for the full dataset)
python3 scripts/labeling/check_labeling.py \
    --dataset dataset/iu-xray/iu_xray_r2gen_labeled \
    --out /path/outside/labeling_check
```

| Option | Meaning |
| :--- | :--- |
| `--dataset DIR` | Input dataset, must contain `images/<ID>/...` (default: `iu_xray_r2gen_labeled`) |
| `--out DIR` | Output directory (default `<dataset>/labeling_check`) |
| `--dry-run` | Report only, copy nothing |
| `--overwrite` | Overwrite existing copies (otherwise they are skipped and reported) |
| `--show N` | How many unclassified/skipped files to list (default 10) |

### `extract_same_view_samples.py` — pull out fixable same-view samples

Selects the samples whose `annotation.json` entry points at two images of the
same view, drops those whose folder holds only those two files (nothing to
recover), and copies the **whole folder** of the remaining ones — the ones with
extra images on disk — so the pairing can be reviewed and corrected.

```bash
# preview the selection without copying
python3 scripts/labeling/extract_same_view_samples.py --dry-run

python3 scripts/labeling/extract_same_view_samples.py \
    --dataset dataset/iu-xray/iu_xray_r2gen_labeled \
    --out scripts/labeling/out/same_view_samples
```

| Option | Meaning |
| :--- | :--- |
| `--dataset DIR` | Input dataset (default: `iu_xray_r2gen_labeled`) |
| `--out DIR` | Output directory (default `scripts/labeling/out/same_view_samples`) |
| `--dry-run` | Report only, copy nothing |
| `--overwrite` | Replace folders already present in the output |
| `--show N` | How many samples to list; `0` lists them all |

### `prepare_copy.py` / `labeler.py` — building the labeled dataset

`prepare_copy.py` creates the working copy the labeler operates on;
`labeler.py` shows one image at a time (`F` = frontal, `L` = lateral,
`Backspace` = undo, `S` = skip study, `Esc`/`Q` = quit) and, once a study is
complete, renames its files and updates `annotation.json`.

```bash
python3 scripts/labeling/prepare_copy.py     # --force recreates from scratch
python3 scripts/labeling/labeler.py          # --include-skipped, --verify
```

> **Note:** both scripts resolve the dataset relative to their own location
> (`ROOT = dirname(__file__)`), so they expect to sit next to `iu_xray_r2gen/`.
> After the move into `scripts/labeling/` they must be run from
> `dataset/iu-xray/`, or their `ROOT` must be updated to point there.

## Results (full dataset)

| Check | Result |
| :--- | :--- |
| Findings, original vs R2Gen | 2955/2955 exact |
| Findings, R2Gen vs translated CSV | 2955/2955 exact, 100% with `findings_it` |
| Images, original vs R2Gen | 6091/6091 pixel-identical |
| Images, R2Gen vs labeled | 2955/2955 samples pixel-identical, 0 report differences |
| IMPRESSION present in the original reports | 2949/2955; 6 samples have an empty section (all in `train`) |
| Annotation views (1 frontal + 1 lateral) | 2898/2955 conform; 57 list two images of the same view (35 frontal, 22 lateral) |
| Of those 57 | 9 have only those two images on disk; 48 have extra images and were extracted for review |
