# Preserved Not Protected: How a “Private” VAE Latent Becomes a High-fidelity Proxy for Sensitive Attributes

**Revisiting FedAR through a demographic lens.**

Poster project for [MLSS 2026](https://mlss2026.is.tuebingen.mpg.de/), Max Planck
Institute for Intelligent Systems, Tübingen, 31 August – 11 September 2026.

Saranya Bhattacharjee · MSc Artificial Intelligence · Pattern Recognition Lab (LME),
Friedrich-Alexander-Universität Erlangen-Nürnberg

---

<p align="center">
  <img src="figures/hero_pipeline.png" alt="Experimental pipeline" width="100%">
</p>

---

## The claim

[FedAR](https://ieeexplore.ieee.org/document/10797675) (Chatterjee, Ghosh,
**Bhattacharjee**, Das & Banerjee, *IEEE Transactions on Affective Computing*, 16(3):1461–1472, 2025)
proposed a federated framework for facial emotion recognition. Face images are
compressed into VAE latent vectors before any gradients are shared; resampling is
then applied to those vectors on the client side to correct class imbalance.

I designed the VAE encoder for that paper. It was trained with reconstruction and
KL loss alone — no demographic supervision, no fairness constraint, no
disentanglement term. The paper included a privacy check: reconstructing latents
through an external decoder gave similarity 0.06, so the raw images were safe.

**That check answers "can you invert the latent?" It does not answer "can you read
attributes off the latent?"** Those are different properties, and only the first
was measured.

This project asks three questions, on two corpora:

1. **Are demographic attributes recoverable from the latent?** (Experiment A)
2. **Does that matter for the downstream task?** (Experiment B)
3. **Can the information be removed?** (Experiments C and D)

Plus two controls that could have invalidated the framing, one of which did.

This is not a claim that latents contain demographic information — that is established 
(see Positioning below). It is an audit of what a deployed federated privacy method inherited 
without checking: an attribute proxy that survives compression, survives resampling, and 
survives adversarial removal.

---

## Why this isn't obvious (positioning)

A reasonable reaction is: of course a face latent encodes race — reconstruction preserves appearance, so a probe will read it off. That intuition is correct, and it is exactly the point. The leakage is not the contribution; the contribution is the gap between what is known and what a shipped privacy method actually verified. Three lines of prior work set that gap up:

Invertibility ≠ attribute privacy. Kaushik et al. (FG 2025) show that purpose-built irreversible face templates — encodings designed so the image cannot be reconstructed — still leak age, gender and ethnicity to attribute inference, and that leakage persists across compression dimensions. So "you can't invert it" is a known-insufficient privacy guarantee. FedAR's privacy check (similarity 0.06) measures precisely the property Kaushik et al. show is not enough.
The proxy is something others build on purpose. Grari, Lamprier & Detyniecki (2021) deliberately train a VAE latent to hold as much sensitive information as possible, to use it as a proxy for fairness when demographic labels are missing. One line of research constructs the exact object our privacy step produces by accident — and at task-level fidelity.
Federated training leaks attribute distributions. Even when a sensitive attribute is never used in training, it can be inferred from the shared model (federated attribute-inference literature). So the federated setting does not neutralise the leak; it carries it.

Put together: the fairness-without-demographics field assumes the sensitive attribute is missing and hunts for a weak proxy. Here the privacy step supplies a near-perfect proxy for free, and — our new result — standard interventions do not remove it. The obvious fact (leakage exists) is the setup. The non-obvious facts (a deployed privacy claim never measured it; and once present it is irremovable by the standard fixes) are the contribution.


---

## Headline result

Attributes are recoverable from the frozen latent at close to task-level
fidelity. On RAF-DB, AUROC:

| target | AUROC |
|---|---|
| emotion — *the task the encoder was built for* | 0.857 |
| gender | 0.835 |
| race | 0.826 |
| age | 0.813 |

**But the encoder is not responsible.** A PCA projection of downsampled pixels to
the same 128 dimensions retains as much or slightly more. So the finding is
**preservation, not learning**: compressing to a low-dimensional latent removes
essentially none of what was already linearly present in the pixels. The
representation step is not a sanitisation step.

And no intervention we tested reaches the resulting subgroup disparity — not
task-label resampling, not demographic balancing, not adversarial removal at any
adversarial strength.

---

## The honest framing

Two caveats belong in front of the results.

**RAF-DB's human demographic annotations were not available.** The official
release carries per-image race, age and gender labels, but access was requested
from BUPT and not granted in time. The public Kaggle mirrors redistribute only
the emotion labels. Demographics for RAF-DB are therefore **inferred with the
FairFace classifier** (Kärkkäinen & Joo, WACV 2021) — the protocol used by the
published fairness-in-FER studies this work compares against. They are model
predictions, not ground truth, and they carry FairFace's own error and bias.
Because label noise biases probe accuracy downward, the RAF-DB demographic
numbers are **lower bounds**.

**UTKFace has no emotion labels.** Its downstream task is age-bucket
classification, which stands in for emotion structurally — an imbalanced
multi-class problem over the same frozen latents — but is not the same task.

The two datasets answer complementary questions rather than the same question
twice:

| | UTKFace | RAF-DB |
|---|---|---|
| images | 23,705 | 15,339 |
| demographic labels | ground truth (filename-encoded) | FairFace-inferred |
| emotion labels | none | human-annotated, 7 classes |
| downstream task | age bucket (4 classes) | emotion (7 classes) |
| task imbalance | 2.1 native, 41.5 induced | **17.0 native** |
| answers | *what does the representation retain?* | *does it matter for the task FedAR was built for?* |

RAF-DB's native imbalance ratio of 17.0 is close to the AffectNet setting FedAR
was designed for (18.7), so the resampling intervention engages without any
artificial skewing.

---

## Results

### Experiment A — attribute recoverability

Three probe families (logistic regression, linear SVM, one-hidden-layer MLP)
predicting each attribute from the **frozen** latent. Best probe reported;
balanced accuracy, since both corpora have skewed class priors.

| attribute | UTKFace | chance | lift | RAF-DB | chance | lift |
|---|---|---|---|---|---|---|
| gender | 0.869 | 0.500 | **+0.369** | 0.753 | 0.500 | **+0.253** |
| race | 0.619 | 0.200 | **+0.419** | 0.439 | 0.200 | **+0.239** |
| age bucket | 0.602 | 0.250 | **+0.352** | 0.515 | 0.250 | **+0.265** |
| emotion (task) | — | — | — | 0.497 | 0.143 | **+0.354** |

Logistic regression alone lands within 1–3 points of the MLP on every target, so
the information is **linearly decodable** — any downstream linear classifier
inherits it for free.

**Controls.** Every target is reported against a stratified dummy, a
majority-class dummy, and a **label-shuffled null** — the same probe trained and
tested on permuted labels, which must land at chance. It does, on all seven
target/dataset pairs. Cross-validated standard deviations are 0.003–0.014.

### Control 1 — is the encoder responsible? (No)

A PCA projection of downsampled pixels to exactly the latent's dimensionality,
fitted on the train split only, run through the identical probe suite.

| | VAE latent | best PCA-128 | difference |
|---|---|---|---|
| UTKFace gender | +0.369 | +0.377 | −0.008 |
| UTKFace race | +0.419 | +0.422 | −0.003 |
| UTKFace age | +0.352 | +0.360 | −0.008 |
| RAF-DB gender | +0.253 | +0.282 | −0.029 |
| RAF-DB race | +0.239 | +0.263 | −0.024 |
| RAF-DB age | +0.265 | +0.265 | 0.000 |

Mean VAE advantage on demographic targets: **−0.006** (UTKFace), **−0.018**
(RAF-DB).

The VAE gives no advantage on the *task* either — RAF-DB emotion: VAE +0.354,
PCA-RGB +0.350, raw 16×16 grayscale **+0.390**. At β=1 on this data scale the
encoder behaves like a nonlinear PCA as far as any downstream probe can tell,
which is the empirical echo of Lucas et al. (NeurIPS 2019) showing that linear
VAEs recover the pPCA solution exactly.

**This control changed the framing.** The claim is preservation, not learning.

### Experiment B — does it matter downstream? (five seeds)

FedAvg simulated on the cached latents. Ten clients, fifty rounds, three
conditions, seeds 42/1/7/13/99. Race and gender are held out as sensitive
attributes and never seen by any model.

Absolute, mean ± std over seeds:

| | UTKFace (age, IMR 41.5) | | RAF-DB (emotion, IMR 17.0) | |
|---|---|---|---|---|
| condition | balanced acc | gap | balanced acc | gap |
| no intervention | 0.4681 ± 0.0091 | 0.3141 ± 0.0113 | 0.5188 ± 0.0086 | 0.1982 ± 0.0328 |
| FedAR resampling | 0.4767 ± 0.0078 | 0.2987 ± 0.0170 | 0.5099 ± 0.0018 | 0.2133 ± 0.0162 |
| demographic balancing | 0.4698 ± 0.0119 | 0.2988 ± 0.0225 | 0.5070 ± 0.0071 | 0.2073 ± 0.0194 |

Paired within-seed differences against no intervention:

| | UTKFace Δgap | UTKFace Δbalanced | RAF-DB Δgap | RAF-DB Δbalanced |
|---|---|---|---|---|
| FedAR | **−0.0154 ± 0.0121**, 5/5 negative | **+0.0086 ± 0.0025**, 5/5 positive | +0.0152 ± 0.0300, p=0.32 | −0.0089 ± 0.0096 |
| demographic balancing | −0.0154 ± 0.0189, 4/5 negative | +0.0017 ± 0.0057 | +0.0091 ± 0.0326, p=0.56 | −0.0118 ± 0.0124 |

**FedAR resampling does what it was designed to do.** On UTKFace it improved
balanced accuracy in every seed. But the largest effect on the demographic
subgroup gap is **4.9% of that gap's own magnitude** (UTKFace) and **7.7%**
(RAF-DB), with no detectable effect on RAF-DB at all.

Demographic balancing is the decisive control: balancing the *sensitive*
attribute directly, rather than the task labels, does not close the gap either.

**A methodological note.** RAF-DB gap variance from seed alone is ±0.0328 on a
gap of 0.1982 — 17% relative. A single-seed run of this experiment reported a
+0.0193 gap widening on RAF-DB that five seeds show to be noise. Single-seed
federated fairness results at this scale are unreliable.

**Worst-group behaviour.** UTKFace: White/Female in 15/15 runs. RAF-DB:
Other/Female 4/5 under no intervention → Black/Male 4/5 under FedAR →
Indian/Male 2/5, Black/Male 2/5, Other/Female 1/5 under demographic balancing. So
the interventions relocate who absorbs the disparity on RAF-DB, modally rather
than universally, and not at all on UTKFace.

### Experiments C and D — can it be removed? (five seeds)

Gradient reversal layer feeding demographic adversaries, fine-tuned for 20
epochs, sweeping λ ∈ {1, 5, 20, 50}. The encoder is then **frozen and a fresh
probe trained from scratch** — the check from Elazar & Goldberg (2018), which
showed that adversarially removed attributes are often recoverable by a newly
initialised probe: the adversary was defeated, the information was not.

Above-chance signal removed, measured on the deterministic linear (LogReg)
probe, **mean ± std over seeds 42/1/7/13/99**. Every cell that matters is
sign-consistent across all five seeds unless noted.

| λ | UTKFace gender / race / age | RAF-DB gender / race / age | RAF-DB emotion (task) |
|---|---|---|---|
| 1  | 0.1±1.0 / 3.3±0.9 / 2.4±0.8      | 3.1±0.8 / 5.1±2.0 / 3.2±1.5       | 0.4±1.4 |
| 5  | 7.1±0.4 / 11.6±1.3 / 8.5±1.6     | 8.4±1.1 / **16.1±2.4** / 10.1±1.1 | 0.3±0.9 |
| 20 | 16.8±2.6 / **19.4±1.7** / 12.9±1.8 | 14.3±4.1 / 12.1±6.0 / 10.9±5.0  | 3.0±1.8 |
| 50 | 10.1±2.2 / 11.2±2.1 / 8.1±2.3    | 8.4±2.6 / 4.2±1.6 / −1.4±1.5 *(1+/4−)* | 1.7±1.5 |

Removal peaks below ~20% and declines with λ, on both corpora. Every
demographic target at every λ is removed *positively* (signal reduced, not
increased) and sign-consistent across all five seeds — with one exception:
RAF-DB age at λ=50 sits at −1.4% ± 1.5 with signs split 1+/4−, spanning zero.

**What the five seeds corrected.** The earlier single-seed run reported, at
λ=50 on RAF-DB, race *and* age coming back more recoverable than before
(−1.7% and −6.6%). Across five seeds that dissolves: **race is +4.2% ± 1.6,
positive in every seed** — the negative sign was noise — and age is only a
weak −1.4% that spans zero rather than a −6.6% collapse. The dramatic
negative-removal artefact does not survive replication. (This is the same
lesson the federated sweep taught: a single-seed point estimate at this scale
cannot separate a real effect from seed noise.)

The RAF-DB runs track **emotion accuracy** as the utility axis. It moves by
0.3–3.0% across every λ. So this is not adversarial pressure destroying the
representation wholesale: the task signal survives and the demographic signal
survives. The intervention does not reach what it is aimed at.

Both the removal sweep and the federated experiment are now five-seed; the
worst-remaining single-seed dependency has been closed.

---

## Pipeline

```
                    UTKFace                      RAF-DB
                       │                            │
                       │                   annotate_demographics.py
                       │                   (FairFace → race/gender/age)
                       │                            │
                       ▼                            ▼
                utkface_dataset.py           rafdb_dataset.py
                       └──────────┬─────────────────┘
                                  ▼
                             train.py
                   ResNet-18 VAE, 128-dim latent
                   reconstruction + KL loss only
                                  │
                                  ▼
                        extract_latents.py
                     frozen encoder → cached μ
                                  │
     ┌──────────┬───────────────┬─┴────────────┬───────────────┐
     ▼          ▼               ▼              ▼               ▼
 probes.py  federated_    debias_        pixel_          figures/
 (Exp. A)   multiseed.py  multiseed.py   baseline.py     (visualize, cross_dataset,
            (Exp. B,      (Exp. C/D,     (control)        control_figures, tradeoff,
             5 seeds)      5 seeds)                        make_*_panels)

 shared building blocks used by all of the above:
     fedar_common/  data.py · probing.py · stats.py · plotting.py
```

Everything after `extract_latents.py` runs on cached 128-dimensional vectors, so
the GPU is needed only twice: once to train the VAE, once to extract.

`federated.py` and `debias.py` are the single-seed *engines*; the
`*_multiseed.py` drivers import them unchanged and run the same pipeline across
seeds `42,1,7,13,99`. A one-seed slice of a multiseed run is bit-identical to
calling the engine directly, so there is one implementation of each experiment,
not two. `fedar_common/` holds the pieces that were previously copy-pasted across
scripts (one `build_dataset`, one probe suite, one `paired_stats`, one palette).

---

## Files

All scripts live in `code/src/` and are run from that directory.

**Shared building blocks** (`code/src/fedar_common/`)

One definition of each cross-cutting piece, imported by the scripts below.
These were previously copy-pasted; centralising them means the PCA control
provably uses the *same* probe suite and the *same* split as the leakage probe.

| file | what it does |
|---|---|
| `fedar_common/data.py` | The single `build_dataset(args, transform, rafdb_split=...)`. `train.py` passes `rafdb_split='train'` to hold out the RAF-DB test split; downstream scripts pass `None` to read all rows. Re-exports `age_to_bucket`. |
| `fedar_common/probing.py` | The probe suite: `make_probes` (LogReg/LinearSVM/MLP), `make_baselines`, `evaluate` (balanced acc / macro-F1 / AUROC). Used by Experiment A and the PCA control. |
| `fedar_common/stats.py` | `paired_stats` — per-seed mean/std/sign-consistency + optional paired *t*-test, used by both multiseed drivers. |
| `fedar_common/plotting.py` | The Okabe-Ito palette (`CB`, `OI`) and `load_json`, previously redefined in eight figure files. |

**Data**

| file | what it does |
|---|---|
| `utkface_dataset.py` | Parses UTKFace filenames (`age_gender_race_timestamp.jpg`) into labels. Owns `age_to_bucket`. Run standalone for the composition report. |
| `rafdb_dataset.py` | Joins RAF-DB emotion labels to the FairFace demographics CSV. Tolerates the three folder layouts the public mirrors use. |
| `annotate_demographics.py` | Runs the FairFace ResNet-34 over RAF-DB and writes `data/rafdb_demographics.csv`. GPU-batched, ~13 s for 15k images. Keeps a local age-bucket variant that clamps out-of-range ages (FairFace needs a valid bucket for every prediction). |

**Model and representation**

| file | what it does |
|---|---|
| `model.py` | ResNet-18 style VAE. 96×96 RGB in, 128-dim latent, mirrored decoder. ~10.2M parameters. Run standalone for a shape sanity check. |
| `train.py` | Trains the VAE. `--dataset {utkface,rafdb}`. Saves checkpoints and reconstruction grids at epochs 1/5/10/25/50. |
| `extract_latents.py` | Frozen encoder → `μ` for every image, cached to `.npz` with all labels and the split. **The central artifact.** |

**Experiments**

| file | what it does |
|---|---|
| `probes.py` | Experiment A. Three probe families × all targets, with stratified/majority/shuffled-null controls and 5-fold CV. |
| `federated.py` | Experiment B engine: one FedAvg run (single seed), three conditions, subgroup breakdown by race × gender. Not run directly for the headline numbers — `federated_multiseed.py` drives it. |
| `federated_multiseed.py` | Experiment B, the version behind the reported numbers. Runs `federated.py` across seeds `42,1,7,13,99` with paired within-seed differences. Pass `--seeds 42` to reproduce a single-seed run. |
| `debias.py` | Experiments C/D engine: GRL adversarial removal with the Elazar-Goldberg fresh-probe recovery check (`fresh_probe` lives here, co-located with the removal loop). |
| `debias_multiseed.py` | Experiments C/D across seeds; drives `debias.py` unchanged. Now the five-seed result behind the reported removal numbers. Pass `--seeds 42` to reproduce a single-seed slice. |
| `pixel_baseline.py` | The control that changed the framing. Runs the identical probe suite (from `fedar_common`) on raw pixels and on dimension-matched PCA. |
| `inversion_check.py` | *Optional.* Reconstruction fidelity of our **own** decoder — a weaker threat model than FedAR's external-decoder check, so its number is not a poster claim. See its header before using it. |

**Figures** (run from `code/src/`; write to `../../figures/`)

| file | what it does |
|---|---|
| `visualize.py` | Per-dataset: probe performance, composition, LDA projection, UMAP. |
| `tradeoff_figure.py` | Per-dataset: removal against λ and against utility cost. |
| `cross_dataset.py` | Both datasets together. Leakage, task-vs-demographic AUROC, federated conditions, worst group, removal ceiling (incl. `figX5_removal_cross.png`). |
| `control_figures.py` | The two control figures: PCA comparison and per-seed paired differences. |
| `make_lda_panels.py`, `make_umap_panels.py`, `make_umap_panels_isolated.py` | Laptop-only projection panels (UMAP is cut from the wall; LDA replaced it). |

---

## Reproducing from scratch

### Environment

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install scikit-learn pandas numpy matplotlib seaborn tqdm imbalanced-learn scipy
pip install umap-learn        # optional; segfaults if TensorFlow is present
```

Built and run on a single NVIDIA RTX 2080 (8 GB) on the FAU CIP Pool.

### Data

**UTKFace** — aligned-and-cropped, filenames encode the labels.

```bash
kaggle datasets download -d jangedoo/utkface-new
unzip -q utkface-new.zip -d data/
# keep only the flat UTKFace/ folder; the zip ships duplicates
```

**RAF-DB** — emotion labels only from the public mirror; demographics inferred.

```bash
kaggle datasets download -d shuvoalok/raf-db-dataset
unzip -q raf-db-dataset.zip -d data/rafdb_raw

# FairFace weights: res34_fair_align_multi_7_20190809.pt
# linked from https://github.com/dchen236/FairFace
mkdir -p weights   # place the .pt file here
```

### Run

The `*_multiseed.py` drivers are the entrypoints for Experiments B and C/D —
they run the underlying engine across seeds `42,1,7,13,99`. The single-seed
`federated.py` / `debias.py` calls below are optional and reproduce one slice.

```bash
cd code/src

# ---------- UTKFace ----------
python utkface_dataset.py                       # composition check
python train.py --data_root ../../data/utkface --epochs 50
python extract_latents.py --data_root ../../data/utkface

python probes.py
python pixel_baseline.py --data_root ../../data/utkface \
    --output ../../results/pixel_baseline_utkface.json

# Experiment B (headline): five-seed federated sweep
python federated_multiseed.py --skew 0.05,1.0,0.5,0.15 \
    --output ../../results/federated_multiseed_utkface.json

# Experiments C/D (headline): five-seed removal sweep, all lambdas
python debias_multiseed.py \
    --data_root ../../data/utkface \
    --checkpoint ../../checkpoints/vae_best.pt \
    --reference_latents ../../latents/utkface_latents.npz \
    --output ../../results/debias_multiseed_utkface.json

# --- optional single-seed slices (the engines, called directly) ---
python federated.py --skew 0.05,1.0,0.5,0.15 \
    --output ../../results/federated_skewed.json
for lam in 1 5 20 50; do
  out=../../results/debias_lam${lam}.json
  [ $lam -eq 1 ] && out=../../results/debias_results.json
  python debias.py --data_root ../../data/utkface --lambda_max $lam --output $out
done

# ---------- RAF-DB ----------
python annotate_demographics.py \
    --image_root ../../data/rafdb_raw \
    --weights ../../weights/res34_fair_align_multi_7_20190809.pt
python rafdb_dataset.py                         # composition check

python train.py --dataset rafdb \
    --data_root ../../data/rafdb_raw \
    --demographics ../../data/rafdb_demographics.csv \
    --checkpoint_dir ../../checkpoints/rafdb \
    --figure_dir ../../figures/recon_rafdb \
    --log_file ../../logs/train_rafdb.csv --epochs 50

python extract_latents.py --dataset rafdb \
    --data_root ../../data/rafdb_raw \
    --demographics ../../data/rafdb_demographics.csv \
    --checkpoint ../../checkpoints/rafdb/vae_best.pt \
    --output ../../latents/rafdb_latents.npz

python probes.py --latents ../../latents/rafdb_latents.npz \
    --output ../../results/probe_results_rafdb.json
python pixel_baseline.py --dataset rafdb \
    --data_root ../../data/rafdb_raw \
    --demographics ../../data/rafdb_demographics.csv \
    --latents ../../latents/rafdb_latents.npz \
    --output ../../results/pixel_baseline_rafdb.json

# Experiment B (headline): five-seed federated sweep
python federated_multiseed.py --latents ../../latents/rafdb_latents.npz \
    --output ../../results/federated_multiseed_rafdb.json

# Experiments C/D (headline): five-seed removal sweep
python debias_multiseed.py --dataset rafdb \
    --data_root ../../data/rafdb_raw \
    --demographics ../../data/rafdb_demographics.csv \
    --checkpoint ../../checkpoints/rafdb/vae_best.pt \
    --reference_latents ../../latents/rafdb_latents.npz \
    --output ../../results/debias_multiseed_rafdb.json

# --- optional single-seed slices (the engines, called directly) ---
python federated.py --latents ../../latents/rafdb_latents.npz \
    --output ../../results/federated_rafdb.json
for lam in 1 5 20 50; do
  python debias.py --dataset rafdb \
    --data_root ../../data/rafdb_raw \
    --demographics ../../data/rafdb_demographics.csv \
    --checkpoint ../../checkpoints/rafdb/vae_best.pt \
    --reference_latents ../../latents/rafdb_latents.npz \
    --lambda_max $lam \
    --output ../../results/debias_rafdb_lam${lam}.json \
    --latent_out ../../latents/rafdb_debiased_lam${lam}.npz \
    --ckpt_out ../../checkpoints/rafdb/vae_debiased_lam${lam}.pt
done

# ---------- figures ----------
python visualize.py --no_umap
python visualize.py --results ../../results/probe_results_rafdb.json \
    --latents ../../latents/rafdb_latents.npz --suffix _rafdb --no_umap
python tradeoff_figure.py --dataset utkface
python tradeoff_figure.py --dataset rafdb --suffix _rafdb
python cross_dataset.py
python control_figures.py
```

End to end on one RTX 2080: roughly two hours, dominated by the eight debias runs
and the two multi-seed federated sweeps.

---

## Methodological notes

**Why balanced accuracy, not accuracy.** UTKFace race is 42.5% White; a
majority-class predictor scores 0.425 on raw accuracy while being useless.

**Why LogReg for the removal metric.** `MLPClassifier(early_stopping=True)`
draws an internal validation split, so its balanced accuracy is not
bit-reproducible across call sites. Logistic regression is deterministic and
matches the probe figures to four decimals, so all figures are locked to it.

**Why λ=1 anchors the utility axis.** Every debias run adds 20 epochs of
ordinary training on top of the pretrained VAE, which changes reconstruction on
its own. Anchoring to the weakest adversarial setting isolates the cost
attributable to adversarial pressure.

**Why paired within-seed differences.** The baseline gap itself varies with the
client partition, which is seed-dependent. Computing `gap(intervention) −
gap(none)` within each seed before averaging removes that shared variance.

**Why sign consistency over p-values.** With five seeds, significance is
fragile. "Negative in all five seeds" is the more robust statement.

**Why RAF-DB's native split is reused.** The dataset ships a train/test
partition; drawing a new one would break comparability with published RAF-DB
results. The VAE trains only on the train split, so test images are never seen.

**Why UMAP is not on the poster.** It showed structure for gender but nothing
for race or age, despite probes recovering race at AUROC 0.826. The LDA
projection replaced it: a linear projection with no free hyperparameters, fitted
on train and shown on held-out test. Probe performance is the evidence; the
projection is the illustration.

---

## Limitations

- **RAF-DB demographics are model-inferred.** FairFace mean race confidence is
  79.3% and the inferred composition matches RAF-DB's published skew, but these
  are predictions. All RAF-DB demographic results should be read as lower bounds.
- **PCA was fitted on 32×32 RGB and 16×16 grayscale, not the 96×96 the VAE saw.**
  A PCA on the full input would likely retain more, which would strengthen rather
  than weaken the conclusion, so the control is conservative in the direction
  that counts against it.
- **Neither dataset provides subject identifiers**, so subject-disjoint probe
  splits cannot be guaranteed. Both are composed largely of distinct individuals.
- **Race categories are coarse and socially constructed.** UTKFace uses five;
  FairFace's seven are collapsed onto those five (Middle Eastern and
  Latino_Hispanic → Other) so the two corpora share an axis. Raw seven-class
  labels are preserved in the CSV.
- **UTKFace's task is not emotion.** Age-bucket classification is a structural
  stand-in.
- **One architecture, one latent dimension, one β.** The findings concern a
  standard reconstruction+KL VAE at 128 dimensions, not VAEs in general.

---

## References

Chatterjee, Ghosh, Bhattacharjee, Das & Banerjee. FedAR: Federated Artificial
Resampling for Imbalanced Facial Emotion Recognition. *IEEE Transactions on
Affective Computing*, 16(3):1461–1472, 2025.

Li, Deng & Du. Reliable Crowdsourcing and Deep Locality-Preserving Learning for
Expression Recognition in the Wild. *CVPR*, 2017. (RAF-DB)

Zhang, Song & Qi. Age Progression/Regression by Conditional Adversarial
Autoencoder. *CVPR*, 2017. (UTKFace)

Kärkkäinen & Joo. FairFace: Face Attribute Dataset for Balanced Race, Gender,
and Age. *WACV*, 2021.

Ganin & Lempitsky. Unsupervised Domain Adaptation by Backpropagation. *ICML*,
2015. (Gradient reversal)

Elazar & Goldberg. Adversarial Removal of Demographic Attributes from Text Data.
*EMNLP*, 2018. (The fresh-probe check)

Lucas, Tucker, Grosse & Norouzi. Don't Blame the ELBO! A Linear VAE Perspective
on Posterior Collapse. *NeurIPS*, 2019. (Why the PCA control comes out level)

Locatello et al. Challenging Common Assumptions in the Unsupervised Learning of
Disentangled Representations. *ICML*, 2019.

Xu, White, Kalkan & Gunes. Investigating Bias and Fairness in Facial Expression
Recognition. *ECCV Workshops*, 2020.

Zhang, Dullerud, Roth, Oakden-Rayner, Pfohl & Ghassemi. Improving the Fairness of
Chest X-ray Classifiers. *CHIL*, 2022. (Leveling-down)

Kaushik, Yalavarthi, Ross, Boddeti & Ratha. Shielding Latent Face Representations From Privacy Attacks. IEEE FG 2025. arXiv:2505.12688. — irreversible/compressed face templates still leak age/gender/ethnicity.

Grari, Lamprier & Detyniecki. Fairness without the Sensitive Attribute via Causal Variational Autoencoder. 2021. arXiv:2109.04999. — deliberately builds a VAE latent as a sensitive-attribute proxy.

Wang, Yin, Yap & Zhang. AI Fairness Beyond Complete Demographics: Current Achievements and Future Directions. ECAI 2025. — survey of the fairness-without-demographics regime this work speaks to.

---

## License

Code released for academic use. Datasets remain under their original licenses:
UTKFace and RAF-DB are non-commercial research only; FairFace is CC-BY-4.0.
