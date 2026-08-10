# What the VAE encoded that we did not ask it to

**Revisiting FedAR through a demographic lens.**

Poster project for [MLSS 2026](https://mlss2026.is.tuebingen.mpg.de/), Max Planck
Institute for Intelligent Systems, Tübingen, 31 August – 11 September 2026.

Saranya Bhattacharjee · MSc Artificial Intelligence · Pattern Recognition Lab (LME),
Friedrich-Alexander-Universität Erlangen-Nürnberg

---

## The claim

[FedAR](https://ieeexplore.ieee.org/document/10797675) (Chatterjee, Ghosh,
**Bhattacharjee**, Das & Banerjee, *IEEE Transactions on Affective Computing*, 16(3):1461–1472, 2025)
proposed a federated framework for facial emotion recognition. Face images are
compressed into VAE latent vectors before any gradients are shared; resampling is
then applied to those vectors on the client side to correct class imbalance.

I designed the VAE encoder for that paper. It was trained with reconstruction and
KL loss alone — no demographic supervision, no fairness constraint, no
disentanglement term. Nobody checked what else it encoded.

This project asks three questions:

1. **Does the latent space encode demographic identity?** (Experiment A)
2. **Does that leakage matter for the downstream task?** (Experiment B)
3. **Can it be removed?** (Experiments C and D)

Every question is answered on two corpora that differ in almost every way that
could confound the result.

---

## The honest framing

Two caveats belong in front of the results, not buried at the end.

**RAF-DB's human demographic annotations were not available.** The official
release carries per-image race, age and gender labels, but access was requested
from BUPT and not granted in time. The public Kaggle mirrors redistribute only
the emotion labels. Demographics for RAF-DB are therefore **inferred with the
FairFace classifier** (Kärkkäinen & Joo, WACV 2021) — the same protocol used by
the published fairness-in-FER studies this work compares against. They are model
predictions, not ground truth, and they carry FairFace's own error and bias.

**UTKFace has no emotion labels.** Its downstream task is age-bucket
classification, which stands in for emotion structurally (an imbalanced
multi-class problem over the same frozen latents) but is not the same task.

The two datasets therefore answer complementary questions rather than the same
question twice:

| | UTKFace | RAF-DB |
|---|---|---|
| images | 23,705 | 15,339 |
| demographic labels | ground truth (filename-encoded) | FairFace-inferred |
| emotion labels | none | human-annotated, 7 classes |
| downstream task | age bucket (4 classes) | emotion (7 classes) |
| task imbalance | 2.1 native, 41.5 induced | **17.0 native** |
| answers | *what does the representation encode?* | *does it matter for the task FedAR was built for?* |

RAF-DB's native imbalance ratio of 17.0 is close to the AffectNet setting FedAR
was designed for (18.7), so the resampling intervention engages without any
artificial skewing.

---

## Results

### Experiment A — demographic leakage

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

The RAF-DB numbers are **lower bounds**. The encoder is weaker (half the training
data) and the demographic targets are noisy pseudo-labels; a probe predicting a
noisy target underestimates the underlying signal.

**The headline, on RAF-DB:**

| target | AUROC |
|---|---|
| emotion — *the task* | 0.857 |
| gender | 0.835 |
| race | 0.826 |
| age | 0.813 |

The encoder separates race almost as well as it separates the emotion it was
built to represent, out of the same 128 dimensions.

**Controls.** Every target is reported against a stratified dummy, a
majority-class dummy, and a **label-shuffled null** — the same probe trained and
tested on permuted labels, which must land at chance. It does, on all seven
target/dataset pairs. Cross-validated standard deviations are 0.003–0.014.

### Experiment B — does it matter downstream?

FedAvg simulated on the cached latents. Ten clients, fifty rounds, three
conditions. Race and gender are held out as sensitive attributes and never seen
by any model.

| | UTKFace (age, IMR 41.5) | | RAF-DB (emotion, IMR 17.0) | |
|---|---|---|---|---|
| condition | balanced acc | subgroup gap | balanced acc | subgroup gap |
| no intervention | 0.4657 | 0.3084 | 0.5182 | 0.1778 |
| FedAR resampling | 0.4774 | 0.3046 | 0.5081 | 0.1970 |
| demographic balancing | 0.4642 | 0.3155 | 0.4981 | 0.2042 |

Change in gap relative to no intervention: **−0.0038** and **+0.0071** on
UTKFace, **+0.0193** and **+0.0264** on RAF-DB. Negligible, and inconsistent in
sign — there is no mechanism by which either intervention reaches the disparity.

Demographic balancing is the decisive control: balancing the *sensitive*
attribute directly, rather than the task labels, does not close the gap either.

On RAF-DB the worst-off subgroup **moves** between conditions — Other/Female,
then Black/Male, then Other/Female — so the interventions relocate who absorbs
the disparity rather than reducing it. On UTKFace it stays White/Female
throughout, so this is a RAF-DB observation, not a general finding.

### Experiments C and D — can it be removed?

Gradient reversal layer feeding demographic adversaries, fine-tuned for 20
epochs, sweeping λ ∈ {1, 5, 20, 50}. The encoder is then **frozen and a fresh
probe trained from scratch** — the check from Elazar & Goldberg (2018), which
showed that adversarially removed attributes are often recoverable by a newly
initialised probe: the adversary was defeated, the information was not.

Above-chance signal removed, measured on the deterministic linear probe:

| λ | UTKFace gender / race / age | RAF-DB gender / race / age |
|---|---|---|
| 1 | 0.1% / 4.6% / 2.4% | 3.9% / 8.2% / 4.5% |
| 5 | 5.6% / 11.7% / 9.3% | 6.6% / **18.2%** / 8.0% |
| 20 | 15.3% / **21.3%** / 10.7% | 18.0% / 2.7% / 13.5% |
| 50 | 6.7% / 14.3% / 6.0% | 12.3% / **−1.7%** / **−6.6%** |

Removal peaks around 20% and then declines. At λ=50 on RAF-DB, race and age
became **more** linearly recoverable than before debiasing, while reconstruction
loss rose by 17.6. Full utility cost, negative invariance benefit.

The RAF-DB runs additionally track **emotion accuracy** as the utility axis.
It moves by 1.1–2.5% across every λ. So this is not a case of adversarial
pressure destroying the representation wholesale: the task signal survives and
the demographic signal survives. The intervention simply does not reach what it
is aimed at.

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
        ┌────────────────┬────────┴────────┬─────────────────┐
        ▼                ▼                 ▼                 ▼
    probes.py      federated.py       debias.py      visualize.py
    (Exp. A)         (Exp. B)        (Exp. C + D)    tradeoff_figure.py
                                                     cross_dataset.py
```

Everything after `extract_latents.py` runs on cached 128-dimensional vectors, so
the GPU is needed only twice: once to train the VAE, once to extract.

---

## Files

All scripts live in `code/src/` and are run from that directory.

**Data**

| file | what it does |
|---|---|
| `utkface_dataset.py` | Parses UTKFace filenames (`age_gender_race_timestamp.jpg`) into labels. Run standalone for the composition report. |
| `rafdb_dataset.py` | Joins RAF-DB emotion labels to the FairFace demographics CSV. Tolerates the three folder layouts the public mirrors use. |
| `annotate_demographics.py` | Runs the FairFace ResNet-34 over RAF-DB and writes `data/rafdb_demographics.csv`. GPU-batched, ~13 s for 15k images. |

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
| `federated.py` | Experiment B. FedAvg simulation, three conditions, subgroup breakdown by race × gender. |
| `debias.py` | Experiments C and D. GRL adversarial removal with the fresh-probe recovery check. |

**Figures**

| file | what it does |
|---|---|
| `visualize.py` | Per-dataset: probe performance, composition, LDA projection, UMAP. |
| `tradeoff_figure.py` | Per-dataset: removal against λ and against utility cost. |
| `cross_dataset.py` | Both datasets together. Produces the poster's headline panels. |

---

## Reproducing from scratch

### Environment

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install scikit-learn pandas numpy matplotlib seaborn tqdm imbalanced-learn
pip install umap-learn        # optional; segfaults if TensorFlow is present
```

Built and run on a single NVIDIA RTX 2080 (8 GB) on the FAU CIP Pool.

### Data

**UTKFace** — aligned-and-cropped, 23,708 images, filenames encode the labels.

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

```bash
cd code/src

# ---------- UTKFace ----------
python utkface_dataset.py                       # composition check
python train.py --data_root ../../data/utkface --epochs 50
python extract_latents.py --data_root ../../data/utkface

python probes.py
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
```

End to end on one RTX 2080: roughly 90 minutes, dominated by the eight debias
runs.

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
- **Neither dataset provides subject identifiers**, so subject-disjoint probe
  splits cannot be guaranteed. Both are composed largely of distinct individuals.
- **Race categories are coarse and socially constructed.** UTKFace uses five;
  FairFace's seven are collapsed onto those five (Middle Eastern and
  Latino_Hispanic → Other) so the two corpora share an axis. Raw seven-class
  labels are preserved in the CSV.
- **UTKFace's task is not emotion.** Age-bucket classification is a structural
  stand-in.
- **One architecture, one latent dimension, one seed per configuration.** The
  findings are about a standard reconstruction+KL VAE at 128 dimensions, not
  about VAEs in general.

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

Xu, White, Kalkan & Gunes. Investigating Bias and Fairness in Facial Expression
Recognition. *ECCV Workshops*, 2020.

Zhang, Dullerud, Roth, Oakden-Rayner, Pfohl & Ghassemi. Improving the Fairness of
Chest X-ray Classifiers. *CHIL*, 2022. (Leveling-down)

---

## License

Code released for academic use. Datasets remain under their original licenses:
UTKFace and RAF-DB are non-commercial research only; FairFace is CC-BY-4.0.