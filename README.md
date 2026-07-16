# DLPEA: Data Leak Prevention with Evolutionary Algorithms

This repository contains the experimental source code used in the dissertation research on the method and means of data leak detection in corporate networks based on evolutionary algorithms.

The implementation covers four experimental stages:

1. classification of documents by confidentiality level: a genetic algorithm evolves a compact set of production IF-THEN rules, compared with five machine-learning baselines on real corpora;
2. controllability of the rule-based policy: when a new identifier type appears in the document flow, an expert appends a single rule without retraining and without labeled examples;
3. behavioral profiling of users: individual adaptive profiles whose configuration (feature mask, weights, threshold, forgetting rate) is optimized by a genetic algorithm, compared with global anomaly detectors on the CERT Insider Threat dataset;
4. concept-drift detection with a multivariate Kolmogorov-Smirnov detector and a confirmation rule, and evolutionary adaptation of classification policies on standard synthetic and real data streams.

## Repository structure

```text
DLPEA/
├── src/
│   ├── ga_classifier.py        # GA rule-based classifier (dissertation Section 3.1)
│   ├── behavior.py             # adaptive behavioral profile + GA optimizer (Section 3.3)
│   ├── drift.py                # multivariate KS drift detector with confirmation (Section 3.2)
│   ├── policy_adapt.py         # evolutionary policy adaptation, prequential protocol (Section 3.2)
│   ├── datasets.py             # document features, synthetic PII corpus, policy drift streams
│   ├── ai4privacy_corpus.py    # loader of the ai4privacy PII-Masking-200k corpus
│   ├── cert_features.py        # user-day features from the CERT r4.2 logs
│   ├── real_streams.py         # loaders of the Electricity (Elec2) and INSECTS streams
│   ├── synth_streams.py        # RandomRBF covariate-drift stream generators
│   ├── run_e1.py               # shared cross-validation harness for experiment E1
│   ├── run_e1_real.py          # E1: classification on real corpora (Section 4.2)
│   ├── run_e1_patch.py         # E1: expert patching scenario (Section 4.2)
│   ├── run_e2_cert.py          # E2: insider identification on CERT r4.2 (Section 4.3)
│   ├── run_e3_synth.py         # E3: drift detection on RandomRBF streams (Section 4.4)
│   ├── run_e3_real.py          # E3: drift detection on INSECTS streams (Section 4.4)
│   ├── run_e4_real.py          # E4: policy adaptation on Elec2 and INSECTS (Section 4.4)
│   ├── run_e4_policy.py        # E4: policy adaptation on policy-structured streams (Section 4.4)
│   ├── make_e4_plot.py         # figure: prequential accuracy curves on a policy stream
│   └── make_e4_real_plot.py    # figure: prequential accuracy curves on Elec2
├── data/
│   └── cert/
│       └── README.md
├── results/                    # exact experiment outputs cited in Chapter 4
├── digest_ch4.py               # rebuilds a text digest of all results
├── README.md
└── requirements.txt
```

## Experimental modules

### `src/run_e1_real.py` (dissertation Section 4.2)

Classification of documents by confidentiality level on two real corpora. The script:

- loads ai4privacy PII-Masking-200k and labels a document as confidential if it contains a critical identifier (payment card, account number or IBAN, SSN, password, cryptocurrency wallet);
- loads 20 Newsgroups as a proxy of topical confidentiality;
- builds the feature space: TF-IDF terms, identifier patterns (regular expressions with Luhn validation), keyword and metadata features;
- runs 5-fold stratified cross-validation of the GA rule classifier against RandomForest, GradientBoosting, LinearSVM, LogisticRegression, and NaiveBayes;
- measures F1, precision, recall, fit and prediction time, and interpretability (number of rules, conditions per rule, share of decisions explained by at most two rules);
- runs the sensitivity study of the regularization coefficient lambda;
- exports `results/e1_real.json`.

### `src/run_e1_patch.py` (dissertation Section 4.2)

Controllability scenario: a new critical identifier type (IBAN) appears in the document flow with zero labeled examples, and twin decoy documents (the same wording with a 27-digit settlement reference instead of IBAN) suppress digit-density shortcuts. The script:

- trains the GA rule classifier and the baselines on a corpus without IBAN positives;
- applies a single expert rule `IF pat_iban > 0 THEN confidential` to the trained rule set;
- measures recall on the new document type and overall F1 before and after the patch;
- measures how many labeled IBAN documents RandomForest needs (10/50/200) to catch up with the patch;
- exports `results/e1_patch.json`.

### `src/run_e2_cert.py` (dissertation Section 4.3)

User-level insider identification on the real CERT Insider Threat r4.2 dataset. The script:

- builds user-day behavioral features (logons, after-hours and weekend activity, distinct PCs, USB connections, file copies, emails, external recipients, mail volume) from the raw logs;
- optimizes the configuration of the individual adaptive profile (feature mask, weights, alarm threshold, forgetting rate) with a genetic algorithm on the training period;
- ranks users by the peak daily deviation from their own baseline on the test period;
- compares the GA profile with global Isolation Forest, LOF, and One-Class SVM detectors using AP, precision@k, recall@k, and AUC;
- exports `results/e2_cert.json`.

### `src/run_e3_synth.py` (dissertation Section 4.4)

Concept-drift detection on standard RandomRBF streams (river generator) with exactly known drift positions. The script:

- generates 10 abrupt and 10 gradual covariate-drift streams with 4 concepts each;
- runs the proposed detector (per-feature Kolmogorov-Smirnov tests with Bonferroni correction and a confirmation rule of k consecutive signals);
- runs the ablation without confirmation and the ADWIN, DDM, EDDM, and KSWIN baselines;
- scores precision, recall, F1, detection delay, and false alarms within a tolerance window;
- exports `results/e3_synth.json`.

### `src/run_e3_real.py` (dissertation Section 4.4)

The same drift-detection protocol on the real INSECTS streams (Souza et al., 2020) with change points published in the primary source; averages the metrics over four stream variants and exports `results/e3_real.json`.

### `src/run_e4_real.py` (dissertation Section 4.4)

Evolutionary policy adaptation on real non-stationary streams. The script:

- evaluates three strategies with the prequential test-then-train protocol on Electricity (Elec2) and INSECTS-Abr: a static policy, the proposed adaptive scheme (the drift detector triggers retraining on a fresh window), and an incremental SGD baseline;
- reports prequential accuracy, the number of retrainings, and the improvement of the adaptive scheme over the static policy;
- exports `results/e4_real.json`.

### `src/run_e4_policy.py` (dissertation Section 4.4)

The same three strategies on policy-structured drift streams (a disjunction of conjunctions that changes at drift points, generated by `datasets.make_drift_stream`). Exports `results/e4_policy.json` and the prequential-accuracy figure `results/e4_prequential.png`.

## Core method implementations

- `src/ga_classifier.py`: the GA classifier over production IF-THEN rules; a chromosome is a rule set, fitness is F1 minus a complexity penalty; heuristic and greedy sequential-covering initialization, tournament selection, crossover on rule sets, mutation operators including directed specialization, diversity injection, multistart.
- `src/behavior.py`: the online behavioral profile (exponentially weighted mean and variance per user, positive z-deviations) and the GA optimizer of the detector configuration.
- `src/drift.py`: the multivariate KS drift detector with Bonferroni correction and the confirmation rule; wrappers for the river baseline detectors; alarm scoring.
- `src/policy_adapt.py`: prequential evaluation of the static, adaptive (exploitation and exploration modes), and incremental strategies.
- `src/datasets.py`: feature extraction for documents, the synthetic PII corpus generator used in the patching scenario, and policy-structured drift streams.

## Requirements

- Python 3.12 or newer
- NumPy
- SciPy
- Scikit-learn
- Matplotlib
- river
- datasets (Hugging Face, only for the ai4privacy corpus)

Create and activate a virtual environment, then install the dependencies from the repository root:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Dataset preparation

- **ai4privacy PII-Masking-200k**: downloaded automatically from Hugging Face on the first run of `run_e1_real.py`.
- **20 Newsgroups**: downloaded automatically by scikit-learn.
- **Electricity (Elec2)** and **INSECTS**: downloaded automatically by the river library on first use.
- **RandomRBF**: generated on the fly, no download required.
- **CERT Insider Threat r4.2**: not included and not downloaded automatically (about 5 GB). Place the extracted files as described in [`data/cert/README.md`](data/cert/README.md) before running `run_e2_cert.py`.

## Running the experiments

Run each experiment from the repository root as a module:

```bash
python -m src.run_e1_real
python -m src.run_e1_patch
python -m src.run_e2_cert
python -m src.run_e3_synth
python -m src.run_e3_real
python -m src.run_e4_real
python -m src.run_e4_policy
```

On Windows set `PYTHONUTF8=1` before running, because the scripts print Ukrainian log messages.

The GA runs and the drift-detection sweeps are computationally intensive; a full pass over all experiments takes several hours on a desktop CPU. `run_e2_cert.py` additionally makes a streaming pass over the raw CERT logs.

## Results

The `results/` directory contains the exact outputs cited in Chapter 4 of the dissertation:

```text
results/e1_real.json          # E1: classification on ai4privacy and 20NG
results/e1_patch.json         # E1: expert patching scenario
results/e2_cert.json          # E2: insider identification on CERT r4.2
results/e3_synth.json         # E3: drift detection on RandomRBF
results/e3_real.json          # E3: drift detection on INSECTS
results/e4_real.json          # E4: policy adaptation on Elec2 and INSECTS-Abr
results/e4_policy.json        # E4: policy adaptation on policy-structured streams
results/e4_prequential.png    # prequential accuracy curves (Figure 4.1)
results/DIGEST_ch4.txt        # text digest of all results (digest_ch4.py)
```

Re-running the scripts overwrites these files. All experiments use fixed random seeds; the exact numbers can still vary slightly across hardware and library versions.

## Research context

The repository supports the experimental validation of:

- interpretable classification policies in the form of production rule sets evolved by a genetic algorithm;
- expert editability of the evolved policies: zero-label reaction to new identifier types;
- individual adaptive behavioral profiles against population-level outlier detection for insider identification;
- multivariate Kolmogorov-Smirnov drift detection with a false-alarm-suppressing confirmation rule;
- evolutionary adaptation of security policies with exploitation and exploration modes under concept drift.

## Author

**Petro Vizhevskyi**
Khmelnytskyi National University

Repository: `https://github.com/sp4rd4/DLPEA`
