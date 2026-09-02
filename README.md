<h1 align="center">ElmanGuard</h1>

<p align="center">
  <strong>Sequence-based intrusion detection with a classic Elman recurrent network — and the controls that say whether it is really detecting anything.</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776ab?logo=python&logoColor=white">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.21-ff6f00?logo=tensorflow&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-app-ff4b4b?logo=streamlit&logoColor=white">
  <img alt="Dataset" src="https://img.shields.io/badge/dataset-CICIDS2017-1baf7a">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-2a78d6">
</p>

<p align="center"><a href="README.pt-br.md">🇧🇷 Leia em português</a></p>

---

An Elman network reads a window of 16 consecutive network flows arriving at a
monitored server and predicts whether that window contains an attack. Trained on
Monday–Wednesday of CICIDS2017, thresholded on Thursday, evaluated on Friday —
whose attack types it has never seen.

The point of the repository is not the detector. It is the four controls around
it, and what they reveal.

## Three findings

**1 · Temporal order carries no measurable signal here — and that null is credible.**
Shuffling the flows inside every window destroys the order while keeping the
contents. The Elman's PR-AUC falls by **0.055**; the seed-to-seed standard
deviation of the same measurement is **0.117**. The effect is smaller than the
noise. What makes this worth stating is the instrument check below: the same
harness detects a *planted* temporal signal at PR-AUC 0.985.

**2 · Most of the apparent skill is host recognition, not attack detection.**
Pooled across servers the models reach PR-AUC 0.31–0.55. Computed *inside* each
host — where "which machine is this" carries no information — every neural model
sits at or below chance on every host. Only a logistic regression on the single
heavily attacked server clears it, at ROC-AUC 0.694.

**3 · A linear model beats every neural architecture, and the four networks are
indistinguishable from each other.**

![Architecture comparison](assets/architecture_comparison.png)

<details>
<summary><strong>The app</strong> (click to expand)</summary>

![Streamlit app](assets/app_screenshot.png)

The test day at the attacked server. The whole day is shaded because 98.6% of its
windows contain attack traffic — and the model's probability spends long stretches
near zero inside it. That picture is the within-host control, drawn.

</details>

## The setup

| | |
|---|---|
| Data | CICIDS2017 `TrafficLabelling`, all 8 day files, **2,827,876 flows** after repair |
| Features | **78** numeric flow statistics; IPs, ports, `Flow ID` and timestamp excluded |
| Sample | 16 consecutive flows arriving at one monitored host, on one day |
| Split | train Mon–Wed (82,968 windows, 3.2% attack) · threshold Thu (29,955, 1.1%) · test Fri (23,858, 8.1%) |
| Test attacks | PortScan, DDoS, Bot — **none of them seen in training** |
| Repeats | every number below is the mean ± sd over 3 seeds |

## Results

PR-AUC on the unseen-attack test day. The prevalence floor — what a detector
that flags everything would score — is 0.081.

| model | PR-AUC | ROC-AUC | recall @ 1% FPR | precision | params |
|---|---|---|---|---|---|
| **logistic regression** | **0.546** | 0.775 | **0.468** | 0.825 | 19,969 |
| window MLP (wide) | 0.349 ± 0.008 | 0.601 | 0.258 ± 0.019 | 0.822 | 80,001 |
| window MLP | 0.347 ± 0.050 | 0.603 | 0.155 ± 0.124 | 0.841 | 8,751 |
| **Elman RNN** | 0.341 ± 0.117 | 0.638 | 0.177 ± 0.097 | 0.815 | 9,217 |
| last-flow MLP | 0.309 ± 0.020 | 0.758 | 0.174 ± 0.056 | 0.767 | 9,201 |
| prevalence floor | 0.081 | 0.500 | — | 0.081 | — |

The four neural models span 0.309 to 0.349 — a range narrower than the Elman's
own standard deviation across seeds. **Any paper reporting that its RNN beats an
MLP by two points on CICIDS2017 is reporting the seed.**

Recall by attack type tells you where the 0.55 comes from:

| attack (unseen) | Elman | window MLP | logistic |
|---|---|---|---|
| DDoS | 0.530 | 0.684 | 0.808 |
| PortScan | 0.000 | 0.000 | 0.431 |
| Bot | 0.000 | 0.000 | 0.036 |

DDoS transfers because the training days are full of DoS. PortScan and Bot do
not transfer at all for the neural models. This is what "generalises to unseen
attack types" actually looks like.

## The controls

### 1 · Within-host: how much is recognition?

Pooled across servers, a model can score well simply by learning *which machines
get attacked* — in this capture the web server is under attack 95% of the week
and the other hosts essentially never are. Inside one host that information is
constant, so within-host numbers measure detection.

ROC-AUC, test day, per monitored host (0.5 = chance):

| host | attack rate | Elman | window MLP | logistic |
|---|---|---|---|---|
| 192.168.10.50 | 98.6% | 0.561 | 0.451 | **0.694** |
| 192.168.10.9 | 13.8% | 0.426 | 0.399 | 0.509 |
| 192.168.10.15 | 8.0% | 0.483 | 0.340 | 0.328 |
| 192.168.10.5 | 4.9% | 0.417 | 0.360 | 0.415 |
| 192.168.10.14 | 4.3% | 0.316 | 0.192 | 0.303 |
| 192.168.10.8 | 3.5% | 0.464 | 0.266 | 0.341 |
| 205.174.165.73 | 56.9% | 0.220 | 0.100 | 0.195 |

Read across: **the pooled PR-AUC of 0.34–0.55 is very largely host identity.**
Strip it out and one cell in the table beats chance.

### 2 · Order shuffle: is the recurrence earning its place?

| model | PR-AUC ordered | PR-AUC shuffled | drop | seed sd |
|---|---|---|---|---|
| Elman RNN | 0.341 ± 0.117 | 0.286 ± 0.039 | 0.055 | 0.117 |
| window MLP | 0.347 ± 0.050 | 0.305 ± 0.024 | 0.042 | 0.050 |

Both drops are smaller than the run-to-run spread. At the flow-window level on
this capture, the order of flows is not carrying information the models can use.

### 3 · The instrument check — the control that saved the project

A null result is worthless unless the instrument can detect a positive one.
`synthetic.order_only_dataset` builds windows where **order is the only signal**:
every window comes from the same AR(1) process, negatives are the same values
permuted in time, so mean, variance, min, max and even the last-step distribution
are identical between classes.

| configuration | PR-AUC on the order-only probe |
|---|---|
| earlier default (48 units, dropout 0.3, lr 1e-3) | **0.483** — chance |
| current default (64 units, dropout 0.2, lr 3e-3) | **0.985** |

The first configuration would have produced a confident, published, and
completely false conclusion: *"the recurrence adds nothing."* It was not a
finding, it was an undertrained network. The probe is now a test
(`tests/test_harness_detects_planted_signal.py`) that fails if the harness ever
loses the ability to see a temporal effect.

A second reproducibility bug surfaced the same way: seeding inside `fit` but not
before the weights were built made two runs of the same configuration differ by
more than 0.25 PR-AUC. Hence three seeds on every number.

### 4 · Grouping by source instead of destination

Nearly every attack flow in CICIDS2017 comes from one host, `172.16.0.1`, whose
traffic is **99.4% attack**. Group sequences by source and the question silently
becomes "is this the attacker's machine". Measured, single run:

| grouping | PR-AUC | precision |
|---|---|---|
| by destination (monitored server) | 0.470 | 0.815 |
| by source | 0.291 | **1.000** |

I expected source-grouping to inflate PR-AUC and it did not; what it does is push
precision to a perfect 1.000, which is the fingerprint of a model that has found
an identity shortcut and fires only when it is certain. Reported as measured
rather than as predicted.

## Method

Full detail in [`docs/how_it_works.en.md`](docs/how_it_works.en.md). The short
version:

**The parser fixes six documented defects** in the published CSVs — leading
spaces in 70 of 85 column names, latin-1 bytes in the web-attack labels, header
rows spliced mid-file, 288,602 empty padding rows, 12-hour timestamps with no
AM/PM marker (`1:02` is 13:02), and an ambiguous day/month order resolved by
checking the weekday against the filename. Each one has a test.

**Duplicate rows are kept, deliberately.** De-duplicating is the standard advice
and it comes from the row-classification framing. Here the split is chronological
so duplicates cannot straddle it — and a brute force *is* a run of near-identical
flows. Removing them would delete the signal. The count is reported instead.

**Windows never cross a host or a day**, so the weekday split cannot put the same
flows on both sides.

**Every architecture gets a matched parameter budget**, because a flattened
16-step window is 16× wider than one step and an unmatched dense layer would win
on capacity rather than on architecture. `window_mlp_wide` is the deliberately
over-provisioned control at 9× the parameters.

**Early stopping watches held-out windows from the training days**, not the
calibration day. Each weekday holds different attacks, so a validation signal
from another day measures transfer, not convergence, and training stops on noise.

```
Input (16, 78) → SimpleRNN(64, tanh) → Dropout(0.2) → Dense(1, sigmoid)
9,217 parameters · binary cross-entropy · Adam 3e-3 · class-weighted
```

## Run it

CICIDS2017 is not redistributable, so the raw capture is not in the repository.

```bash
git clone https://github.com/AlanFMF/elman-guard.git
cd elman-guard
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest -m "not slow"        # 22 tests, no dataset needed
pytest                      # adds the instrument checks (trains small models)
```

With the data (download `GeneratedLabelledFlows.zip` from the
[CIC](https://www.unb.ca/cic/datasets/ids-2017.html) — **not** the
`MachineLearningCVE` version, which has the IP and timestamp columns stripped
and cannot be sequenced):

```bash
python scripts/prepare_data.py --raw path/to/TrafficLabelling --out data/processed
python scripts/run_experiment.py --split chronological --seeds 0 1 2
python scripts/run_experiment.py --split random --seeds 0 1 2   # the inflated protocol
python scripts/export_demo.py
python scripts/make_figures.py
streamlit run app.py
```

The app is bilingual and shows the attack probability through the test day for a
chosen server, with the ground-truth attack periods shaded, alongside all of the
measured results above.

## Limitations

- **One week, one testbed, 2017.** The traffic is generated, the attacks are
  scripted, and the topology is fixed. Nothing here transfers to a real network
  without re-measurement.
- **Known dataset defects.** Beyond the six the parser repairs, published work
  documents labelling errors and a flow-construction bug in the CICFlowMeter
  release used to build these CSVs. A corrected re-release exists; this project
  uses the original, so its absolute numbers carry that caveat.
- **The window is 16 flows.** A slow attack spread over hours may simply not fit
  inside the context. Longer windows were not swept.
- **Three seeds** is enough to show the gaps are noise, not enough to resolve
  small real differences.
- **Only the chronological and random protocols** were compared. Host-disjoint
  splitting is impossible here: one machine sends nearly all attack traffic.

### What I would try next

Longer and multi-scale windows; a per-host baseline model so the score is
"unusual *for this server*" rather than "unusual"; and the corrected CICIDS2017
re-release, to see which of these findings survive better labels.

## ⚠️ Disclaimer

A research demonstration of evaluation methodology for sequence models, **not** a
production intrusion detection system. Do not deploy it on a real network.

## License

MIT — see [LICENSE](LICENSE).
