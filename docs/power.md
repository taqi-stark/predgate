# Statistical Power & Minimum Group Sizes

PredGate gates slice regressions (`slices.py`) and fairness gap increases (`fairness.py`) using **95% paired bootstrap confidence intervals** so that small-sample noise does not hard-block benign PRs. The unavoidable statistical consequence is that **small evaluation groups cannot reliably detect small drops** — which is why PredGate marks wide slice intervals as `inconclusive` (`slices.on_inconclusive: warn` by default) and emits 80%-power Minimum Detectable Effect (MDE) notes in `fairness.py`.

This guide gives the exact formulas and minimum rows per group required to detect a **3-percentage-point ($\Delta = 0.03$)** and **5-percentage-point ($\Delta = 0.05$)** drop at **80% power** ($1 - \beta = 0.80$) and **95% confidence** (two-sided $\alpha = 0.05$).

---

## 1. Constants

Using the standard normal quantile function $\Phi^{-1}$:

- $z_{1 - \alpha/2} = \Phi^{-1}(0.975) = 1.9600$ (95% two-sided CI / $\alpha = 0.05$)
- $z_{1 - \beta} = \Phi^{-1}(0.80) = 0.8416$ (80% statistical power)
- $(z_{1 - \alpha/2} + z_{1 - \beta})^2 = (1.9600 + 0.8416)^2 = \mathbf{7.8489}$

---

## 2. Paired Per-Slice Drop (`slices.py`)

In `slices.py`, champion and candidate are evaluated on the **same $n$ rows** of a slice. For a binary correctness indicator $d_i = \mathbb{I}(\hat{y}^{\text{champ}}_i = y_i) - \mathbb{I}(\hat{y}^{\text{cand}}_i = y_i) \in \{-1, 0, +1\}$, the variance of the paired difference is:

$$\operatorname{Var}(d_i) = p_d - \Delta^2$$

where:
- $\Delta = \mathbb{E}[d_i] > 0$ is the true metric drop (e.g., `0.03` or `0.05`), and
- $p_d = P(\text{champion correct} \neq \text{candidate correct})$ is the **disagreement rate** between champion and candidate predictions on that slice (typically $0.05$–$0.10$ for a routine retrain, $0.15$–$0.30$ for an algorithm swap, and $\le 0.50$ as an unpaired worst-case bound).

### Formula

Minimum rows $n$ in the slice to distinguish a true drop $\Delta$ from $0$ at 80% power (two-sided $\alpha = 0.05$):

$$n_{\text{paired}}(\Delta, p_d) = \left\lceil \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \, (p_d - \Delta^2)}{\Delta^2} \right\rceil$$

Because PredGate's hard-block (`FAIL`) rule requires `ci_lower > max_drop` (default `max_drop = 0.03`), clearing the `FAIL` gate for a true drop $\Delta > \Delta_{\max}$ has effective margin $\Delta_{\text{eff}} = \Delta - \Delta_{\max}$ (still using $z_{0.975}$ because `ci_lower` is the lower bound of a 95% two-sided CI):

$$n_{\text{FAIL}}(\Delta, \Delta_{\max}, p_d) = \left\lceil \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \, (p_d - \Delta^2)}{(\Delta - \Delta_{\max})^2} \right\rceil$$

### Minimum Rows per Slice (80% Power, $\alpha = 0.05$)

| Champion–Candidate Disagreement Rate ($p_d$) | Typical Scenario | Detect 3-pt Drop ($\Delta = 0.03$ vs. $0$) | Detect 5-pt Drop ($\Delta = 0.05$ vs. $0$) | Trigger Hard `FAIL` on 5-pt Drop ($\text{CI}_{\text{lo}} > 0.03$) |
|---|---|---:|---:|---:|
| **$p_d = 0.05$** | Minor retrain / hyperparameter tweak | **429** | **150** | **933** |
| **$p_d = 0.10$** | Typical feature / data update | **865** | **307** | **1,914** |
| **$p_d = 0.20$** | Architecture / algorithm swap | **1,737** | **621** | **3,876** |
| **$p_d = 0.30$** | Large model shift / noisy cohort | **2,609** | **935** | **5,838** |
| **$p_d = 0.50$** | Unpaired worst-case bound | **4,353** | **1,562** | **9,763** |

> **Why a 3-pt drop cannot trigger `FAIL` at `max_drop = 0.03`**: when $\Delta = \Delta_{\max} = 0.03$, the effective margin is $\Delta - \Delta_{\max} = 0$. Half the time the point estimate falls above $0.03$ (`WARN`) and half below (`PASS`), while `ci_lower` stays below $0.03$ regardless of $n$. If you must hard-block 3-pt drops, set `slices.max_drop` below `0.03` (for example, `preset: strict` uses `max_drop = 0.02`, giving a 1-pt margin) or treat `WARN` as blocking in CI.
>
> **Note on `f1` vs. `accuracy`**: The paired formula above is exact for binary correctness (`accuracy`). For ratio metrics such as `f1` (PredGate's default slice metric), sampling variance depends on positive-class prevalence as well as $p_d$, so treat the table above as a lower bound when slice positive rate is low (`< 20%`).

---

## 3. Two-Group Fairness Gap & Unpaired Proportion Difference (`fairness.py`)

For comparing two independent subgroups of equal size $n$ per group (matching `_compute_min_detectable_gaps` in [`src/predgate/checks/fairness.py`](../src/predgate/checks/fairness.py)), the variance of the difference in subgroup rates is $2 p (1 - p) / n$, giving:

$$n_{\text{per group}}(\Delta, p) = \left\lceil \frac{2 \, (z_{1-\alpha/2} + z_{1-\beta})^2 \, p(1 - p)}{\Delta^2} \right\rceil$$

where $p$ is the baseline rate (selection rate for demographic parity; worst-case $p = 0.50$ gives $p(1-p) = 0.25$ for TPR/FPR in equalized odds, where $n$ counts **positive-label rows** per group).

### Minimum Rows per Group (Equal Groups, 80% Power, $\alpha = 0.05$)

| Baseline Rate $p$ | Variance $p(1 - p)$ | Detect 3-pt Gap ($\Delta = 0.03$) | Detect 5-pt Gap ($\Delta = 0.05$) |
|---|---:|---:|---:|
| **$p = 0.50$** (worst-case binomial / EO) | `0.2500` | **4,361** / group | **1,570** / group |
| **$p = 0.25$** (e.g., Adult `>50K` rate) | `0.1875` | **3,271** / group | **1,178** / group |
| **$p = 0.10$** (rare-event positive rate) | `0.0900` | **1,570** / group | **566** / group |

For unequal group sizes $n_1 \neq n_2$, replace $2/n$ with $1/n_1 + 1/n_2$; the minimum detectable gap at 80% power is:

$$\text{MDE}_{80\%} = (z_{1-\alpha/2} + z_{1-\beta}) \sqrt{p(1 - p) \left(\frac{1}{n_1} + \frac{1}{n_2}\right)}$$

---

## 4. What This Means for Real Test Sets

Comparing the five benchmark datasets against the tables above explains the empirical detection rates in [`docs/benchmark.md`](benchmark.md):

| Dataset | Test Set $N$ | Smallest Protected Group (`min_n`) | Fairness $\text{MDE}_{\text{DP}}$ (80% Power) | Practical Takeaway |
|---|---:|---:|---:|---|
| **German Credit** | `300` | `93` (`sex=female`) | `0.160` | Far underpowered for 3–5 pt drops; `cohort_feature_corruption` was caught only `5/30` FAIL (`12/30` Flagged) and slices are flagged `inconclusive`. |
| **Bank Marketing** | `1,500` | `169` (`marital=divorced`) | `0.076` | Catches large regressions (`>= 8–10 pt`), emits low-power / inconclusive notes on smaller 3–5 pt shifts. |
| **UCI Adult** | `1,500` | `495` (`sex=Female`) | `0.066` | Adequate for paired 5-pt slice drops at low $p_d$; emits MDE note for `fail_gap_increase = 0.05`. |
| **COMPAS** | `1,500` | `594` (`race=Caucasian`) | `0.074` | Catches strong cohort & fairness regressions reliably (`29–30/30`), still below 4,361/group for 3-pt unpaired gaps. |
| **Synthetic Credit** | `1,800` | `814` (`gender=female`) | `0.038` | Sufficient power for 5-pt DP shifts (no MDE low-power note); still misses rare-event $\tau=0.75$ shifts where base rate is ~0.2%. |

### Practical Rule of Thumb
- Aim for **$\ge 500$ test rows per slice** to catch 5-point regressions reliably during routine retrains, and **$\ge 1,500$–$2,000$ rows per slice** to hard-block 5-point drops above a 3-point allowance (`max_drop = 0.03`) or during model-family swaps.
- Keep `slices.on_inconclusive: warn` (the default) whenever you have small slices so wide confidence intervals are surfaced prominently instead of silently passing.
