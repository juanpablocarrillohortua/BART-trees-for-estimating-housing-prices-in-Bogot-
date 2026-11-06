# Estimating Housing Prices in Bogotá with BART

- **Course:** Estadística Bayesiana, Universidad Externado de Colombia, 2026-2
- **Team:** Julian Rodriguez Ramirez, Juan Pablo Carrillo Hortua, Samuel Acosta Morales
- **Results as of:** commit `3900a4d`

Every number below comes from the stored outputs of
[bayesiana_primera_pt.ipynb](../notebooks/bayesiana_primera_pt.ipynb) and
[BART_Trees.ipynb](../notebooks/BART_Trees.ipynb), or from the
[scraper README](../scraper/README.md) and [project README](../README.md).
No notebook was re-run to write this report.

## Summary

We compared BART, with and without adaptive outlier trimming, against a median
baseline, OLS, Random Forest, XGBoost and LightGBM for predicting asking prices of
residential listings. Scores come from 5-fold cross-validation on 17,711 training
listings.

- **BART did not win on prediction.** LightGBM had the lowest RMSE (COP 336.0M) and
  highest R² (0.830), and XGBoost had the lowest MAPE (0.158). BART(Atr) scored
  RMSE COP 345.0M and MAPE 0.168, ahead of OLS but behind LightGBM in every fold on
  both metrics.
- **The benefit of trimming is unclear.** BART(Atr) beat classic BART on MAPE and MAE
  in 4 of 5 folds, but on RMSE in only 2 of 5. It took about 63 times longer to fit,
  and the two used different MCMC settings.
- **The BART chains did not converge.** R-hat reaches 2.71 and effective sample sizes
  are around 3 to 5. The predictive scores are still valid, but the posterior
  uncertainty, which is the main reason to choose BART, cannot yet be trusted.
- **The final evaluation is still open.** No real model has been scored on the 4,428
  held-out listings, and BART's prediction intervals have not been computed.

For point prediction alone, the current evidence favours LightGBM or XGBoost. BART's
case depends on fixing convergence and showing that its intervals are calibrated.

## 1. Formulation

### 1.1 Problem

Estimate the asking price of a residential property in Bogotá and nearby municipalities
from its listing, and test whether BART with adaptive trimming (Cao, Wu & Wang, 2024)
improves out-of-sample accuracy over established econometric and machine-learning
methods.

- **Response:** `price_amount`, the asking price in COP, modelled as $\log y_i$.
- **Data:** FincaRaíz sale listings scraped in August 2026.
- **Unit:** one listing. These are asking prices, not transaction prices.

### 1.2 Conjugate models

The first stage fits conjugate models to samples of $n = 300$ listings.

**Beta-Binomial**, for the share of houses:

$$X \mid \theta \sim \text{Binomial}(n, \theta), \quad \theta \sim \text{Beta}(a, b)
\quad\Rightarrow\quad \theta \mid x \sim \text{Beta}(a + x,\ b + n - x)$$

**Dirichlet-Multinomial**, for price segment and property type:

$$\mathbf{X} \mid \boldsymbol{\theta} \sim \text{Multinomial}(n, \boldsymbol{\theta}), \quad
\boldsymbol{\theta} \sim \text{Dirichlet}(\boldsymbol{\alpha})
\quad\Rightarrow\quad \boldsymbol{\theta} \mid \mathbf{x} \sim \text{Dirichlet}(\boldsymbol{\alpha} + \mathbf{x})$$

**Normal-Inverse-Gamma**, for log price and log built area:

$$y_i \mid \mu, \sigma^2 \overset{iid}{\sim} N(\mu, \sigma^2), \quad
\mu \mid \sigma^2 \sim N(\mu_0, \sigma^2 / \kappa_0), \quad
\sigma^2 \sim \text{Inv-Gamma}(a_0, b_0)$$

with posterior $\text{NIG}(\mu_n, \kappa_n, a_n, b_n)$, where

$$\kappa_n = \kappa_0 + n, \quad
\mu_n = \frac{\kappa_0 \mu_0 + n \bar{y}}{\kappa_n}, \quad
a_n = a_0 + \frac{n}{2}, \quad
b_n = b_0 + \frac{(n-1)s^2}{2} + \frac{\kappa_0 n (\bar{y} - \mu_0)^2}{2 \kappa_n}$$

### 1.3 BART

BART models the log price as a sum of $m$ regression trees:

$$\log y_i = \sum_{j=1}^{m} g(x_i; T_j, M_j) + \varepsilon_i, \qquad
\varepsilon_i \overset{iid}{\sim} N(0, \sigma^2)$$

The priors follow Chipman et al. (2010), as implemented in
[bart_article.py](../utils/bart_article.py) on top of `stochtree`:

| Component | Prior | Values used |
|---|---|---|
| Tree shape | P(split at depth $d$) $= \alpha (1 + d)^{-\beta}$ | $m = 50$, $\alpha = 0.95$, $\beta = 2$ |
| Leaf values | $\mu \sim N(0, \sigma_\mu^2)$, with $\sigma_\mu = \text{range}(y) / (2k\sqrt{m})$ fixed | $k = 2$ |
| Noise | $\sigma^2 \sim \nu\lambda / \chi^2_\nu$, with $\lambda$ set so that $P(\sigma < \hat\sigma_{OLS}) = q$ | $\nu = 3$, $q = 0.9$ |

### 1.4 Adaptive trimming: BART(Atr)

BART(Atr) fits BART, flags the observations whose residuals are too large to be
consistent with normal noise, refits without them, and repeats. With $r_{(j)}$ the
absolute residuals sorted in ascending order:

**Expected quantiles (eq. 4)**, from the folded standard normal:

$$z_j = \Phi^{-1}\!\left(\frac{1 + p_j}{2}\right), \qquad p_j = \frac{j}{n + 1}$$

**Robust noise variance (eq. 6)**, from the smallest 25% of residuals, $k = \lfloor 0.25 n \rfloor$:

$$\hat\sigma^2 = \frac{\sum_{j=1}^{k} r_{(j)}^2}{\sum_{j=1}^{k} z_j^2}$$

**Clean share (eq. 7)**, with tolerance $Q = 1.28$:

$$R_i = \frac{\sum_{j \le i} r_{(j)}^2}{\hat\sigma^2 \sum_{j \le i} z_j^2}, \qquad
\hat{p} = \frac{|\{i : R_i \le Q\}|}{n}$$

**Weights (eq. 8)**, keeping observations at or below the critical residual:

$$w_i = \mathbb{1}\left(|r_i| \le r_{(\lfloor n \hat{p} \rfloor)}\right)$$

The loop stops when $|\hat{p}_t - \hat{p}_{t-1}| / \hat{p}_t < 10^{-4}$ or after 20
iterations. Only the final fit runs all chains; intermediate refits use one chain.

The implementation adds one safeguard that is **not in the paper**: $\hat{p}$ is floored
at `p_min = 0.5`, so one unstable iteration cannot discard most of the data.

## 2. Analytical approach

**Two stages.** The conjugate models establish what the data say about market
composition, price and area, and how far that departs from external priors. The
predictive comparison then asks how well price can be estimated from listing features.

**Log price.** Raw prices are extremely right-skewed: before cleaning, the mean is about
COP 3.64 billion and the median COP 860M. Models fit on the log scale and predictions
return to pesos through `exp`, with no retransformation correction.

**Why BART.** OLS fails the RESET test for functional form, which points to
non-linearity. BART can capture interactions between area, stratum and location without
specifying them, and offers a posterior over predictions. Trimming targets the data
errors and extreme listings found throughout the portal data.

**Keeping the test data out of training.** Row-level steps that estimate nothing happen
before the split. Everything estimated from data (imputation, outlier fences, encodings,
feature selection) happens inside the pipeline and is relearned in every fold.

**Benchmarks.** A median baseline sets the floor, OLS is the econometric reference, and
Random Forest, XGBoost and LightGBM are the machine-learning references. All use the same
preprocessing and the same folds, so differences come from the model and the number of
features. No hyperparameter search was run; doing it properly would require nested
cross-validation.

## 3. Methodology

### 3.1 Data acquisition

A purpose-built [scraper](../scraper/README.md) collected the listings. The completed
sale run parsed 39,206 records, removed 7,954 duplicates and kept 31,252 listings, at a
measured coverage of 91.3%. Contact details are never collected.

### 3.2 Cleaning before the split

From [BART_Trees.ipynb](../notebooks/BART_Trees.ipynb):

| Step | Rows remaining | % of initial |
|---|---:|---:|
| Initial | 31,252 | 100.00 |
| COP prices only | 31,251 | 100.00 |
| Residential only | 22,853 | 73.12 |
| Valid stratum (1 to 6, or missing) | 22,754 | 72.81 |
| Price between COP 30M and COP 20,000M | 22,715 | 72.68 |
| Price within 2.5 × IQR of own locality | 22,157 | 70.90 |
| Inside Cundinamarca | 22,141 | 70.85 |
| Locality can be placed | 22,139 | 70.84 |

Identifiers, the address and URL were dropped, along with `total_monthly_cost`. So were
`floors_total` (71% missing), `age_bracket_label` (duplicates the code),
`agency_name` (reflects the seller, not the property), `updated_at` and `price_per_m2`
(computed from the target, so it would leak). Impossible values in `bedrooms`,
`bathrooms`, `parking_spaces` and `floor` were corrected or set to missing.

The conjugate models use the separately prepared file from
[data_cleaning.ipynb](../notebooks/data_cleaning.ipynb) and
[data_preparation.ipynb](../notebooks/data_preparation.ipynb), which has 20,442 listings.

### 3.3 Split and spatial features

The 22,139 listings were split 80/20 with a fixed seed into **17,711 training** and
**4,428 test** rows. Two distance features were added using the haversine formula, from
[bogota_landmarks.yaml](../geodata/bogota_landmarks.yaml):

- `dist_nearest_positive_km`: distance to the nearest of 61 amenities such as shopping
  centres and parks.
- `dist_nearest_negative_km`: distance to the nearest of 59 disamenities such as
  detention centres and waste facilities.

### 3.4 Pipeline

```
Pipeline
├── imputation
│   ├── simple    mode, 50/50 bimodal draw, 1-NN, constants
│   └── mice      IterativeImputer + ExtraTrees
└── outliers      OutlierFilterRegressor (drops rows while fitting only)
    └── TransformedTargetRegressor (log / exp)
        └── Pipeline
            ├── encoding    one-hot, frequency, months listed
            ├── amenities   two flags and a count
            ├── selection   SelectKBest on mutual information (k = 6 or 18)
            └── model
```

**Imputation**

| Variables | Method |
|---|---|
| `city`, `construction_state`, `bedrooms`, `bathrooms` | Mode |
| `age_bracket_code` | 50/50 draw between the 16-30 and over-30-years brackets |
| `stratum` | 50/50 draw between strata 3 and 4 |
| `neighborhood` | Nearest neighbour on coordinates |
| `amenities`, `publisher_type` | Constants "Ninguna" and "Particular" |
| `admin_fee`, areas, `parking_spaces`, `floor` | MICE with ExtraTrees, assuming MAR |

**Outlier filter.** While fitting only, this drops rows with built or private area outside
10 to 5,000 m², then rows beyond 3 × IQR on price, administration fee, price per m² and
both areas. Using 3 rather than 1.5 avoids removing legitimate luxury homes. In the
full-training fit it dropped 71 rows by rule and 1,683 by IQR, keeping 15,957 of 17,711.
At prediction time every row passes through.

**Encoding.** One-hot for `construction_state` (top 4 plus "Otros"), `city`,
`publisher_type` and `property_type`. Frequency for `locality` and `neighborhood` (top 20
plus "Otros"). `published_at` becomes months listed up to August 2026, and
`area_private_m2` is removed for collinearity. This yields 32 features.

**Linear models only.** Area and administration fee enter in logs, and each feature is
clipped to its training range, because a data-entry error in a test row would otherwise
produce an infinite prediction.

### 3.5 Model settings

| Model | Settings |
|---|---|
| Baseline | Median of log price |
| OLS | `LinearRegression`; refit with `statsmodels` for inference, with HC3 errors |
| Classic BART | `trim=False`, 2 chains, 2,000 burn-in, 4,000 kept draws per chain |
| BART(Atr) | `trim=True`, 2 chains, 3,000 burn-in, 6,000 kept draws per chain, `num_gfr=2`, `keep_every=5` |
| Random Forest | 500 trees, `max_features=1/3`, `min_samples_leaf=5` |
| XGBoost | 600 trees, learning rate 0.05, depth 6, row and column subsampling 0.8 |
| LightGBM | 600 trees, learning rate 0.05, 31 leaves, `min_child_samples=20`, subsampling 0.8 |

Each model ran with both 6 and 18 selected features, except BART(Atr), which ran with 18
only.

### 3.6 Evaluation

All models were scored by 5-fold `KFold` (shuffled, seed 42) on the training set, with
identical folds. Scores are in pesos, on all rows of each test fold.

| Metric | Meaning |
|---|---|
| RMSE, MAE | Error in COP |
| MAPE | Mean absolute percentage error, as a fraction |
| R² | Variance explained in pesos |
| Acc_10, Acc_20 | % of listings predicted within ±10% and ±20% of the asking price |
| Over-prediction | % of listings where the prediction exceeds the asking price; 50% is balanced |

MCMC convergence was checked with `arviz` on $\sigma^2$ and on $f(x)$ at each fold's test
rows, using R-hat and effective sample size (ESS). The usual target is R-hat below 1.01.

## 4. Results

### 4.1 Conjugate models

95% intervals below are exact equal-tailed. Exact and simulated intervals agree closely
for all but the sparsest categories (see the [appendix](#appendix-notebook-discrepancies)).

**Share of houses, Beta-Binomial.** The prior comes from the 2018 census (DANE), which
measures housing occupied rather than listed.

| Prior | Observed | Posterior | Posterior mean | 95% interval |
|---|---|---|---|---|
| Beta(12.4, 7.6), mean 0.620 | 109 of 300 (0.363) | Beta(121.4, 198.6) | 0.379 | 0.327 to 0.433 |

Predicted houses among 150 new listings fall from 93.0 under the prior to 56.9 under the
posterior.

**Price segment, Dirichlet-Multinomial.** The prior comes from Guerrero Hurtado (2025),
with $\alpha_0 = 100$.

| Segment | Prior mean | Observed share | Posterior mean | 95% interval |
|---|---:|---:|---:|---|
| Económico (COP 19.5M to 175.5M) | 0.6935 | 0.0133 | 0.1834 | 0.147 to 0.223 |
| Normal (COP 175.5M to 1,007.5M) | 0.2357 | 0.6400 | 0.5389 | 0.490 to 0.588 |
| Lujo (over COP 1,007.5M) | 0.0708 | 0.3467 | 0.2777 | 0.235 to 0.323 |

The prior and data conflict sharply here. The Económico interval excludes the observed
share of 0.013, so it reflects a compromise with a prior built on 2006 to 2013 housing
production, not a market estimate.

**Property type, Dirichlet-Multinomial.** The prior comes from the UAECD and FincaRaíz
study of used housing, with $\alpha_0 = 150$.

| Type | Prior mean | Observed share | Posterior mean | 95% interval |
|---|---:|---:|---:|---|
| Apartamento | 0.6732 | 0.5533 | 0.5933 | 0.548 to 0.638 |
| Casa | 0.2970 | 0.3633 | 0.3412 | 0.298 to 0.386 |
| Apartaestudio | 0.0198 | 0.0800 | 0.0599 | 0.040 to 0.084 |
| Casa campestre | 0.0033 | 0.0033 | 0.0033 | 0.0002 to 0.0104 |
| Habitación | 0.0033 | 0.0000 | 0.0011 | under 0.0001 to 0.0056 |
| Finca | 0.0033 | 0.0000 | 0.0011 | under 0.0001 to 0.0056 |

The same 109 houses give a posterior mean of 0.379 under the census prior and 0.341 under
this one, which shows how much the prior matters at $n = 300$. Apartaestudios stand out,
at about four times their prior share.

**Log price and log area, Normal-Inverse-Gamma.** Priors are calibrated from Toloza-Delgado
et al. (2021). The price prior is adjusted to August 2026 pesos with a CPI factor of 1.5453.

| Variable | Prior | $\bar{y}$, $s^2$ | Posterior | Predictive median, prior to posterior | 95% interval for $e^\mu$ | 95% interval for $e^\sigma$ |
|---|---|---|---|---|---|---|
| Price | NIG(20.0474, 10, 5, 1.5985) | 20.3902, 0.5396 | NIG(20.3791, 310, 155, 82.8306) | COP 508.7M to 708.8M | COP 653.2M to 769.2M | 1.97 to 2.21 |
| Built area | NIG(4.1223, 10, 5, 0.6512) | 4.7841, 0.6852 | NIG(4.7628, 310, 155, 105.2104) | 61.7 m² to 117.1 m² | 106.8 to 128.4 m² | 2.15 to 2.45 |

The team attributes the gaps from these priors to two mechanisms: real market change, and
a sample skewed towards the north of the city and towards larger, pricier homes. These
data cannot separate the two. Normality of the logs is an approximation, since neither
variable is normal even after the transformation.

### 4.2 Feature relevance

Top 10 of 32 features against log price, from the full-training fit, ranked by a
composite of scaled correlation and mutual information:

| Rank | Feature | Correlation | Mutual information |
|---:|---|---:|---:|
| 1 | `area_built_m2` | 0.767 | 0.964 |
| 2 | `bathrooms` | 0.701 | 0.563 |
| 3 | `admin_fee` | 0.636 | 0.642 |
| 4 | `stratum` | 0.654 | 0.347 |
| 5 | `longitude` | 0.434 | 0.507 |
| 6 | `parking_spaces` | 0.447 | 0.482 |
| 7 | `latitude` | 0.304 | 0.541 |
| 8 | `bedrooms` | 0.407 | 0.262 |
| 9 | `dist_nearest_negative_km` | 0.334 | 0.334 |
| 10 | `locality_frec` | 0.394 | 0.240 |

The models use only the mutual information, not this composite. In the full-training OLS
fit, the 18 selected features were those in the table in section 4.3.

### 4.3 OLS

The 18-feature model was refit on the 15,957 retained training rows, with log price as the
response.

| Fit statistic | Value |
|---|---:|
| R² (log scale, in sample) | 0.892 |
| Residual standard error | 0.25 |
| Durbin-Watson | 1.97 |
| Jarque-Bera | 6,692.7, reject normality |
| Breusch-Pagan | 2,079.4, reject homoscedasticity |
| Ramsey RESET | 75.9, reject functional form |

| Feature | Coefficient | 95% HC3 interval | Significant at 5% with HC3 |
|---|---:|---|---|
| `log_area_built_m2` | 0.636 | 0.595 to 0.677 | Yes |
| `stratum` | 0.158 | 0.147 to 0.170 | Yes |
| `age_bracket_code` | -0.034 | -0.038 to -0.031 | Yes |
| `log_admin_fee` | 0.065 | 0.050 to 0.079 | Yes |
| `property_type_casa` | 0.031 | 0.016 to 0.045 | Yes |
| `dist_nearest_negative_km` | 0.012 | 0.007 to 0.016 | Yes |
| `floor` | 0.0067 | 0.0055 to 0.0080 | Yes |
| `n_amenities` | 0.0017 | 0.0013 to 0.0020 | Yes |
| `meses_ofertado` | -0.0010 | -0.0014 to -0.0005 | Yes |
| `locality_frec` | -0.620 | -0.688 to -0.551 | Yes |
| `neighborhood_grouped_frec` | 0.017 | 0.003 to 0.031 | Yes |
| `latitude` | -0.405 | -0.566 to -0.244 | Yes |
| `longitude` | 3.269 | 3.069 to 3.468 | Yes |
| `bathrooms` | 0.038 | -0.005 to 0.081 | No |
| `parking_spaces` | 0.013 | -0.004 to 0.030 | No |
| `bedrooms` | -0.008 | -0.022 to 0.005 | No |
| `property_type_apartaestudio` | 0.005 | -0.013 to 0.022 | No |
| `dist_nearest_positive_km` | 0.001 | -0.006 to 0.009 | No |

Holding the others fixed, a 1% larger built area goes with a price about 0.64% higher.
Each stratum level goes with about 17% more, each age bracket with about 3.4% less, and
each extra kilometre from a disamenity with about 1.2% more. Distance to a positive
landmark shows no detectable effect.

Read these as descriptive associations, not causal effects. The model fails all three
assumption tests, and the p-values are optimistic because the features were selected on
the same data.

### 4.4 Predictive comparison

Means over the 5 folds, with money in millions of COP. Fold standard deviations are 11.2M
to 23.0M for RMSE and 0.006 to 0.020 for MAPE.

| Model | Features | RMSE | MAE | MAPE | R² | Acc_10 | Acc_20 | Over-prediction | Fit (s/fold) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Median baseline | none | 860.8 | 564.4 | 0.7242 | -0.117 | 10.2 | 19.4 | 49.1 | 19 |
| OLS | 6 | 450.1 | 255.2 | 0.2795 | 0.694 | 28.2 | 52.4 | 53.6 | not shown |
| OLS | 18 | 412.4 | 224.5 | 0.2174 | 0.743 | 33.5 | 60.4 | 54.0 | not shown |
| Classic BART | 6 | 399.7 | 212.6 | 0.2158 | 0.759 | 36.2 | 63.4 | 53.0 | 158 |
| Classic BART | 18 | 349.0 | 179.8 | 0.1724 | 0.816 | 42.0 | 70.1 | 51.8 | 149 |
| Random Forest | 6 | 380.2 | 197.7 | 0.2027 | 0.782 | 40.2 | 66.3 | 53.7 | 43 |
| Random Forest | 18 | 356.7 | 175.6 | 0.1700 | 0.808 | 44.8 | 70.8 | 52.9 | 47 |
| XGBoost | 6 | 372.4 | 192.2 | 0.1934 | 0.791 | 41.5 | 68.4 | 52.1 | 39 |
| XGBoost | 18 | 338.8 | **165.9** | **0.1576** | 0.827 | **46.4** | **73.7** | 51.5 | 38 |
| LightGBM | 6 | 374.1 | 193.2 | 0.1934 | 0.789 | 41.9 | 68.3 | 52.0 | 41 |
| LightGBM | 18 | **336.0** | 167.2 | 0.1583 | **0.830** | 45.9 | 73.2 | 51.6 | 37 |
| BART(Atr) | 18 | 345.0 | 176.7 | 0.1684 | 0.821 | 43.1 | 71.3 | 51.4 | 9,437 |

Bold marks the best value in each accuracy column. LightGBM's RMSE lead over XGBoost (2.8M)
is far smaller than the fold-to-fold variation. Fit times are not strictly comparable,
because parallelism differed between runs.

- **Boosting leads.** LightGBM beats BART(Atr) on RMSE, and XGBoost beats it on MAPE, in
  all 5 folds.
- **More features help more than a change of tree model.** Going from 6 to 18 features
  lowers MAPE by 0.033 to 0.043 within each tree family. At 18 features, the whole gap
  across families is 0.015.
- **The errors remain large in absolute terms.** Even the best model is within ±20% for
  about 74% of listings, and within ±10% for only about 46%.
- **Over-prediction is mild.** All feature-based models sit between 51% and 54%.

### 4.5 BART(Atr) against classic BART

| Fold | RMSE: Atr / classic | MAE: Atr / classic | MAPE: Atr / classic |
|---:|---|---|---|
| 1 | **332.7** / 344.0 | **172.0** / 175.6 | **0.1611** / 0.1658 |
| 2 | 333.5 / **332.0** | **170.2** / 173.2 | **0.1599** / 0.1679 |
| 3 | **341.4** / 358.4 | **183.8** / 195.4 | **0.1711** / 0.1845 |
| 4 | 352.4 / **351.5** | **176.8** / 180.0 | **0.1690** / 0.1728 |
| 5 | 365.1 / **359.0** | 180.7 / **175.1** | 0.1807 / **0.1713** |
| Atr wins | 2 of 5 | 4 of 5 | 4 of 5 |

Trimming seems to help typical errors more than squared error, but mean differences are
smaller than fold-to-fold variation, and this is not a controlled comparison. BART(Atr)
also ran with more iterations, thinning and a GFR warm start, so its settings and its
trimming are confounded. The share of data actually trimmed was not recorded.

### 4.6 MCMC convergence

Ranges across the 5 folds:

| Fit | Mean $\sigma^2$ | $\sigma^2$ R-hat | $\sigma^2$ bulk ESS | $f(x)$ median R-hat | $f(x)$ max R-hat | % rows R-hat > 1.01 | $f(x)$ median bulk ESS |
|---|---|---|---|---|---|---|---|
| Classic BART, 6 | 0.051 to 0.053 | 1.45 to 1.78 | 2.95 to 3.88 | 1.40 to 1.45 | 2.49 to 2.64 | 99.86 to 100 | 4.01 to 4.30 |
| Classic BART, 18 | 0.038 to 0.039 | 1.26 to 1.61 | 3.30 to 5.55 | 1.43 to 1.44 | 2.45 to 2.71 | 99.83 to 99.94 | 4.03 to 4.13 |
| BART(Atr), 18 | 0.027 to 0.033 | 1.29 to 1.51 | 3.63 to 5.09 | 1.31 to 1.37 | 2.38 to 2.50 | 99.86 to 99.97 | 4.55 to 5.23 |

**None of the BART fits converged.** Thousands of retained draws carry the information of
only about 3 to 5 independent ones, and the two chains disagree for nearly every test row.
The fold scores in section 4.4 are still honest measures of the predictions those fits
produced. But posterior means may shift with longer runs, and credible intervals from
these fits should not be used. The lower $\sigma^2$ in BART(Atr) is expected after removing
the largest residuals, so it does not by itself show a better fit.

## 5. Limitations and next steps

### Limitations

- **Hold-out not used.** Only the placeholder median model was scored on the 4,428 test
  rows (R² -0.114, MAPE 0.713). All other results are cross-validated, which leaves the
  hold-out available for one clean final test.
- **Uncertainty not evaluated.** `predict_interval` is never called, so interval coverage
  is unmeasured.
- **Trimming not inspected.** The estimated $\hat{p}$ and trimmed rows are stored on the
  fitted model, but the notebook never reads them.
- **No BART feature importance.** The effect sizes above come only from OLS.
- **Selection bias.** The sample skews to the north of the city and to larger, pricier
  homes, and coverage is 91.3%. The rebalancing planned in the first deliverable was not
  carried out, so results describe this listing population, not all of Bogotá.
- **Asking prices.** The models estimate what sellers ask, not what buyers pay.
- **Missing values treated as MAR.** `floor` is 44% missing and `admin_fee` 35% before
  imputation, yet missingness on this site reflects what sellers choose to disclose, so it
  may be informative. The disclosure mechanism is not modelled.
- **Outliers remain in test folds.** The filter acts only while fitting, so scores include
  listings with data errors, which can inflate RMSE for every model.

### Next steps

1. **Fix convergence first:** run more chains for longer, and diagnose why they disagree.
2. **Make the trimming comparison fair:** give classic BART and BART(Atr) identical MCMC
   settings, and compare them fold by fold.
3. **Record what trimming does:** log $\hat{p}$, iterations and trimmed rows per fold.
4. **Measure interval coverage:** check 95% intervals on held-out folds. This is where
   BART could add value over boosting.
5. **Add BART feature importance**, to compare with the OLS effects.
6. **Address selection bias:** reweight or rebalance towards the wider Bogotá market.
7. **Score the hold-out once,** after these choices are final.

## Appendix: notebook discrepancies

Where notebook text disagrees with its own code or outputs, this report uses the computed
values. The notebooks themselves were not changed.

| Location | Text or output says | Code or output shows |
|---|---|---|
| `BART_Trees`, intro | Three spatial distances, including TransMilenio | Two distances; the landmark file has no TransMilenio group |
| `BART_Trees`, feature selection note | Embedded forward selection | `SelectKBest` on mutual information, which is a filter |
| `bayesiana_primera_pt`, Beta-Binomial reading | House share falls to about 30% | Posterior mean 0.379, sample share 0.363 |
| `bayesiana_primera_pt`, NIG price prior text | $\mu_0 = 19.95$, $\sigma^2 \approx 0.83$, adjusted to 2024 pesos | $\mu_0 = 20.0474$, $\hat\sigma^2 = 0.3996$, adjusted to August 2026 pesos |
| `bayesiana_primera_pt`, property-type intervals | Exact HPDI for Habitación and Finca equals the equal-tailed limits (upper 0.0056) | Simulated HPDI upper limit is 0.0043. These densities are monotone, so the optimiser likely stopped at its starting point |

## References

These are the works the notebooks and code cite. Where they give only author and year, this
list does too, with a topic label in place of a title where one helps.

- Breiman (2001). Random Forest.
- Cao, Wu & Wang (2024). Adaptive trimming for BART.
- Chang (2006).
- Chen & Guestrin (2016). XGBoost.
- Chipman et al. (2010). BART.
- DANE. Censo Nacional de Población y Vivienda 2018. <https://www.dane.gov.co/files/censo2018/informacion-tecnica/cnpv-2018-comunicado-3ra-entrega.pdf>
- DANE. Índice de Precios al Consumidor (IPC).
- Guerrero Hurtado, A. (2025). Concentración de capital para la producción de vivienda en el Borde Metropolitano de Bogotá: un enfoque multiescalar del oligopolio inmobiliario. *Bitácora Urbano Territorial*, 35(3). <https://doi.org/10.15446/bitacora.v35n2.122787>
- Iban (2022).
- Ke et al. (2017). LightGBM.
- Loret de Mola et al. (2017).
- Lozano-Gracia & Anselin (2012).
- Tapia et al. (2025).
- Toloza-Delgado, Melo-Martínez & Azcárate-Romero (2021). <https://doi.org/10.17230/ingciencia.17.34.2>
- UAECD & FincaRaíz. Perspectiva del mercado inmobiliario en Colombia. <https://www.catastrobogota.gov.co/noticia/fincaraiz-presento-la-perspectiva-del-mercado-inmobiliario-en-colombia>
- Villada-Medina et al. (2023).
