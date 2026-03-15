# BigMart Sales Prediction — Project Summary

**Objective:** Predict item-level sales for 1,559 products across 10 BigMart outlets (2013 data)  
**Metric:** Root Mean Squared Error (RMSE) — lower is better  
**Best score:** 1140.078905956031

---


## Experiment Journey

| # | Notebook Name | Key idea | Findings |
|---|---|---|---|
| 1 | `experiment1_naive_baseline` | RF, single split, global mean imputation | Wrong year (2026 vs 2013), wrong imputation strategy, single split is unreliable |
| 2 | `experiment2_kfold_ensemble` | KFold CV, per-item weight imputation, RF+GBM | Per-item imputation is correct; KFold > single split; GBM adds no real LB gain |
| 3 | `experiment3_lb_simulated_validation` | Hold out 40% items per outlet, leaf=38 | Tuning method changes the tuned value; standard KFold is too optimistic |
| 4 | `experiment4_feature_pruning` | Dropped 7 features → 9 clean ones | Removing noisy features improved LB even when OOF got worse |
| 5 | `experiment5_store_efficiency` | `selling_price_est` = MRP × outlet efficiency ratio | Best single feature; captures within-outlet-type variation |
| 6 | `experiment6_visibility_imputation` | Replace visibility zeros with per-item mean | 6.2% of rows have physically impossible zero visibility |
| 7 | `experiment7_store_weight_profile` | `pct_lightweight_items` per store | Hypothesis was wrong — all 10 outlets stock the same item mix. Tiny gain anyway |

---


## What Worked vs What Didn't

**Kept:**
- `selling_price_est` — MRP × outlet sales-per-MRP ratio (store efficiency)
- `outlet_mean_sales` — 10 outlets × ~900 rows each → stable, safe aggregate
- `vis_ratio` — relative shelf space (not target-derived)
- `min_samples_leaf=75` — more regularisation needed than OOF suggested

**Rejected:**
- Outlet dimension interactions — `outlet_mean_sales` already captures this
- LightGBM / XGBoost — both scored 1147 on LB (RF at 1140 won)
- Log-transform target — +30 RMSE worse for RF
- `Item_Weight` memorises training patterns

---

## Final Model
**Notebook**: experiment8_final.ipynb
**Algorithm:** Random Forest  
The dominant signal (`selling_price_est`) is near-linear. RF's averaging handles this cleanly on 8,523 rows without the variance amplification that boosting causes on a right-skewed target.

**Parameters:** `n_estimators=500`, `min_samples_leaf=75`, `max_features=0.5`, `K=5`, `seeds=5`  
**Features:** 19 (all imputed on combined train+test, none target-derived)
