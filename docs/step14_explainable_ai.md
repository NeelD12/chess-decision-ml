# Step 14 — Explainable AI (SHAP)

**Status: ⏳ Do not start until Step 12 produces a working model.** SHAP explains an
*already-trained* model's predictions — there's nothing to explain until then.

## Why this step exists at all

The project isn't just "predict blunder probability" — it's specifically about
**explainable** ML. Reporting `Blunder Probability: 82%` on its own isn't the
deliverable; being able to say *why* the model thinks that is:

```
Low remaining time        ↑ Risk
High position complexity  ↑ Risk
Recent CPL increasing     ↑ Risk
Strong rating advantage   ↓ Risk
```

## What to call, and why (when you get here)

- `shap.TreeExplainer(model)` — the natural starting point once Model A/B (Step 12) are
  tree-based (Random Forest or XGBoost). It's fast and *exact* for tree models,
  unlike the more general (and much slower) `shap.KernelExplainer`, which you shouldn't
  need for this project.
- `explainer.shap_values(X_test)` (classification) or `explainer(X_test)`
  (regression, newer SHAP API) — computes the per-prediction, per-feature contribution
  values.
- `shap.summary_plot(...)` — the standard "which features matter most, overall" view.
- `shap.force_plot(...)` or a waterfall plot — the "why did the model say 82% for
  *this specific* move" view, which is the one that matches the example above.

## Do this for both models

Model B (blunder classifier) is the more natural fit for the risk-factor framing above.
Model A (decision quality regressor) is worth explaining too — it answers "why did the
model think this particular move was low quality," which is arguably the more
pedagogically interesting of the two for a project about human decision-making.

## Resources

- SHAP docs: https://shap.readthedocs.io/
- SHAP's own example notebooks (search their docs site for "Tree explainer examples") —
  reading a couple of these before writing your own code will save you time.
