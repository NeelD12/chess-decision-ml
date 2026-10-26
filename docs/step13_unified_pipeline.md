# Step 13 — One System, Not Two Projects

**Status: ℹ️ Read only — a design principle, not a task with its own code.**

It's easy, once Step 12 has you training two separate models (a regressor and a
classifier), to let them drift into two disconnected mini-projects with duplicated
data-loading, duplicated feature lists, and inconsistent train/test splits. Keep them
architecturally together instead:

```
                 ┌──→ Decision Quality (regression)
Features (X) ────┤
                 └──→ Blunder Probability (classification)
```

**In practice, this means:**
- One shared function/module that builds `X` from the features dataset — both models
  call it, so a feature change updates both automatically instead of two copies drifting
  apart.
- The **same** train/test split (same `game_id`s on each side) for both models, so
  results are directly comparable and there's no chance one model's test set leaked
  into the other's training set.
- One evaluation script/notebook that reports both models' metrics side by side, rather
  than two separate ad-hoc printouts.

**What this is *not* asking for:** a single combined multi-task neural network. The
roadmap is explicit that classical ML with separate estimators is the priority for this
project's scope — a true multi-task model is only a possible *future* extension, not
part of the current plan. Don't reach for a neural net here; reach for consistent
plumbing between two classical models.
