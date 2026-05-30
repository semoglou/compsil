# CompSil

<p align="center">
  <a href="https://pypi.org/project/compsil/"><img src="https://img.shields.io/pypi/v/compsil.svg?color=blue" alt="PyPI version"></a>&nbsp;&nbsp;
  <a href="https://pypi.org/project/compsil/"><img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python 3.8+"></a>&nbsp;&nbsp;
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-yellow.svg" alt="License: MIT"></a>&nbsp;&nbsp;
  <a href="https://pepy.tech/project/compsil"><img src="https://pepy.tech/badge/compsil" alt="Downloads"></a>&nbsp;&nbsp;
  <a href="#"><img src="https://img.shields.io/badge/ECML%20PKDD-2026-green" alt="ECML PKDD 2026"></a>
</p>

<table>
<tr>
<td>

📄 **Accepted at _ECML PKDD 2026_**  

**Composite Silhouette**  

</td>
</tr>
</table>

**CompSil** is an open-source Python package for selecting the number of clusters in unlabeled data using **Composite Silhouette**, an internal validation criterion that adaptively combines micro- and macro-averaged Silhouette scores across repeated subsampled clusterings.

### Composite Silhouette: A Subsampling-based Aggregation Strategy

Selecting the number of clusters is a central challenge in unsupervised learning, where ground-truth labels are usually unavailable.

The standard Silhouette coefficient is one of the most widely used internal validation metrics for this task. However, its usual **micro-averaged** form aggregates Silhouette values over all data points, which can make the score strongly influenced by large clusters. In imbalanced datasets, this may mask poor separation or instability in smaller but meaningful groups.

A natural alternative is **macro-averaging**, where Silhouette values are first averaged within each cluster and then averaged across clusters. This gives every cluster equal influence, reducing the dominance of majority groups. However, macro-averaging can also overemphasize small, noisy, or under-represented clusters.

<img src="https://raw.githubusercontent.com/semoglou/compsil/main/figs/agg.png" alt="Micro vs Macro Silhouette Aggregation" width="800">

These complementary failure modes create a practical dilemma:

- **Micro-averaging** reflects global, point-wise clustering quality but can favor majority clusters.

- **Macro-averaging** reflects cluster-wise balance but can overemphasize small or noisy groups.

In many applications, it is unclear in advance which view should be trusted.

**CompSil** addresses this issue by using the disagreement between micro- and macro-averaged Silhouette scores as a local signal for adaptive aggregation.

Composite Silhouette evaluates candidate numbers of clusters through repeated subsampled clusterings. For each candidate value of `k`, the method:

1. Draws multiple subsamples of the dataset.

2. Clusters each subsample.

3. Computes both micro- and macro-averaged Silhouette scores.

4. Measures their discrepancy.

5. Converts this discrepancy into a smooth convex weight.

6. Combines the two Silhouette views into a subsample-level composite score.

7. Averages the composite scores across subsamples.

<img src="https://raw.githubusercontent.com/semoglou/compsil/main/figs/smm.png" alt="Composite Silhouette pipeline" width="800">

For each subsample, Composite Silhouette combines the two views as:

```text

S_mM = w * S_micro + (1 - w) * S_macro

```

where the weight `w` is determined adaptively from the normalized discrepancy between `S_micro` and `S_macro`.

This produces a single internal validation score that can be maximized over candidate values of `k`.

CompSil enables:

- Selection of the number of clusters without labels.

- Adaptive balancing of micro- and macro-averaged Silhouette.

- More robust cluster-count selection under size imbalance.

- Repeated subsampling for stable internal validation.

- Optional lower-confidence-bound selection using subsampling variability.

# 

## Citation

If you find this work useful, please consider citing:

Semoglou, A., Likas, A., & Pavlopoulos, J. (2026). Composite Silhouette.  

Accepted at *ECML PKDD 2026*.

```bibtex

@inproceedings{semoglou2026composite,

  title     = {Composite Silhouette},

  author    = {Semoglou, Aggelos and Likas, Aristidis and Pavlopoulos, John},

  booktitle = {Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases},

  year      = {2026}

}

```

## Installation

Install **CompSil** from [PyPI](https://pypi.org/project/compsil/):

```bash

pip install compsil

```

Import the main class in Python as:

```python

from compsil import CompSil

```

## API Reference

CompSil provides a simple class-based interface for evaluating Composite Silhouette over one or more candidate numbers of clusters.

---

#### `CompSil`

Computes Composite Silhouette scores for candidate cluster counts using repeated subsampled KMeans clusterings.

```python

CompSil(

    data,

    ground_truth=None,

    k_values=range(2, 11),

    num_samples=10,

    sample_size="auto",

    random_state=42,

    n_jobs=-1,

    eps=1e-12,

)

```

**Inputs**

- `data`: array-like of shape `(n_samples, n_features)`  

  Input data matrix.

- `ground_truth`: int or None, default `None`  

  Optional reference number of clusters.  

  Used only for visualization.

- `k_values`: iterable of int or int, default `range(2, 11)`  

  Candidate number or candidate numbers of clusters to evaluate.

- `num_samples`: int, default `10`  

  Number of subsamples used for each candidate value of `k`.

- `sample_size`: int, float, None, or `"auto"`, default `"auto"`  

  Subsample size used in each repeated clustering.

  - If `int`, it is interpreted as the absolute subsample size.

  - If `float` in `(0, 1]`, it is interpreted as a fraction of the dataset size.

  - If `None` or `"auto"`, the subsample size is selected automatically from the dataset size and the largest candidate value of `k`.

- `random_state`: int, default `42`  

  Base random seed used for reproducible subsampling and clustering.

- `n_jobs`: int, default `-1`  

  Number of parallel jobs used during evaluation.

- `eps`: float, default `1e-12`  

  Numerical stability constant used when normalizing micro–macro discrepancies.

---

#### `evaluate`

Evaluates Composite Silhouette over all candidate values of `k`.

```python

model.evaluate()

```

After calling `evaluate`, the results are stored in:

```python

model.results_df

```

The results table contains:

- `k`: candidate number of clusters.

- `avg S_micro`: average micro-averaged Silhouette across subsamples.

- `avg S_macro`: average macro-averaged Silhouette across subsamples.

- `w_micro`: average adaptive weight assigned to the micro view.

- `S_mM`: Composite Silhouette score.

- `std S_mM`: standard deviation of subsample-level composite scores.

- `se S_mM`: standard error of the Composite Silhouette estimate.

- `LCB S_mM`: lower-confidence-bound score, computed as `S_mM - se S_mM`.

- `B_eff`: number of valid subsampling trials.

- `sample_size`: resolved subsample size.

- `sample_fraction`: resolved subsample fraction.

---

#### `get_optimal_k`

Returns the selected number of clusters.

```python

model.get_optimal_k(use_lcb=False)

```

**Inputs**

- `use_lcb`: bool, default `False`  

  If `False`, selects the `k` that maximizes `S_mM`.  

  If `True`, selects the `k` that maximizes `LCB S_mM`.

**Returns**

- `optimal_k`: int  

  Selected number of clusters.

---

#### `get_results_dataframe`

Returns the results as a pandas DataFrame indexed by `k`.

```python

results = model.get_results_dataframe()

```

**Returns**

- `results`: pandas DataFrame  

  Table containing the Composite Silhouette results for all candidate values of `k`.

---

#### `plot_results`

Plots the Composite Silhouette curve together with the subsample-averaged micro- and macro-averaged Silhouette curves.

```python

model.plot_results()

```

If `ground_truth` was provided, it is shown as a vertical reference line.


## Acknowledgments

This work was partially supported by project MIS 5154714 of the National Recovery and Resilience Plan Greece 2.0, funded by the European Union under the NextGenerationEU Program.

## License

This project is licensed under the [MIT License](https://github.com/semoglou/compsil/blob/main/LICENSE).

## Links

- Repository: [GitHub](https://github.com/semoglou/compsil)

- Package: [PyPI](https://pypi.org/project/compsil/)

- Paper: Accepted at ECML PKDD 2026

- Preprint: Coming soon

- DOI: Coming soon
