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

Aggelos Semoglou, Aristidis Likas, and John Pavlopoulos

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


