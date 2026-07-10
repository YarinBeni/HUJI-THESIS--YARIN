# P8 — the supervision-dial spectral probe (λ-probe)

> Implements §4 of the working note *"Supervised Nonlinear Dimension Reduction for
> Manifold-Structured Chronology"* (June 2026): a single generalized eigenproblem with a
> supervision dial λ ∈ [0,1]. λ = 1 is pure manifold geometry (Laplacian eigenmaps — the
> embedding never sees the year); λ = 0 is pure supervised kernel dependence (HSIC on a
> year kernel — no geometry at all). The λ-path answers, in one figure, *how much
> supervision is needed to align the embedding with chronology*.

## 1. The objective and its two ingredients

Data per training fold: rows $X \in \mathbb{R}^{m\times p}$ (pooled activations or TF–IDF,
L2-normalized), years $y \in \mathbb{R}^m$. $H = I_m - \tfrac1m \mathbf{1}\mathbf{1}^\top$
is the centering matrix.

**Ingredient A — predictive dependence (the non-geometric side).**
The Hilbert–Schmidt Independence Criterion between an embedding $Z \in \mathbb{R}^{m\times d}$
and $y$, using a *linear* kernel on $Z$ and a kernel $K_y$ on the year
($[K_y]_{ij} = \exp(-(y_i-y_j)^2/2\sigma_y^2)$, bandwidth $\sigma_y$ = median heuristic), is

$$\widehat{\mathrm{HSIC}}(Z, y) \;=\; \tfrac{1}{(m-1)^2}\,\mathrm{tr}\!\big(Z Z^\top\, H K_y H\big)
\;=\; \tfrac{1}{(m-1)^2}\,\mathrm{tr}\!\big(Z^\top H K_y H\, Z\big).$$

*Backing.* HSIC is the squared Hilbert–Schmidt norm of the cross-covariance operator
$C_{zy}$ between RKHSs (Gretton, Bousquet, Smola & Schölkopf, ALT 2005, Def. 1, Lemma 1);
the biased empirical estimator above is their Lemma 1/Theorem 2 and converges at
$O(m^{-1/2})$ (their Theorem 3). **HSIC = 0 iff $Z \perp\!\!\!\perp y$** when both kernels
are universal on compact domains (their Theorem 4); the Gaussian kernel is characteristic
on $\mathbb{R}$ (Fukumizu, Gretton, Sun & Schölkopf, NeurIPS 2008), so $K_y$ detects *any*
dependence on the year, not just linear correlation. Maximizing
$\mathrm{tr}(Z^\top H K_y H Z)$ therefore extracts the directions of maximal (kernel-)
dependence with chronology — the same objective that defines *supervised PCA* (Barshan,
Ghodsi, Azimifar & Jahromi, Pattern Recognition 44(7), 2011), whose solution with target
kernel $K_y$ is exactly our λ = 0 limit.

**Ingredient B — manifold smoothness (the geometric side).**
Build a k-NN graph on the training rows with heat-kernel weights
$w_{ij} = \exp(-\|x_i-x_j\|^2/\sigma_x^2)$ on edges ($\sigma_x^2$ = median squared edge
distance), symmetrized $W \leftarrow \max(W, W^\top)$; degree $D = \mathrm{diag}(W\mathbf 1)$,
Laplacian $L = D - W$. The identity

$$\mathrm{tr}(Z^\top L Z) \;=\; \tfrac12 \sum_{i,j} w_{ij}\,\|z_i - z_j\|^2$$

says minimizing $\mathrm{tr}(Z^\top L Z)$ keeps graph-neighbors close — the Laplacian
eigenmaps objective (Belkin & Niyogi, Neural Computation 15(6), 2003, §3). The
$Z^\top D Z = I$ constraint is theirs (it fixes scale and weights vertices by degree).

*Backing.* For data sampled from a manifold $\mathcal M$, the graph Laplacian converges
to the Laplace–Beltrami operator $\Delta_{\mathcal M}$ (Belkin & Niyogi, "Towards a
theoretical foundation for Laplacian-based manifold methods," JCSS 74(8), 2008; Coifman &
Lafon, ACHA 2006), whose low eigenfunctions are the smoothest functions *on the manifold*
— this is the precise sense in which the λ = 1 end is "geometry aware": its coordinates
are intrinsic manifold harmonics, blind to $y$ by construction.

## 2. The dial: one generalized eigenproblem

$$\max_{Z^\top D Z = I_d}\;\; (1-\lambda)\,\mathrm{tr}\!\big(Z^\top \underbrace{H K_y H}_{M} Z\big)\;-\;\lambda\,\mathrm{tr}\!\big(Z^\top L\, Z\big),\qquad \lambda\in[0,1].$$

Both terms are quadratic, so with $A_\lambda = (1-\lambda)\,M - \lambda\,L$ the solution is
the top-$d$ generalized eigenvectors of

$$A_\lambda\, z \;=\; \gamma\, D\, z.$$

*Backing.* This is the Ky Fan maximum principle (Ky Fan, PNAS 35, 1949; Courant–Fischer
in the generalized form): $\max_{Z^\top B Z = I}\mathrm{tr}(Z^\top A Z)$ over
$B$-orthonormal $Z$ equals the sum of the top-$d$ eigenvalues of the pencil $(A, B)$ and
is attained at the corresponding eigenvectors. Exact, closed-form, no iteration.

**Limits.** λ = 1: maximize $-\mathrm{tr}(Z^\top L Z)$ ⇒ *bottom* eigenvectors of
$(L, D)$ ⇒ Laplacian eigenmaps, fully unsupervised. λ = 0: supervised PCA with target
kernel $K_y$ (Barshan et al. 2011) under $D$-orthogonality — no geometry term at all.
The path in between is the dial.

**Trivial-solution guard.** $\mathbf 1$ is the $\gamma=0$ eigenvector of $(L,D)$ and is
annihilated by $M$ (because of $H$), so it can surface as a spurious top eigenvector for
large λ. We deflate it: discard any eigenvector with near-zero variance or near-perfect
correlation with $\mathbf 1$ (standard practice in eigenmaps; Belkin–Niyogi drop the
trivial eigenvector explicitly).

**Term scaling (practical, ours).** $M$ and $L$ have incommensurate scales, so raw λ
would not be interpretable. We normalize both to unit spectral norm,
$\hat M = M/\|M\|_2$, $\hat L = L/\|L\|_2$, before mixing. λ then measures the *relative*
weight of geometry vs supervision on a common scale. (Any monotone reparameterization of
the dial changes only the x-axis labels of the λ-plot, not its shape or endpoints.)

## 3. Out-of-sample: the linear-projection (LPP) variant

Eigenmaps assigns coordinates only to training points. For leakage-free evaluation on
held-out rulers we restrict $Z = \tilde X V$ (rows of $\tilde X$ = train-PCA-projected
features; test rows are mapped through the *train-fitted* PCA and $V$). Substituting
turns the problem into the $r\times r$ pencil

$$\big[(1-\lambda)\,\tilde X^\top \hat M \tilde X \;-\; \lambda\,\tilde X^\top \hat L \tilde X\big]\, v \;=\; \gamma\; \tilde X^\top D \tilde X\, v ,$$

which is exactly He & Niyogi's Locality Preserving Projections construction (NeurIPS
2003) applied to $A_\lambda$; the same restriction on the HSIC term is Barshan et al.'s
supervised-PCA projection. The train-only PCA to $r = \min(m-1, 100)$ dims makes
$\tilde X^\top D \tilde X$ well-conditioned (we add $\varepsilon I$, $\varepsilon =
10^{-8}\,\mathrm{tr}/r$) and caps the cost at an $r\times r$ symmetric eigensolve. The
alternative Nyström extension (Bengio et al., NeurIPS 2004) gives the nonparametric
version; we use the projection variant because it is closed-form, strictly train-fitted,
and mirrors how every other probe in the suite maps held-out rows.

## 4. Readouts, protocol, predictions

**Readouts per fold (test side only):**
- `align1` = $|\rho_{\text{Spearman}}(z_1^{\text{test}}, y^{\text{test}})|$ — is the *leading*
  coordinate chronological? (the unsupervised-style question, asked at every λ);
- `pred` = Spearman of ridge-on-$Z_d$ predictions (small $\alpha$, fit on train
  embedding) — the probe-style question, comparable to the PLS/Ridge numbers in P1.

**Leakage-safe evaluation (identical to every probe in the suite):** the 200 balanced
draws (8 rulers × 21 fragments, `draws_matrix.npy`), GroupKFold-by-ruler within each draw
(test rulers never seen in training); *everything* — PCA, graph, $\sigma_x$, $\sigma_y$,
$K_y$, eigenvectors, ridge — is fit on the training rows of the fold. Fold-mean per draw,
mean ± std over draws. Grid: λ ∈ {0, 0.1, …, 1}, d = 3, k ∈ {5, 10, 20} neighbors.

**Interpretation of the λ-curve (from the working note §4–5, adapted to our findings):**

| shape of `align1`(λ) | reading |
|---|---|
| flat and high | timeline is intrinsic to the geometry; supervision adds nothing (the note's (F2) prediction) |
| rising as λ→0 | dependence on year exists but is NOT the dominant geometric axis — supervision must dig it out; "unsupervised timeline" claim fails |
| flat and low (≈ random-init control) | no recoverable chronological structure even with full supervision at the embedding level — consistent with the stress-test nulls (mean-pool ≈ random ≈ TF-IDF) |
| high at λ=0 only for trained models, not random | the one pattern that would rescue a *learned* (if not geometric) year signal |

NB the stress-test results to date (P1/P3/T10: mean-pool ≈ random ≈ 0.35, P3-3b texts
never land on the anchor line) predict the third row on Akkadian inputs, with the
translation arm (engtier0) the only candidate for the fourth. The λ-probe turns that
expectation into a single decisive figure, with random-init and TF-IDF on the same axes.

## 5. References

- A. Gretton, O. Bousquet, A. Smola, B. Schölkopf. *Measuring Statistical Dependence
  with Hilbert-Schmidt Norms.* ALT 2005. (HSIC; empirical estimator; Thm 4: HSIC=0 ⇔
  independence for universal kernels.)
- K. Fukumizu, A. Gretton, X. Sun, B. Schölkopf. *Kernel Measures of Conditional
  Dependence.* NeurIPS 2008. (Characteristic kernels; Gaussian is characteristic.)
- M. Belkin, P. Niyogi. *Laplacian Eigenmaps for Dimensionality Reduction and Data
  Representation.* Neural Computation 15(6), 2003. (Objective, $Lz=\gamma Dz$, trivial
  eigenvector.)
- M. Belkin, P. Niyogi. *Towards a Theoretical Foundation for Laplacian-Based Manifold
  Methods.* JCSS 74(8), 2008. (Graph Laplacian → Laplace–Beltrami.)
- X. He, P. Niyogi. *Locality Preserving Projections.* NeurIPS 2003. (Linear
  out-of-sample variant of eigenmaps.)
- E. Barshan, A. Ghodsi, Z. Azimifar, M. Zolghadri Jahromi. *Supervised Principal
  Component Analysis.* Pattern Recognition 44(7), 2011. (max tr HSIC projection = our λ=0.)
- K. Fan. *On a Theorem of Weyl Concerning Eigenvalues of Linear Transformations I.*
  PNAS 35, 1949. (Trace-max principle.)
- Y. Bengio, J.-F. Paiement, P. Vincent, O. Delalleau, N. Le Roux, M. Ouimet.
  *Out-of-Sample Extensions for LLE, Isomap, MDS, Eigenmaps, and Spectral Clustering.*
  NeurIPS 2004. (Nyström alternative to the projection variant.)
- J. Ham, D. Lee, S. Mika, B. Schölkopf. *A Kernel View of the Dimensionality Reduction
  of Manifolds.* ICML 2004. (Isomap = kernel PCA — the identity behind the note's §2
  G-KPLS sibling.)
- R. Rosipal, L. Trejo. *Kernel Partial Least Squares Regression in RKHS.* JMLR 2, 2001.
  (Kernel PLS — the G-KPLS engine, for the companion experiment.)
- K. Fukumizu, F. Bach, M. Jordan. *Dimensionality Reduction for Supervised Learning
  with Reproducing Kernel Hilbert Spaces.* JMLR 5, 2004. (KDR / central subspace — the
  note's §3, not implemented here.)
