# Design.md

## 0. Purpose, scope, and freeze policy

### 0.1 Purpose

This document is the single project specification for the C-DAY epidemiology topological-learning project. It is not a README, not a literature summary, and not an informal planning note. Its purpose is to define the project completely and rigorously before final freezing of claims, experiments, implementations, and outputs. It serves simultaneously as the mathematical specification, the statistical design specification, the machine learning system specification, the network-science specification, and the public-health interpretation specification.

### 0.2 Scope

This document governs five interlocking domains:

1. Mathematics.
2. Statistics and information theory.
3. Machine learning and deep learning.
4. Network science.
5. Public health.

Every scientific claim, implementation choice, dataset inclusion rule, training protocol, evaluation result, and interpretation that enters the final project must be justified within this document or within a later formal amendment to it.

### 0.3 Freeze rule

Once this document is frozen, all subsequent work must conform to it. No change in mathematical object, estimand, baseline family, simulator family, metric, split logic, or implementation contract is permitted without a versioned revision to `Design.md`. Silent drift is forbidden.

### 0.4 Exactness taxonomy

Each object, claim, or procedure in this document must be labeled conceptually as belonging to one of the following six classes:

- **Exact mathematical definition**: a formally defined object or construction.
- **Theorem-backed construction under explicit assumptions**: valid only under stated assumptions.
- **Statistical estimand or evaluation functional**: a formally defined inferential or predictive target.
- **Computational approximation**: an approximation to an exact object.
- **Engineering convention**: a project-level implementation choice made for tractability or standardization.
- **Deferred semantic choice**: a mathematically admissible construction whose final applied interpretation is not fixed until later sections explicitly freeze it.

The project must never conflate these classes.

In particular:

- no computational approximation may be described as an exact object;
- no engineering convention may be described as a theorem-backed necessity;
- no deferred semantic choice may be treated as though it were already scientifically validated;
- no statistical target may be replaced implicitly by a convenient computational surrogate.

Every section must preserve these distinctions.

### 0.5 Failure philosophy

The project adopts a fail-fast philosophy. Logical inconsistency, unsupported assumptions, chronology violation, leakage, invalid cache reuse, undefined evaluation scope, or mathematically illegal configuration must cause immediate termination. Silent fallback is prohibited. A hard crash upon logical error is a feature, not a bug.

---

# I. Mathematics

## I.1 Purpose and governing mathematical principle

This section gives the complete formal mathematical specification of the project.

Its purpose is to define, without omission:

1. the exact topological state spaces;
2. the algebra of persistence objects;
3. the virtual persistence diagram (VPD) group construction;
4. the Wasserstein and heat-kernel / RKHS analytic pipeline;
5. the uniformly discrete infinite extension;
6. admissible higher-order temporal and structural constructions;
7. all admissible topology-aware loss families;
8. the full differentiation pipeline;
9. epidemic dynamical mathematics;
10. computational and approximation status.

This section must be sufficient for complete mathematical reconstruction of the pipeline.

No later section may introduce a mathematical object not formally defined here.

---

## I.2 Primitive mathematical objects and notation

Let
\[
(X,d)
\]
be a metric space.

Where recursive interval constructions require order structure, define
\[
(X,d,\preceq)
\]
as a preordered metric space.

Let
\[
A\subseteq X,\qquad A\neq\varnothing
\]
be the distinguished diagonal / null subset.

The fundamental primitive is the **metric pair**
\[
(X,d,A).
\]

Define the quotient space
\[
X/A
\]
by collapsing all points of \(A\) into the distinguished basepoint
\[
[A].
\]

Let
\[
q:X\to X/A
\]
denote the quotient map.

Let time be indexed by
\[
T=\{t_0,t_1,\dots,t_N\}.
\]

Define temporal lag
\[
\tau>0.
\]

Define:

- \(k\): homology dimension
- \(n\): structural recursive depth
- \(r\): recursive comparison depth
- \(m\): feature dimension for finite approximations
- \(\omega\): trainable parameter vector

All homology is taken over a fixed field
\[
\mathbb F.
\]

---

## I.3 Persistent homology and ordinary persistence diagrams

### I.3.1 Filtration

Let
\[
\mathcal F=\{K_\alpha\}_{\alpha\in\mathbb R}
\]
be a filtration satisfying
\[
\alpha\le \beta \implies K_\alpha\subseteq K_\beta.
\]

For graph pipelines this may arise from edge-weight threshold filtrations.

For output-topology losses this may arise from superlevel or sublevel filtrations induced by model output
\[
u(\omega).
\]

---

### I.3.2 Persistence module

For fixed homology dimension \(k\), define
\[
V_\alpha := H_k(K_\alpha;\mathbb F).
\]

This yields the persistence module
\[
\{V_\alpha,\varphi_{\alpha\beta}\}_{\alpha\le\beta}.
\]

---

### I.3.3 Barcode

The decomposition into intervals gives the barcode
\[
\mathcal B_k=\{[b_i,d_i)\}_{i\in I}.
\]

---

### I.3.4 Ordinary persistence diagram

Define the persistence diagram
\[
D_k=\{(b_i,d_i)\}_{i\in I}
\]
as a multiset in
\[
\mathbb R^2,\qquad b_i<d_i.
\]

The diagonal
\[
\Delta=\{(x,x):x\in\mathbb R\}
\]
is assigned infinite multiplicity.

---

## I.4 Metric-pair geometry

For
\[
x,y\in X
\]
define the distance to the distinguished subset
\[
d(x,A):=\inf_{a\in A} d(x,a).
\]

For
\[
p\in[1,\infty]
\]
define the strengthened metric
\[
d_p(x,y)
=
\min\left\{
d(x,y),
\left(
d(x,A)^p+d(y,A)^p
\right)^{1/p}
\right\}.
\]

The project’s canonical metric is the \(p=1\) case:
\[
d_1(x,y)=\min\{d(x,y),d(x,A)+d(y,A)\}.
\]

This induces the quotient metric
\[
\overline d_1
\]
on
\[
X/A.
\]

---

## I.5 Diagram monoid

Define the free commutative monoid
\[
D(X)
=
\left\{
\alpha:X\to\mathbb N
\;\middle|\;
\alpha \text{ has finite support}
\right\}.
\]

Addition is **pointwise multiplicity addition**
\[
(\alpha+\beta)(x)=\alpha(x)+\beta(x).
\]

Define
\[
D(A)\subseteq D(X)
\]
as the submonoid supported on \(A\).

The ordinary persistence-diagram monoid is
\[
D(X,A):=D(X)/D(A).
\]

This is the exact algebraic object for finite persistence diagrams.

---

## I.6 Wasserstein geometry on the monoid

For
\[
\alpha,\beta\in D(X,A)
\]
define the \(p\)-Wasserstein metric
\[
W_p(\alpha,\beta)
=
\inf_{\pi\in\Pi(\alpha,\beta)}
\left(
\sum_{(x,y)} d_p(x,y)^p\,\pi(x,y)
\right)^{1/p}.
\]

The project’s exact group lift uses
\[
W_1.
\]

This is essential because translation invariance is guaranteed in the \(W_1\) regime.

---

## I.7 Grothendieck completion and virtual persistence diagrams

Define the Grothendieck completion of a cancellative commutative monoid \(M\) by
\[
K(M)
=
(M\times M)/\sim
\]
where
\[
(a,b)\sim(c,d)
\iff
a+d=b+c.
\]

The equivalence class of
\[
(a,b)
\]
is written
\[
a-b.
\]

Addition is
\[
(a-b)+(c-d)=(a+c)-(b+d).
\]

Inverse:
\[
-(a-b)=b-a.
\]

Neutral element:
\[
0=a-a.
\]

Now specialize:
\[
K(X,A):=K(D(X,A)).
\]

This is the **virtual persistence diagram group**.

This is the project’s foundational topological state space.

---

## I.8 Lifted Wasserstein metric on the VPD group

Define
\[
\rho(\alpha-\beta,\gamma-\delta)
=
W_1(\alpha+\delta,\gamma+\beta).
\]

This defines a translation-invariant metric on
\[
K(X,A).
\]

This metric is exact.

---

## I.9 Finite VPD harmonic analysis

Assume
\[
|X\setminus A|<\infty.
\]

Then
\[
K(X,A)\cong \mathbb Z^d
\]
for
\[
d=|X\setminus A|.
\]

Its Pontryagin dual is
\[
\widehat K \cong \mathbb T^d.
\]

Define characters
\[
\chi_\theta(z)
=
e^{i\langle \theta,z\rangle},
\qquad
\theta\in\mathbb T^d.
\]

---

## I.10 Heat semigroup

Define graph Laplacian
\[
L.
\]

Define heat semigroup
\[
P(t)=e^{-tL}.
\]

The corresponding translation-invariant heat kernel is
\[
k_t(x,y)=P(t)(x-y).
\]

---

## I.11 RKHS pipeline

Define the RKHS
\[
\mathcal H_t
\]
associated with
\[
k_t.
\]

The feature map is
\[
\Phi_t:K(X,A)\to\mathcal H_t,
\qquad
\Phi_t(z)=k_t(\cdot,z).
\]

The induced semimetric is
\[
d_{\mathcal H_t}(x,y)
=
\|\Phi_t(x)-\Phi_t(y)\|_{\mathcal H_t}.
\]

This is the canonical RKHS branch.

---

## I.12 Uniformly discrete extension

The finite-dimensional harmonic-analysis and heat-kernel construction do not automatically extend to the infinite case. The missing issue is local compactness of the VPD group.

Define uniform discreteness of the pointed metric quotient
\[
(X/A,\overline d_1,[A])
\]
by the existence of a constant
\[
\varepsilon>0
\]
such that for all distinct non-basepoint elements
\[
x\neq y,\qquad x,y\in X/A,\qquad x\neq [A],\ y\neq [A],
\]
we have
\[
\overline d_1(x,y)\ge \varepsilon.
\]

The standing assumption for the infinite-dimensional harmonic/RKHS branch is:

\[
(H)\qquad (X/A,\overline d_1,[A])\text{ is uniformly discrete.}
\]

Under \((H)\), the project assumes the following theorem-backed equivalence as part of the formal design contract:

> The metric abelian group \((K(X,A),\rho)\) is locally compact abelian if and only if the pointed quotient metric space \((X/A,\overline d_1,[A])\) is uniformly discrete.

Accordingly, the infinite-dimensional harmonic-analysis branch is admissible **only** under \((H)\). If \((H)\) fails, the infinite-group heat-semigroup/RKHS construction is outside the exact theoretical scope of the project and may not be invoked without amendment.

The section must therefore distinguish three regimes:

1. **finite regime**: exact finite harmonic-analysis and heat-kernel construction;
2. **infinite uniformly discrete regime**: exact theorem-backed extension under \((H)\);
3. **general infinite non-uniformly-discrete regime**: out of scope for the exact harmonic/RKHS branch.

---

## I.13 Finite-volume exhaustion and infinite semigroup

Let
\[
\mathcal F_{\mathrm{fin}}(X\setminus A)
\]
denote the collection of finite subsets of \(X\setminus A\).

For each
\[
F\in \mathcal F_{\mathrm{fin}}(X\setminus A),
\]
define the finite restricted metric pair
\[
X_F:=A\cup F.
\]

Construct the corresponding finite VPD group
\[
K(X_F,A),
\]
equipped with the restricted lifted metric
\[
\rho_F.
\]

For each such finite restriction, define the finite heat semigroup
\[
P_F(t)=e^{-tL_F},
\]
where \(L_F\) is the finite-group Laplacian or finite generator specified by the heat-kernel branch.

The project freezes the following consistency requirement:

> If \(F\subseteq F'\), then the semigroup \(P_{F'}(t)\), when restricted to functions depending only on \(K(X_F,A)\), must agree with \(P_F(t)\).

This consistency-under-restriction property is not optional. It is the mechanism by which the infinite-dimensional semigroup is defined.

Under \((H)\), define the infinite semigroup
\[
P_\infty(t)
\]
as the unique consistent semigroup whose restriction to each finite \(K(X_F,A)\) agrees with \(P_F(t)\).

The associated infinite heat kernel is
\[
k_t^\infty(x,y),
\]
defined by the semigroup action of \(P_\infty(t)\) on the infinite VPD group.

The corresponding infinite-dimensional RKHS and canonical feature map are admissible only in this theorem-backed uniformly discrete regime.

This subsection must be interpreted as:
- exact in the finite case,
- theorem-backed under \((H)\) in the infinite uniformly discrete case,
- unavailable otherwise.

---

## I.14 Official topological state families

The mathematics allows many admissible state families, but the official benchmark freezes a strict active subset.

### I.14.1 Official state family S0: time-indexed VPD states

For each benchmark window \(t\), define the official topological state
\[
D_t \in K(X,A).
\]

This is the foundational topology-bearing state object used throughout the benchmark.

### I.14.2 Official state family S1: first-order internal VPD comparisons

For the official benchmark, the primary comparison object is the first-order internal comparison
\[
C_t := D_t - D_{t-\tau},
\]
defined wherever both windows exist.

This is the canonical signed structural-change object for topology-aware training and comparison.

### I.14.3 Secondary admissible state family S2: higher-order temporal differences

For sensitivity or secondary experiments only, define
\[
\Delta_\tau^k D_t
=
\sum_{j=0}^{k}
(-1)^{k-j}
\binom{k}{j}
D_{t+j\tau}.
\]

These higher-order finite differences are **not** part of the official core benchmark unless an amendment explicitly promotes them.

### I.14.4 Forbidden ambiguity rule

The official benchmark must not use vague phrases such as “arbitrary internal comparisons” or “any admissible state object” in empirical sections. All empirical use must be expressed in terms of:
- \(D_t\),
- \(C_t\),
- and any explicitly amendment-approved higher-order state.

---

## I.15 Recursive higher-depth persistence hierarchy

Define
\[
D^{(0)}(X)=X.
\]

For
\[
n\ge0
\]
define recursively
\[
X^{(n)}=D^{(n)}(X)\times D^{(n)}(X).
\]

Define recursive diagonal
\[
A^{(n)}.
\]

Define recursive strengthened metric
\[
d_1^{(n)}.
\]

Define recursive monoid
\[
D^{(n+1)}(X)=D(X^{(n)},A^{(n)}).
\]

Define recursive VPD group
\[
K^{(n)}(X)=K(D^{(n)}(X)).
\]

---

## I.16 Official topological loss families

The official benchmark freezes exactly two active topology-aware loss families and one secondary family.

### I.16.1 Official loss family L1: Wasserstein structural-alignment loss

For the official signed structural-change object \(C_t\), let
\[
\widehat C_t
\]
denote the model-induced or auxiliary-head-predicted structural-change object used for topology-aware supervision.

The official Wasserstein loss is
\[
\mathcal L_W(t)
=
W_1\!\bigl(\widehat C_t, C_t^{\mathrm{ref}}\bigr).
\]

The official reference object is frozen as
\[
C_t^{\mathrm{ref}} = C_t^{\mathrm{obs}},
\]
the observed structural-change object computed from the corresponding supervision target under chronology-respecting training.

The zero-reference convention is not the official benchmark convention. It may be used only as an explicit ablation.

### I.16.2 Official loss family L2: RKHS structural-alignment loss

For the same structural-change objects, define
\[
\mathcal L_H(t)
=
\|\Phi_t(\widehat C_t)-\Phi_t(C_t^{\mathrm{obs}})\|_{\mathcal H_t}^2.
\]

This is the official harmonic-analytic topology-aware loss family. It is interpreted as a smoothed structural-alignment penalty, not as a replacement for the primary intervention estimand.

### I.16.3 Official aggregated topology loss

If both official topology-aware losses are active, the official form is
\[
\mathcal L_{\mathrm{top}}(t)
=
\lambda_W \mathcal L_W(t)
+
\lambda_H \mathcal L_H(t),
\]
with
\[
\lambda_W,\lambda_H \ge 0.
\]

In the official benchmark, \(\lambda_W\) and \(\lambda_H\) are tuned hyperparameters chosen during validation. They are not learned unconstrained inside the model.

### I.16.4 Secondary higher-depth loss family

If recursive higher-depth topology is activated by formal amendment, the secondary higher-depth loss takes the form
\[
\mathcal L_R(t)
=
\sum_{n=1}^{N_d}
\mu_n \mathcal L^{(n)}(t),
\]
with fixed nonnegative coefficients \(\mu_n\) chosen before official evaluation.

This family is excluded from the official core benchmark unless explicitly promoted.

### I.16.5 Frozen implementation rule

Once the benchmark reaches freeze, every active topology-aware loss family must have:

- a frozen source object \(\widehat C_t\),
- a frozen reference object \(C_t^{\mathrm{obs}}\),
- a frozen coefficient policy,
- a frozen task role,
- and a frozen gradient path through the implementation.

The document must not describe any active topology-aware loss using unresolved phrases such as “semantic choice is deferred,” “reference object to be determined later,” or “implementation-dependent comparison target.”

---

## I.17 Output-topology loss paradigm

Let model output be
\[
u(\omega).
\]

If a frozen output object admits a filtration construction, then \(u(\omega)\) may induce
\[
\mathcal F(u(\omega)),
\]
and hence induced persistence coordinates
\[
(b_i(\omega),d_i(\omega)).
\]

A generic output-topology loss may then be written
\[
\mathcal L_{\mathrm{PH}}
=
\sum_i
g(b_i(\omega),d_i(\omega)).
\]

This includes segmentation-style topological losses and, more generally, any topology-aware penalty computed from a filtration induced by a model output field, score surface, or structured prediction object.

For the official benchmark, this subsection is a mathematically admissible loss paradigm, not the default active topology branch. The official core benchmark uses structural-alignment losses based on the observed-window topological state pipeline defined through \(D_t\) and \(C_t\).

An output-topology loss becomes active only if a later subsection explicitly freezes:

- the output object on which filtration is constructed,
- the filtration rule,
- the reference topology object,
- the statistical role of the loss,
- and the exact implementation pathway by which gradients are propagated.

No empirical section may invoke output-topology language without freezing those objects explicitly.

---

## I.18 Smooth-stratum convention

All differentiation is classical only on smooth strata.

A smooth stratum is a region where the following remain fixed:

1. active simplices / cells
2. support labels
3. transport matching
4. active minimum branch
5. recursive comparison branches

At boundary crossings classical differentiability may fail.

---

## I.19 Generalized gradient convention

At nondifferentiable boundaries use generalized gradients.

Default:
Clarke generalized gradient.

---

## I.20 Differential pipeline

### I.20.1 Parameter to output
\[
\frac{\partial u}{\partial \omega}.
\]

### I.20.2 Output to filtration
\[
\frac{\partial f}{\partial u}.
\]

### I.20.3 Filtration to persistence coordinates
\[
\frac{\partial b_i}{\partial f},
\qquad
\frac{\partial d_i}{\partial f}.
\]

### I.20.4 Chain rule
\[
\frac{\partial b_i}{\partial \omega}
=
\frac{\partial b_i}{\partial f}
\frac{\partial f}{\partial u}
\frac{\partial u}{\partial \omega}.
\]

Same for
\[
d_i.
\]

---

## I.21 Differential pipeline for Wasserstein loss

On smooth stratum:
\[
\frac{\partial \mathcal L_W}{\partial \omega}
=
\sum_i
\frac{\partial \mathcal L_W}{\partial c_i}
\frac{\partial c_i}{\partial \omega}.
\]

Matching changes require generalized gradients.

---

## I.22 Differential pipeline for RKHS loss

Exact branch:
\[
\frac{\partial \mathcal L_H}{\partial \omega}
=
2
\langle
\Phi(Z_1)-\Phi(Z_2),
\frac{\partial \Phi}{\partial \omega}
\rangle.
\]

Approximate finite-dimensional branch:
\[
\phi_m
\]
used instead of exact
\[
\Phi.
\]

---

## I.23 Recursive derivative calculus

For recursive product metric:
\[
\frac{\partial d_{\mathrm{prod}}^{(n)}}{\partial \omega}.
\]

For recursive strengthened metric:
\[
\frac{\partial d_1^{(n)}}{\partial \omega}.
\]

For recursive Wasserstein:
\[
\frac{\partial W_p^{(n)}}{\partial \omega}.
\]

Apply recursive chain rule until base persistence coordinates.

---

## I.24 Epidemic differential equations

This subsection defines the deterministic population-level epidemic baselines used for interpretation, calibration, and comparison. All compartmental variables are population fractions unless otherwise stated.

### I.24.1 SIS

For SIS dynamics with normalized population state
\[
S(t)+I(t)=1,
\]
the governing equations are
\[
\frac{dS}{dt}=-\beta S I+\gamma I,
\qquad
\frac{dI}{dt}=\beta S I-\gamma I.
\]

Equivalently, eliminating \(S=1-I\),
\[
\frac{dI}{dt}=\beta(1-I)I-\gamma I.
\]

This model is substantively appropriate for recurrent or non-immunizing settings.

### I.24.2 SIR

For SIR dynamics with normalized population state
\[
S(t)+I(t)+R(t)=1,
\]
the governing equations are
\[
\frac{dS}{dt}=-\beta S I,
\qquad
\frac{dI}{dt}=\beta S I-\gamma I,
\qquad
\frac{dR}{dt}=\gamma I.
\]

If the project chooses to work with counts rather than fractions, the count-based normalization by \(N\) must be made explicit and used consistently:
\[
\frac{dS}{dt}=-\beta \frac{SI}{N},
\qquad
\frac{dI}{dt}=\beta \frac{SI}{N}-\gamma I,
\qquad
\frac{dR}{dt}=\gamma I.
\]

The document must freeze one convention globally and forbid mixing fraction-based and count-based notation without explicit conversion.

### I.24.2A SAIS governing equations

For normalized population fractions

\[
S(t)+A(t)+I(t)=1,
\]

the governing SAIS system is

\[
\frac{dS}{dt}
=
-\beta S I
-\kappa S I
+\delta I
+\alpha A,
\]

\[
\frac{dA}{dt}
=
\kappa S I
-
\beta_A A I
-
\alpha A,
\]

\[
\frac{dI}{dt}
=
\beta S I
+
\beta_A A I
-
\delta I.
\]

Where:

- \(\beta\): susceptible infection rate
- \(\beta_A\): alert-state infection rate
- \(\kappa\): alerting / awareness activation rate
- \(\alpha\): awareness decay rate
- \(\delta\): recovery rate

This is the primary epidemic system for all official intervention-sensitive tasks.

### I.24.3 Official epidemic-model scope

The official benchmark epidemic-model scope is frozen as a single epidemic family:

Primary and only official benchmark model:
- SAIS (Susceptible--Alert--Infected--Susceptible)

SAIS is the sole official epidemic system for benchmark construction, simulator-defined label generation, intervention-response estimation, threshold analysis, containment analysis, and all primary scientific claims.

SIS and SIR may appear only as background mathematical preliminaries or informal lower-order mechanistic reference points. They are not official benchmark model families, not official ablation classes, not official sensitivity axes, and not official leaderboard comparators.

The reason for this freeze is scientific and design-level exactness: the benchmark is intervention-first, and SAIS is the only active family in the document that explicitly represents behavior-aware alertness dynamics relevant to intervention sufficiency, awareness sufficiency, and containment under public-health action.

Accordingly:

- all primary estimands are defined under the SAIS simulator;
- all official supervised labels are generated under the SAIS simulator;
- all primary evaluations are conditional on the frozen SAIS parameterization and intervention family;
- no later section may write as though SIS, SIR, SEIR, SEIS, or any other epidemic family is an active benchmark class unless this subsection is explicitly amended and the full governing equations and stochastic process specification for that family are inserted into this section.

Any reference to epidemic-family comparison outside SAIS is out of scope for the official benchmark unless introduced by formal amendment.

---

## I.25 Stochastic SAIS epidemic process

The official stochastic epidemic process for the benchmark is a node-level SAIS process on the frozen temporal contact structure, optionally modified by a frozen intervention operator.

### I.25.1 Temporal contact substrate

For each benchmark instance indexed by window \(t\), let
\[
G_t^{\mathrm{temp}}=(V,\mathcal E_t)
\]
denote the frozen temporal contact object used by the simulator over the forecast horizon \(H\).

If an intervention of type \(v\) with intensity \(a\) is applied, let
\[
\mathcal E_t^{(v,a)}
\]
denote the intervention-modified temporal contact object, or equivalently let
\[
A_{ij}^{(v,a)}(u)
\]
denote the intervention-modified contact indicator or weight between nodes \(i\) and \(j\) at simulator time \(u\).

No simulator implementation may silently alternate between event-level and window-level transmission semantics. One convention must be frozen globally for the official benchmark implementation.

### I.25.2 Node state space

For each node \(i\in V\) and simulator time \(u\), define the node state
\[
X_i(u)\in\{S,A,I\},
\]
where:

- \(S\): susceptible,
- \(A\): alert,
- \(I\): infected.

The benchmark uses the SAIS state space only.

### I.25.3 SAIS jump hazards

Conditional on the active temporal contact structure and the current node states, the official SAIS jump process is defined by the following transition hazards.

Susceptible-to-alert:
\[
S_i \to A_i
\quad\text{at rate}\quad
\lambda_i^{SA}(u)
=
\kappa
\sum_j
A_{ij}^{(v,a)}(u)\,
\mathbf 1\{X_j(u)=I\}.
\]

Susceptible-to-infected:
\[
S_i \to I_i
\quad\text{at rate}\quad
\lambda_i^{SI}(u)
=
\beta
\sum_j
A_{ij}^{(v,a)}(u)\,
\mathbf 1\{X_j(u)=I\}.
\]

Alert-to-infected:
\[
A_i \to I_i
\quad\text{at rate}\quad
\lambda_i^{AI}(u)
=
\beta_A
\sum_j
A_{ij}^{(v,a)}(u)\,
\mathbf 1\{X_j(u)=I\}.
\]

Infected-to-susceptible:
\[
I_i \to S_i
\quad\text{at rate}\quad
\delta.
\]

If awareness decay is active in the official simulator, then additionally:
\[
A_i \to S_i
\quad\text{at rate}\quad
\alpha.
\]

### I.25.4 Official decay convention

The deterministic SAIS equations in Section I.24 include awareness decay \(\alpha\). Therefore the official stochastic simulator must use the same decay convention unless a formal amendment removes \(\alpha\) from both the deterministic and stochastic benchmark definitions simultaneously.

Accordingly, the official benchmark simulator includes the transition
\[
A_i \to S_i
\]
at rate \(\alpha\).

### I.25.5 Initial condition and seeding rule

For each benchmark instance \(t\), the simulator must freeze:

- the seed-selection rule,
- the number or fraction of initially infected nodes,
- the initialization rule for alert nodes if nonzero alert initialization is permitted,
- the forecast horizon \(H\),
- and all simulator random seeds used for training-label and evaluation-label generation.

No later section may refer to a large-outbreak probability, intervention sufficiency quantity, or containment quantity without these initialization conventions having been frozen.

### I.25.6 Intervention dependence

The official benchmark is intervention-first. Therefore intervention enters the stochastic process by one or more frozen operators acting on either:

- the temporal contact structure \(A_{ij}^{(v,a)}(u)\),
- the transmission parameters \(\beta\) and \(\beta_A\),
- the alerting parameter \(\kappa\),
- or another explicitly declared simulator component.

No intervention may be described informally. Every intervention family must specify exactly:

- what object it modifies,
- how intensity \(a\) enters,
- whether the modification is global or targeted,
- and whether the modification is deterministic or randomized conditional on \(\mathcal W_t\).

### I.25.7 Benchmark prohibition rule

The benchmark must never write a generic stochastic epidemic process that silently assumes SIS, SIR, or any epidemic family other than SAIS.

Every official stochastic statement must be interpretable under the frozen SAIS jump process defined in this subsection.

---

## I.26 Spectral epidemic comparators

Define adjacency matrix
\[
A_G.
\]

Define spectral radius
\[
\rho(A_G).
\]

Define Laplacian
\[
L_G.
\]

These are explicit non-topological mathematical comparators.

---

## I.27 Exactness and approximation ledger

### Exact
- persistence modules
- ordinary diagrams
- VPD group
- lifted \(W_1\)
- heat semigroup
- exact RKHS
- recursive hierarchy
- smooth-stratum derivatives

### Theorem-backed under assumptions
- uniformly discrete extension
- infinite heat kernel
- infinite RKHS

### Approximate
- finite random features
- finite-dimensional semimetric approximation

### Deferred semantic choice
- choice of comparison states
- choice of loss aggregation
- depth activation
- epidemiological interpretation

---

## II. Statistics and Information Theory

### 2.1 Purpose and governing statistical principle

This section gives the complete statistical and information-theoretic specification of the project. Its purpose is to define, without omission:

1. the observable data objects;
2. the stochastic and fixed components of the data-generating process;
3. the target population and statistical units;
4. the forecast targets and estimands;
5. the Monte Carlo label-generation mechanism and its uncertainty;
6. the calibration problem as a statistical intervention on the target distribution;
7. the precise semantic role of topology-aware constraints;
8. the training, tuning, and evaluation protocol as a finite-sample statistical design;
9. the metric and scoring-rule framework;
10. the uncertainty-quantification and model-comparison framework;
11. the sensitivity and robustness program;
12. the exclusion, invalidity, and failure rules for the benchmark.

This section must be sufficient for a statistician or information theorist to reconstruct the full inferential and predictive design of the project and to determine exactly what is, and is not, being claimed.

The governing principle is the following:

> Every topology-aware term, every predictive metric, every simulation-derived label, and every reported gain must correspond to an explicitly defined statistical object.

No later section may introduce a new estimand, comparison functional, uncertainty protocol, or evaluation rule unless it is formally defined here.

---

### 2.2 Statistical ontology and exactness taxonomy

To prevent conceptual drift, all quantities in the project are classified into the following categories:

1. **Observed object**: directly observed from data.
2. **Derived deterministic object**: deterministically computed from observed data.
3. **Latent stochastic object**: not directly observed; defined by a stochastic model.
4. **Primary estimand**: the target quantity that the project seeks to predict or estimate.
5. **Monte Carlo estimator**: a simulation-based estimator of a latent quantity.
6. **Training target**: the quantity supplied to the learning algorithm.
7. **Regularization or constraint target**: a quantity used to shape the learned predictor but not itself the primary estimand.
8. **Evaluation functional**: a function of predictions and targets used for assessment.
9. **Sensitivity parameter**: a quantity deliberately varied in robustness analyses.
10. **Post-selection reported quantity**: a quantity reported after tuning/model-selection procedures.

This distinction is mandatory throughout the section.

---

### 2.3 Scientific problem in statistical language

The project studies prediction from temporal contact-network data under a frozen SAIS epidemic process and a frozen family of intervention operators. The observed data are temporal contact events, which are aggregated into benchmark windows. Conditional on each observed window, a fixed SAIS simulator with declared calibration and intervention rules induces a family of simulator-defined epidemic outcomes.

The primary statistical task is not generic outbreak-risk prediction. The primary statistical task is estimation of intervention-response functionals under SAIS dynamics from observed temporal contact structure.

The core scientific object is the window-conditional intervention-response surface
\[
Y_t(v,a,\kappa)
=
\Pr\!\left(
Z_{t,H}^{(v,a,\kappa)} \ge z_{\mathrm{out}}
\;\middle|\;
\mathcal W_t
\right),
\]
or, when burden is the relevant quantity,
\[
\mu_t(v,a,\kappa)
=
\mathbb E\!\left(
Z_{t,H}^{(v,a,\kappa)}
\;\middle|\;
\mathcal W_t
\right),
\]
where:

- \(v\) indexes intervention family,
- \(a\) indexes intervention intensity,
- \(\kappa\) is the alerting or awareness parameter under SAIS,
- \(H\) is the forecast horizon,
- and \(Z_{t,H}^{(v,a,\kappa)}\) is the simulator-defined epidemic burden functional.

The primary benchmark estimands are derived from this response surface. They include:

- minimum intervention sufficiency,
- minimum awareness sufficiency,
- containment-region estimation,
- containment-boundary estimation,
- and regime-transition risk.

Simulator-defined outbreak-risk prediction and attack-rate prediction are retained as secondary tasks because they are scientifically useful, easier to estimate, and informative about whether learned representations contain epidemic structure relevant to the primary intervention task.

Topology-aware terms are not primary targets. They are auxiliary statistical devices used to constrain or regularize learning toward epidemic-relevant structural signals not fully captured by simpler non-topological summaries.

Thus, the fundamental statistical question is:

\[
\textit{Given a temporal contact window, can a topology-aware predictor improve estimation of SAIS intervention-response quantities relative to appropriate non-topological alternatives, while remaining well calibrated under chronology-respecting evaluation?}
\]

---

### 2.4 Observed data, sigma-fields, and derived window objects

#### 2.4.1 Event-level data object

Let the raw observed data be a temporal event collection
\[
\mathcal E = \{e_\ell\}_{\ell=1}^L,
\]
where each event
\[
e_\ell = (i_\ell,j_\ell,s_\ell,u_\ell,w_\ell,m_\ell)
\]
contains, where relevant:

- sender or first participant \(i_\ell\),
- receiver or second participant \(j_\ell\),
- start time \(s_\ell\),
- end time \(u_\ell\),
- duration or weight \(w_\ell\),
- optional metadata \(m_\ell\).

The exact event representation must be fixed per dataset.

#### 2.4.2 Observation filtration

Let
\[
\mathcal F_t^{\mathrm{obs}}
\]
denote the sigma-field generated by all observed contact events up to time \(t\), or its practical equivalent when a fully measure-theoretic treatment is not operationally needed.

This sigma-field is the information set relative to which prediction targets and chronology-respecting evaluation must be interpreted.

#### 2.4.3 Window-construction operator

Define a deterministic windowing operator
\[
\mathsf W_t = \mathsf W(\mathcal E; t,\Delta_w,\Pi_w),
\]
where:

- \(t\) indexes the window,
- \(\Delta_w\) is the window span,
- \(\Pi_w\) is the windowing policy (daily, disjoint, rolling, overlapping, etc.).

The project must specify this operator exactly, including whether windows overlap and whether windows are treated as independent observational units or as serially dependent forecast instances.

#### 2.4.4 Window-level derived network object

For each time index \(t\), let
\[
\mathcal W_t
\]
denote the observed event window produced by the frozen windowing operator.

The official benchmark freezes the following object hierarchy:

1. **Primitive observed object**
   \[
   \mathcal W_t^{\mathrm{evt}}
   \]
   consisting of the event list within the window.

2. **Official weighted window graph**
   \[
   G_t=(V,E_t,w_t),
   \]
   obtained by deterministic aggregation of \(\mathcal W_t^{\mathrm{evt}}\) under the frozen edge-weighting rule.

3. **Official filtration-ready topological object**
   the clique-complex filtration induced from the weighted graph \(G_t\) by the frozen filtration rule.

4. **Derived non-topological summaries**
   any graph statistics, spectral summaries, heuristic scores, or other tabular covariates computed from \(G_t\) or \(\mathcal W_t^{\mathrm{evt}}\).

Accordingly, the benchmark does not permit ambiguity about the phrase “window object.” The official statistical primitive is the event window \(\mathcal W_t^{\mathrm{evt}}\); the official graph object is \(G_t\); and the official topological object is the filtration-ready complex derived from \(G_t\).

No later section may refer vaguely to “the topology of the window” or “the network window” without identifying which object in this hierarchy is being used.

---

### 2.5 Statistical units and target population

#### 2.5.1 Target population

The target population is the population represented by the observed temporal contact system, together with the family of simulator-defined epidemic outcomes that arise when the declared epidemic process is run conditionally on the observed contact windows.

Claims beyond this population are not licensed unless external validation is conducted.

#### 2.5.2 Observational unit

The observational unit is the raw contact event.

#### 2.5.3 Structural unit

The structural unit is the frozen event window
\[
\mathcal W_t^{\mathrm{evt}},
\]
together with its deterministic graph aggregation
\[
G_t=(V,E_t,w_t)
\]
when graph-level or topology-level summaries are required.

No later section may use \(\mathcal W_t\) ambiguously to refer interchangeably to the raw event window, the aggregated graph, or the filtration-ready topological object.

#### 2.5.4 Prediction unit

The prediction unit is the benchmark instance indexed by window \(t\), consisting of the admissible encoder input together with the frozen intervention query object.

Depending on model family, the admissible encoder input is either:

- the event-stream history \(\mathcal E_{t,h}\), or
- the window-summary object \(\mathcal X_t\).

For the official primary task, this prediction unit is paired with an intervention family \(v\) and the corresponding frozen intervention grid \(\mathcal G_v\), because the model predicts an intervention-response object rather than a single scalar by default.

#### 2.5.5 Evaluation unit

The evaluation unit is the held-out chronologically valid benchmark instance, consisting of:

- the admissible input at time \(t\),
- the corresponding intervention query object,
- and the independently generated evaluation Monte Carlo targets for the same instance.

If blocked evaluation is used, the evaluation unit is a chronologically valid block of such benchmark instances.

#### 2.5.6 Effective sample size

If windows overlap or exhibit serial dependence, the effective sample size is strictly smaller than the nominal number of windows. Any uncertainty procedure must respect this dependence structure rather than pretending all windows are independent.

---

### 2.6 Statistical mode of analysis

Every experiment in the project must be labeled as one of the following:

1. **Primary intervention prediction**: forecasting or estimating a frozen SAIS intervention-response quantity conditional on observed windows.
2. **Support-task prediction**: forecasting a support structural or epidemic quantity used for representation evaluation, auxiliary learning, or ablation.
3. **Description**: summarizing structural, predictive, or calibration properties without forecasting claims beyond the declared benchmark target.
4. **Mechanistic comparison**: comparing alternate admissible SAIS parameterizations, intervention rules, or simulator conventions within the official benchmark scope.
5. **Causal analysis**: estimating intervention or behavior effects under a separate identification design.

Unless explicitly stated otherwise, the project operates in intervention-first benchmark mode.

Primary mode:
- SAIS intervention-response estimation

Secondary mode:
- support-task epidemic prediction

Tertiary mode:
- support-task structural temporal graph learning

Accordingly:

- the project does not, by default, estimate causal effects;
- the project does not, by default, identify true real-world outbreak probabilities;
- the project does not, by default, identify the true behavioral mechanism of the population;
- and the project does not, by default, support policy-effect claims outside the declared simulator-conditional benchmark scope.

---

### 2.7 Data-generating process

The full data-generating process consists of several conceptually distinct components.

#### 2.7.1 Contact-process component

The observed temporal contact process is treated as a realized contact history, possibly subject to measurement limitations specified elsewhere. Unless a later subsection introduces a stochastic model for the contact process itself, the contact windows are treated as conditionally fixed inputs to the epidemic simulator and prediction pipeline.

#### 2.7.2 Structural summarization component

A deterministic non-topological summarization map
\[
\mathsf S: \mathcal W_t^{\mathrm{evt}} \mapsto \mathsf S(\mathcal W_t^{\mathrm{evt}})
\]
produces graph statistics, spectral summaries, heuristic covariates, and any other non-topological structural summaries used by baselines or diagnostics.

A distinct deterministic topological map
\[
\mathsf T: \mathcal W_t^{\mathrm{evt}} \mapsto (G_t,\mathcal F_t,P_t,D_t,C_t)
\]
produces the official topology-bearing objects through the frozen pipeline:
\[
\mathcal W_t^{\mathrm{evt}}
\;\to\;
G_t
\;\to\;
\mathcal F_t
\;\to\;
P_t
\;\to\;
D_t
\;\to\;
C_t.
\]

Here:

- \(G_t\) is the weighted window graph,
- \(\mathcal F_t\) is the frozen filtration on the clique-complex lift of \(G_t\),
- \(P_t\) is the resulting persistence object,
- \(D_t\) is the official time-indexed VPD state,
- and \(C_t=D_t-D_{t-\tau}\) is the official first-order structural-change object.

The existence of \(\mathsf T\) does not itself justify statistical use. Its statistical role is frozen later through topology-aware loss semantics, support-task semantics, and comparison against non-topological competitors.

No later section may replace this map by an undocumented alternative pipeline.

#### 2.7.3 Epidemic-simulator component

For each benchmark instance indexed by window \(t\), define the official SAIS simulator
\[
\eta_{\theta}^{\mathrm{SAIS}}\!\left(\mathcal W_t^{\mathrm{evt}}, v, a, \kappa, U_t\right),
\]
where:

- \(\theta\) is the frozen SAIS parameter vector,
- \(v\) is the intervention family,
- \(a\) is the intervention intensity,
- \(\kappa\) is the alerting or awareness parameter when treated as part of the intervention-response query,
- \(U_t\) is simulator randomness,
- and the output is an epidemic path or a derived burden quantity over the frozen horizon \(H\).

The official simulator output must be sufficient to construct:

- the probability response surface \(Y_t(v,a,\kappa)\),
- the burden response surface \(\mu_t(v,a,\kappa)\) if activated,
- and all derived intervention estimands retained in the benchmark.

The simulator family is SAIS only. No later section may state or imply a generic epidemic simulator family once the benchmark reaches freeze.

#### 2.7.4 Behavior-related structural signals of interest

The project treats the following families of structural signals as potentially epidemiologically meaningful:

1. **Connectivity heterogeneity induced by local interactions**.
2. **Community closure versus inter-community leakage**.
3. **Bridge creation, bridge destruction, and component accessibility**.
4. **Loop creation, loop snapping, filling creation, and filling removal**.
5. **Repeated-contact persistence or structural retention across windows**.
6. **Behavior-driven rewiring or effective transmission restructuring**.
7. **Higher-order clustering of structural events over time**, where mathematically admissible.

These are not automatically estimands. They are latent structural signals whose statistical relevance must be tested through prediction and ablation.

---

### 2.8 Forecast targets and estimands

The benchmark uses an intervention-first estimand hierarchy.

#### Support estimands — structural learning and epidemic prediction

These support tasks are retained because they test whether the encoder learns temporally meaningful structure and because they provide lower-level diagnostics that can build toward the main intervention task.

Support structural estimands include:

- future contact probability,
- transmission-path emergence probability,
- bridge-formation probability.

Support epidemic estimands include:

- large-outbreak probability at a frozen baseline intervention setting,
- expected attack rate at a frozen baseline intervention setting,
- horizon-specific outbreak risk at a frozen baseline intervention setting.

These are support estimands, not the primary scientific endpoint.

#### Primary estimands — SAIS intervention-response estimands

Fix a window \(\mathcal W_t\), intervention family \(v\), intervention intensity \(a\), awareness parameter \(\kappa\), forecast horizon \(H\), and outbreak-defining burden threshold \(z_{\mathrm{out}}\).

Let
\[
Z_{t,H}^{(v,a,\kappa)}
\]
denote the SAIS simulator-defined epidemic burden over horizon \(H\) under the frozen intervention and initialization rules.

Define the primary latent response surface
\[
Y_t(v,a,\kappa)
=
\Pr\!\left(
Z_{t,H}^{(v,a,\kappa)} \ge z_{\mathrm{out}}
\;\middle|\;
\mathcal W_t
\right).
\]

If a continuous burden estimand is also retained, define
\[
\mu_t(v,a,\kappa)
=
\mathbb E\!\left(
Z_{t,H}^{(v,a,\kappa)}
\;\middle|\;
\mathcal W_t
\right).
\]

##### 2.8.1 Minimum intervention sufficiency

For fixed intervention family \(v\), fixed awareness parameter \(\kappa\), and fixed acceptance threshold \(q^\star\), define
\[
a_t^\star(v,\kappa)
=
\inf\left\{
a\in\mathcal A_v:
Y_t(v,a,\kappa)\le q^\star
\right\}.
\]

If burden rather than outbreak probability is used for sufficiency, define instead
\[
a_t^\star(v,\kappa)
=
\inf\left\{
a\in\mathcal A_v:
\mu_t(v,a,\kappa)\le b^\star
\right\},
\]
for a frozen burden tolerance \(b^\star\).

This is the primary intervention estimand.

##### 2.8.2 Minimum awareness sufficiency

For fixed intervention family \(v\), fixed intervention intensity \(a\), and fixed acceptance threshold \(q^\star\), define
\[
\kappa_t^\star(v,a)
=
\inf\left\{
\kappa\in\mathcal K:
Y_t(v,a,\kappa)\le q^\star
\right\}.
\]

This is the primary awareness-sufficiency estimand.

##### 2.8.3 Containment region

Define the containment region
\[
\mathcal C_t(v)
=
\left\{
(a,\kappa)\in\mathcal A_v\times\mathcal K:
Y_t(v,a,\kappa)\le q^\star
\right\}.
\]

The benchmark may report either the full set \(\mathcal C_t(v)\) on a frozen intervention grid or a frozen boundary summary derived from it.

##### 2.8.4 Operational threshold functional

If an epidemic-threshold-type quantity is reported, it must be defined operationally as a finite-window, simulator-defined threshold functional under the official SAIS process. It must not be described as an asymptotic epidemic threshold unless the exact asymptotic object is separately defined and justified.

Any such threshold quantity must specify:

- which parameter is varied,
- which intervention variables are held fixed,
- which horizon \(H\) is used,
- and which criterion defines threshold crossing.

##### 2.8.5 Regime-transition risk

For a frozen intervention configuration \((v,a,\kappa)\) and future horizon \(h\), define
\[
\pi_t^{\mathrm{trans}}(v,a,\kappa)
=
\Pr\!\left(
\exists s\in\{t+1,\dots,t+h\}:
Y_s(v,a,\kappa) > q^\star
\;\middle|\;
\mathcal F_t^{\mathrm{obs}}
\right).
\]

This is the official regime-transition estimand.

#### Frozen hierarchy rule

The benchmark is intervention-first. Support structural and epidemic estimands may be used for benchmarking, auxiliary training, ablation, or representation diagnosis, but they do not define the primary leaderboard or primary scientific claim.

---

### 2.9 Monte Carlo label-generation model

The project uses simulation-derived labels rather than directly observed epidemic outcomes. Therefore the supervised targets are Monte Carlo estimators of latent SAIS intervention-response quantities.

#### 2.9.1 Monte Carlo response-surface estimator

Fix a window \(\mathcal W_t\), intervention family \(v\), intervention intensity \(a\), awareness parameter \(\kappa\), and forecast horizon \(H\). Run
\[
M_{t,v,a,\kappa}
\]
independent SAIS simulations and let
\[
Z_{t,H}^{(1;v,a,\kappa)},\dots,Z_{t,H}^{(M_{t,v,a,\kappa};v,a,\kappa)}
\]
denote the simulated epidemic burden values.

Define the Monte Carlo estimator of the outbreak-probability response surface:
\[
\widehat Y_t(v,a,\kappa)
=
\frac{1}{M_{t,v,a,\kappa}}
\sum_{m=1}^{M_{t,v,a,\kappa}}
\mathbf 1\!\left\{
Z_{t,H}^{(m;v,a,\kappa)} \ge z_{\mathrm{out}}
\right\}.
\]

#### 2.9.2 Conditional expectation

Conditional on \(\mathcal W_t\),
\[
\mathbb E\!\left[\widehat Y_t(v,a,\kappa)\mid \mathcal W_t\right]
=
Y_t(v,a,\kappa),
\]
under correct simulator implementation.

#### 2.9.3 Conditional variance

\[
\operatorname{Var}\!\left(
\widehat Y_t(v,a,\kappa)\mid \mathcal W_t
\right)
=
\frac{
Y_t(v,a,\kappa)\left(1-Y_t(v,a,\kappa)\right)
}{
M_{t,v,a,\kappa}
}.
\]

Thus label noise is heteroskedastic across windows and across intervention-grid cells.

#### 2.9.4 Continuous Monte Carlo burden target

If the continuous burden estimand is retained, define
\[
\widehat \mu_t(v,a,\kappa)
=
\frac{1}{M_{t,v,a,\kappa}}
\sum_{m=1}^{M_{t,v,a,\kappa}}
Z_{t,H}^{(m;v,a,\kappa)},
\]
with conditional variance
\[
\operatorname{Var}\!\left(
\widehat \mu_t(v,a,\kappa)\mid \mathcal W_t
\right)
=
\frac{
\operatorname{Var}\!\left(
Z_{t,H}^{(v,a,\kappa)}\mid \mathcal W_t
\right)
}{
M_{t,v,a,\kappa}
}.
\]

#### 2.9.5 Derived Monte Carlo estimators for primary intervention labels

From the estimated response surface, define the derived label estimators:

\[
\widehat a_t^\star(v,\kappa)
=
\inf\left\{
a\in\mathcal A_v:
\widehat Y_t(v,a,\kappa)\le q^\star
\right\},
\]

\[
\widehat \kappa_t^\star(v,a)
=
\inf\left\{
\kappa\in\mathcal K:
\widehat Y_t(v,a,\kappa)\le q^\star
\right\},
\]

and
\[
\widehat{\mathcal C}_t(v)
=
\left\{
(a,\kappa)\in\mathcal A_v\times\mathcal K:
\widehat Y_t(v,a,\kappa)\le q^\star
\right\}.
\]

These are the official supervised labels for the primary intervention tasks, up to the frozen intervention-grid resolution and interpolation rule.

#### 2.9.6 Official statistical status of labels

The official benchmark treats the Monte Carlo response surface
\[
\widehat Y_t(v,a,\kappa)
\]
as a noisy simulation-based probability target and treats the derived quantities
\[
\widehat a_t^\star(v,\kappa),\qquad
\widehat \kappa_t^\star(v,a),\qquad
\widehat{\mathcal C}_t(v)
\]
as noisy Monte Carlo functionals of that surface.

Training may operate either:

- directly on the response surface,
- or on derived intervention labels computed from that surface,

provided the official implementation states which object is supervised directly and which quantities are derived deterministically from model predictions.

Variance-aware weighting is not part of the official core benchmark unless explicitly activated by amendment.

#### 2.9.7 Label intervals

When reported, uncertainty bands for Monte Carlo labels must be defined explicitly. This includes:

- binomial-style intervals for \(\widehat Y_t(v,a,\kappa)\),
- Monte Carlo or bootstrap intervals for \(\widehat \mu_t(v,a,\kappa)\),
- and propagated uncertainty procedures for \(\widehat a_t^\star\), \(\widehat \kappa_t^\star\), and \(\widehat{\mathcal C}_t\).

No primary intervention quantity may be reported without stating how Monte Carlo uncertainty in the response surface propagates into the derived estimand.

---

### 2.10 Calibration as a statistical design intervention

Calibration is a frozen statistical procedure that selects the epidemic-simulator parameter vector used for benchmark label generation.

#### 2.10.1 Calibration parameter vector

Let
\[
\theta=(\beta,\beta_A,\kappa,\alpha,\delta,\theta_{\mathrm{aux}})
\]
denote the official SAIS epidemic parameter vector, where:

- \(\beta\): susceptible-state infection rate,
- \(\beta_A\): alert-state infection rate,
- \(\kappa\): alerting or awareness activation rate,
- \(\alpha\): awareness decay rate,
- \(\delta\): infected-to-susceptible recovery rate,

and where \(\theta_{\mathrm{aux}}\) collects any additional frozen simulator hyperparameters such as:

- seed-selection rules,
- initial infection fraction or count,
- initial alert-state rule if used,
- horizon-specific modifiers,
- and any intervention-family parameters that are part of the official simulator rather than part of the prediction target.

Because the official benchmark epidemic family is SAIS only, the calibration vector must be written in SAIS form throughout the document. No later section may silently revert to an SIS- or SIR-style reduced parameterization.

#### 2.10.2 Calibration sample

Calibration is performed on a frozen calibration subset
\[
\mathcal T_{\mathrm{cal}}
\subseteq
\mathcal T_{\mathrm{train}},
\]
using a fixed number
\[
M_{\mathrm{cal}}
\]
of epidemic simulations per window.

No test windows may enter calibration.

#### 2.10.3 Target-distribution constraints

Because the official benchmark is intervention-first, calibration admissibility must be defined relative to a frozen calibration slice or a frozen low-dimensional summary of the intervention-response surface.

Let
\[
\widehat Y_t^{\mathrm{cal}}(\theta)
\]
denote the response-surface quantity used for calibration, where the document must freeze whether this is:

- a single baseline slice \(\widehat Y_t(v_0,a_0,\kappa_0;\theta)\),
- an averaged surface summary over a frozen calibration grid,
- or another explicitly defined scalar summary of the SAIS response surface.

Then let the benchmark admissibility set be
\[
\mathcal A
=
\left\{
\theta:
\begin{array}{l}
\mathrm{median}(\widehat Y_t^{\mathrm{cal}}(\theta):t\in\mathcal T_{\mathrm{cal}})\in [m_{\min},m_{\max}],\\[4pt]
\mathrm{Var}(\widehat Y_t^{\mathrm{cal}}(\theta):t\in\mathcal T_{\mathrm{cal}})\ge v_{\min},\\[4pt]
\Pr(\widehat Y_t^{\mathrm{cal}}(\theta)\le q_{\mathrm{low}})\ge \ell_{\min},\\[4pt]
\Pr(\widehat Y_t^{\mathrm{cal}}(\theta)\ge q_{\mathrm{high}})\ge u_{\min}
\end{array}
\right\},
\]
with all thresholds frozen explicitly.

The benchmark must not silently calibrate on one scalar target while claiming that a different intervention-response object is primary.

#### 2.10.4 Calibration search procedure

Calibration uses a frozen candidate set
\[
\Theta_{\mathrm{cand}}
\]
and a frozen search rule. The default rule is:

1. evaluate every \(\theta\in\Theta_{\mathrm{cand}}\) on \(\mathcal T_{\mathrm{cal}}\);
2. retain only admissible candidates in \(\mathcal A\);
3. rank admissible candidates by the calibration objective
   \[
   \mathcal C(\theta),
   \]
   defined as the weighted deviation from target-distribution desiderata;
4. select
   \[
   \widehat\theta_{\mathrm{cal}}
   =
   \arg\min_{\theta\in\mathcal A}\mathcal C(\theta).
   \]

If no candidate is admissible, the run fails and the calibration grid must be amended by versioned design revision.

#### 2.10.5 Calibration outputs and persistence requirements

The calibration stage must persist:

- the full candidate set \(\Theta_{\mathrm{cand}}\),
- the admissibility table for all candidates,
- the selected calibrated parameter vector \(\widehat\theta_{\mathrm{cal}}\),
- the calibration simulation seed set,
- the exact diagnostics used for ranking and selection.

#### 2.10.6 Statistical consequence of calibration

Calibration changes:

- the marginal distribution of labels,
- the prevalence of easy versus difficult windows,
- the variance structure of Monte Carlo targets,
- and potentially relative model rankings.

Therefore all benchmark claims are conditional on the official calibrated simulator regime.

#### 2.10.7 Freeze rule

After selection, \(\widehat\theta_{\mathrm{cal}}\) is frozen for the official benchmark. No later analysis may silently replace it.

Any alternate admissible calibration must be labeled as a sensitivity analysis with its own identifier and may not overwrite the official benchmark calibration.

#### 2.10.8 Calibration sensitivity

The official robustness program includes alternate admissible calibrations only if they are explicitly listed in Section VII’s sensitivity matrix. No implicit calibration variation is allowed.

---

### 2.11 Prediction, description, and causation boundary

#### 2.11.1 Prediction target

The project predicts simulator-defined SAIS intervention-response quantities conditional on observed temporal contact windows.

The primary prediction target is the intervention-response object or a frozen functional derived from it, such as:

- minimum intervention sufficiency,
- minimum awareness sufficiency,
- containment-region membership,
- containment-boundary summaries,
- or regime-transition risk.

Scalar epidemic-risk prediction is secondary and is interpreted as a support-task or response-surface marginal, not as the sole official benchmark target.

#### 2.11.2 Descriptive outputs

The project may also describe:

- structural properties of contact windows,
- topological state changes,
- regime composition of the benchmark,
- and behavior of baseline models across structural conditions.

Such description is scientifically useful but is not itself the primary inferential objective.

#### 2.11.3 Non-causal status

Unless explicitly extended by a separate identification design, the project does **not** identify:

- causal effects of topological structure,
- causal effects of awareness processes,
- causal effects of interventions in the real population,
- causal effects of vaccines, masking, or policy,
- or transportable policy effects.

The benchmark is predictive and simulator-conditional, not causally identified.

#### 2.11.4 Interpretation limit

A predictive gain from a topology-aware model means only that the added inductive bias or constraint improves prediction of the declared SAIS intervention-response target under the declared benchmark design.

It does not, by itself, establish that the topological object is a true real-world causal mechanism, nor that the simulator is a perfect representation of the population process.

---

### 2.12 Statistical semantics of topology-aware constraints

This subsection answers the deferred question from the mathematics section: what do the topology-aware constructions mean statistically, and how do they enter an intervention-first SAIS benchmark?

#### 2.12.1 General admissibility principle

A topology-aware loss term is statistically admissible only if it can be interpreted as regularizing toward one or more epidemic-relevant structural signals that are plausibly useful for estimating SAIS intervention-response quantities and that are not already exhausted by simpler non-topological competitors.

#### 2.12.2 Statistical meaning of the official state pipeline

The official topological state pipeline
\[
\mathcal W_t^{\mathrm{evt}}
\;\to\;
G_t
\;\to\;
\mathcal F_t
\;\to\;
P_t
\;\to\;
D_t
\;\to\;
C_t
\]
is interpreted statistically as a map from observed contact structure to a compact structural-change representation.

In this interpretation:

- \(D_t\) is the topological summary of the current window under the frozen filtration;
- \(C_t=D_t-D_{t-\tau}\) is the signed structural-change object;
- and topology-aware learning acts by constraining the model to remain sensitive to structural reorganization patterns plausibly relevant to epidemic spread and intervention effectiveness.

#### 2.12.3 Wasserstein structural-alignment semantics

The Wasserstein topology-aware loss is interpreted as a direct geometric alignment penalty on structural change.

Its statistical semantics include sensitivity to:

- fragmentation versus reconnection,
- bridge creation and destruction,
- loop creation and loop snapping,
- filling creation and filling removal,
- and displacement of structural mass across windows.

This is the benchmark’s direct geometric topology-aware constraint.

#### 2.12.4 Internal VPD comparison semantics

The internal VPD comparison object \(C_t\) is not interpreted as a static topological descriptor. It is interpreted as a signed structural-change state.

Its intended statistical role is to encode reorganization patterns that may affect epidemic accessibility, leakage across community boundaries, redundancy of transmission pathways, or intervention leverage.

#### 2.12.5 RKHS semimetric semantics

The RKHS semimetric is interpreted as a smoothed structural-similarity geometry on signed structural-change states.

Optimization-level meaning:
it is a loss term.

Statistical meaning:
it regularizes the predictor toward preserving a positive-definite similarity geometry among structural-change objects.

Applied meaning:
it pressures the learned system to treat structurally similar reorganization patterns as epidemiologically related, while still allowing nonlinearly smoothed comparison rather than raw pointwise matching.

This interpretation is a modeling hypothesis, not a theorem about real-world mechanism.

#### 2.12.6 Higher-order recursive semantics

When activated, higher-order interval-of-interval constructions are interpreted as encoding compatible aggregations of structural events across time.

Their admissible network-science semantics include:

- persistence of persistence,
- repeated or clustered structural change,
- higher-order containment or comparability structure,
- and event-cluster geometry.

These objects are scientifically meaningful only if the compatibility relations among constituent events are formally respected and if the network-science section provides a coherent interpretation.

#### 2.12.7 Negative-control semantics

The benchmark does **not** presume that topology is automatically superior for signals already well captured by simpler alternatives.

Negative-control examples include:

- mean degree level,
- raw contact totals,
- trivial density variation,
- or simple prevalence-history surrogates when such histories are explicitly supplied.

This restriction prevents semantic overclaiming and forces topology-aware gains to compete against serious simpler baselines.

---

### 2.13 Statistical interpretation of losses, penalties, and constraints

#### 2.13.1 Primary intervention loss

The primary predictive loss is the part of the objective directly tied to the intervention-response target.

Depending on the frozen implementation, this may be:

- a response-surface loss on \(\widehat Y_t(v,a,\kappa)\),
- a response-surface loss on \(\widehat \mu_t(v,a,\kappa)\),
- a derived-label loss on \(\widehat a_t^\star\), \(\widehat \kappa_t^\star\), or \(\widehat{\mathcal C}_t\),
- or a jointly defined loss combining these objects.

The document must freeze which object is supervised directly.

#### 2.13.2 Statistical meaning of topology-aware loss

The topology-aware term is not the primary estimand. It is an auxiliary loss term whose constraint semantics arise from the class of predictors it favors.

Accordingly, one and the same topology-aware term may be described simultaneously as:

- a loss function in optimization,
- a regularizer in statistics,
- a geometry-preserving constraint on learned structure,
- a surrogate structural-alignment device,
- or a multi-objective component.

These are not contradictory descriptions. They refer to different levels of interpretation.

#### 2.13.3 Official interpretation rule

In the official benchmark, topology-aware losses are interpreted as structural-alignment constraints acting on the learned representation through auxiliary topology supervision.

They do not replace the intervention target. They shape the class of admissible or favored predictors so that the learned system remains sensitive to epidemic-relevant structural reorganization.

#### 2.13.4 Multi-term decomposition

If multiple predictive and topology-aware terms are used, the objective must be written explicitly as
\[
\mathcal L_{\mathrm{total}}
=
\lambda_{\mathrm{int}}\mathcal L_{\mathrm{int}}
+
\lambda_{\mathrm{supp}}\mathcal L_{\mathrm{supp}}
+
\sum_{j=1}^{J}\lambda_j \mathcal L_j^{\mathrm{top}},
\]
with all coefficients frozen as fixed hyperparameters unless the benchmark explicitly allows learning them.

#### 2.13.5 Weight-policy rule

If any coefficient is learned rather than tuned externally, the benchmark must state whether it is:

- unconstrained,
- nonnegative,
- normalized,
- hierarchically structured,
- or regularized.

Its statistical interpretation must also be declared.

#### 2.13.6 Identifiability warning

A topology-aware penalty does not identify a unique epidemiological mechanism. At most, it supports the claim that structural alignment of the declared type improves prediction of the declared simulator-defined target under the declared benchmark design.

---

### 2.14 Scientific questions and formal hypotheses

#### 2.14.1 Domain-legible research questions

The benchmark should state a small set of research questions readable by epidemiologists, statisticians, network scientists, and public-health readers without topology specialization.

Official question classes include:

1. Can temporal contact structure predict SAIS intervention-response quantities from observed windows?
2. Do topology-aware structural-alignment constraints improve intervention-response estimation relative to serious non-topological alternatives?
3. Are topology-aware gains concentrated in structural regimes where intervention leverage plausibly depends on reorganization, bridging, closure, or redundancy?
4. Do lower-level support tasks such as link prediction or structural-transition forecasting help build representations useful for the primary intervention task?

#### 2.14.2 Formal primary hypotheses

Each primary scientific question must map to explicit hypotheses on frozen intervention metrics.

Example for intervention sufficiency:
\[
H_0:
\mathbb E\!\left[
\mathrm{Err}_{a^\star}(M_{\mathrm{top}})
-
\mathrm{Err}_{a^\star}(M_{\mathrm{base}})
\right]
\ge 0,
\]
versus
\[
H_1:
\mathbb E\!\left[
\mathrm{Err}_{a^\star}(M_{\mathrm{top}})
-
\mathrm{Err}_{a^\star}(M_{\mathrm{base}})
\right]
< 0.
\]

Example for containment quality:
\[
H_0:
\mathbb E\!\left[
\mathrm{IoU}_{\mathcal C}(M_{\mathrm{top}})
-
\mathrm{IoU}_{\mathcal C}(M_{\mathrm{base}})
\right]
\le 0,
\]
versus
\[
H_1:
\mathbb E\!\left[
\mathrm{IoU}_{\mathcal C}(M_{\mathrm{top}})
-
\mathrm{IoU}_{\mathcal C}(M_{\mathrm{base}})
\right]
> 0.
\]

#### 2.14.3 Secondary and support-task hypotheses

Secondary hypotheses may be stated for:

- response-surface Brier score,
- baseline outbreak-probability slices,
- baseline attack-rate slices,
- or support-task performance.

These hypotheses are secondary and may not overwrite the interpretation of the primary intervention results.

#### 2.14.4 Structural-signal hypotheses

The benchmark may additionally posit preregistered hypotheses such as:

- topology-aware gains are concentrated in bridge-rich windows,
- topology-aware gains are concentrated in highly reconfigured windows,
- or topology-aware gains are strongest near containment-boundary regimes.

Such hypotheses must be stated before evaluation and tested under frozen regime definitions.

---

### 2.15 Metrics, proper scoring, and information-theoretic evaluation

#### 2.15.1 Frozen evaluation-target principle

The benchmark distinguishes sharply among:

1. the latent intervention-response estimand
   \[
   Y_t(v,a,\kappa),
   \]
2. its Monte Carlo evaluation estimator
   \[
   \widehat Y_t^{\mathrm{eval}}(v,a,\kappa),
   \]
3. the continuous burden estimand
   \[
   \mu_t(v,a,\kappa),
   \]
4. its Monte Carlo evaluation estimator
   \[
   \widehat \mu_t^{\mathrm{eval}}(v,a,\kappa),
   \]
5. and the derived intervention estimands
   \[
   a_t^\star(v,\kappa),\qquad
   \kappa_t^\star(v,a),\qquad
   \mathcal C_t(v).
   \]

The official benchmark uses independently generated evaluation Monte Carlo batches distinct from all training-label batches. This separation applies both to response-surface evaluation and to all derived intervention quantities.

#### 2.15.2 Primary intervention metrics

The benchmark is intervention-first. Therefore the primary reported metrics are defined on the primary intervention estimands.

For minimum intervention sufficiency, define
\[
\mathrm{MAE}_{a^\star}
=
\frac{1}{N_{\mathrm{test}}}
\sum_{t\in\mathcal T_{\mathrm{test}}}
\left|
\widehat a_t^\star - \widehat a_{t,\mathrm{eval}}^\star
\right|,
\]
or the corresponding RMSE version if the benchmark freezes squared error.

For minimum awareness sufficiency, define
\[
\mathrm{MAE}_{\kappa^\star}
=
\frac{1}{N_{\mathrm{test}}}
\sum_{t\in\mathcal T_{\mathrm{test}}}
\left|
\widehat \kappa_t^\star - \widehat \kappa_{t,\mathrm{eval}}^\star
\right|.
\]

For containment-region estimation, define a frozen set-level metric on the intervention grid, for example intersection-over-union:
\[
\mathrm{IoU}_{\mathcal C}
=
\frac{1}{N_{\mathrm{test}}}
\sum_{t\in\mathcal T_{\mathrm{test}}}
\frac{
\left|
\widehat{\mathcal C}_t \cap \widehat{\mathcal C}_{t,\mathrm{eval}}
\right|
}{
\left|
\widehat{\mathcal C}_t \cup \widehat{\mathcal C}_{t,\mathrm{eval}}
\right|
}.
\]

If an alternate containment metric is preferred, such as symmetric-difference area or boundary-distance error, that metric must be frozen explicitly and used consistently.

#### 2.15.3 Primary surface-level score

Because primary intervention quantities are derived from the SAIS response surface, the benchmark also reports a primary surface-level score:
\[
\mathrm{BS}_{\mathrm{surf}}
=
\frac{1}{N_{\mathrm{test}}\,|\mathcal G|}
\sum_{t\in\mathcal T_{\mathrm{test}}}
\sum_{(v,a,\kappa)\in\mathcal G}
\left(
\widehat Y_t(v,a,\kappa)
-
\widehat Y_t^{\mathrm{eval}}(v,a,\kappa)
\right)^2,
\]
where \(\mathcal G\) is the frozen intervention grid.

This score is secondary to the intervention metrics above, but primary among surface-level metrics.

#### 2.15.4 Secondary probability and burden metrics

Secondary metrics may include:

- Brier score on a frozen baseline probability slice,
- log score on a frozen baseline probability slice,
- MAE or RMSE for \(\widehat \mu_t(v,a,\kappa)\) on a frozen baseline slice,
- calibration diagnostics for selected probability slices.

These are scientifically useful but do not define the main benchmark claim.

#### 2.15.5 Calibration diagnostics

Calibration diagnostics must state exactly which target they reference.

For scalar probability slices, calibration diagnostics are computed relative to
\[
\widehat Y_t^{\mathrm{eval}}(v,a,\kappa)
\]
at the frozen slice.

For intervention-decision outputs such as sufficiency classification, calibration must be defined on the corresponding binary decision event and not conflated with probability-surface calibration.

#### 2.15.6 Metric freeze

The final benchmark freeze must declare:

- primary intervention metric for \(a_t^\star\),
- primary intervention metric for \(\kappa_t^\star\),
- primary containment metric for \(\mathcal C_t\),
- primary surface-level score,
- any secondary scalar-risk metrics,
- tie-break rule,
- and practical significance thresholds for declaring a win.

No later section may use a different implicit target object.

#### 2.15.7 Multi-metric reporting rule

No single metric fully characterizes intervention-quality prediction. The benchmark must therefore report a small, coherent set of complementary metrics spanning:

- primary intervention sufficiency error,
- containment quality,
- response-surface fidelity,
- and calibration where relevant.

#### 2.15.8 Information-theoretic interpretation

If log score or related quantities are used on any probability slice, the benchmark must state their information-theoretic interpretation and must identify the exact slice or response-surface marginal to which that interpretation applies.

---

### 2.16 Uncertainty quantification

#### 2.16.1 Label uncertainty

Uncertainty from Monte Carlo estimation of labels must be quantified or at least bounded.

#### 2.16.2 Optimization uncertainty

Variation across random initializations, stochastic minibatching, and random-feature approximations must be quantified.

#### 2.16.3 Evaluation uncertainty

Uncertainty in held-out metrics must respect chronology and dependence.

#### 2.16.4 Pairwise model-comparison uncertainty

For model comparison, the project must report uncertainty on differences in metrics, not only point estimates per model.

#### 2.16.5 Regime-specific uncertainty

When regime-conditioned results are reported, uncertainty must be recomputed for each regime rather than inherited from the global analysis.

#### 2.16.6 Approved interval procedures

The project must freeze which procedures are used, for example:
- paired bootstrap,
- block bootstrap,
- rolling-origin repeated splits,
- permutation tests,
- or another dependency-respecting method.

#### 2.16.7 Required reporting

Every reported gain must state whether the uncertainty interval excludes a null or practically negligible effect.

---

### 2.17 Statistical design of training, tuning, and selection

#### 2.17.1 Chronology-respecting split design

The project must define train, validation, and test sets in chronological order.

Random shuffling is prohibited for official evaluation.

#### 2.17.2 Information-flow firewall

The project must specify exactly what each stage may access:

- training data may access only the training set;
- hyperparameter tuning may access only the validation set;
- final model assessment may access only the test set after all design choices are frozen.

#### 2.17.3 Hyperparameter search design

Search space, search budget, search method, and selection criterion must be specified in advance.

#### 2.17.4 Early stopping as statistical selection

Early stopping is a finite-sample model-selection rule and must be treated as such.

Validation peeking through repeated retries must be governed explicitly.

#### 2.17.5 Batch size, epoch budget, patience, restart count

These are part of the statistical design because they affect variance, optimization noise, implicit regularization, and selection bias.

#### 2.17.6 Nested versus non-nested selection

The project must declare whether hyperparameter tuning and model-family comparison are nested or whether a single validation set is reused. The consequences for optimism must be acknowledged.

#### 2.17.7 Final refit policy

If the final model is refit on train+validation before test evaluation, this must be declared explicitly.

---

### 2.18 Multiple comparisons and benchmark credibility

#### 2.18.1 Family-wise comparison problem

The project compares multiple model families, baselines, penalties, sensitivity settings, and possibly multiple topological-loss constructions.

This creates a multiple-comparison problem.

#### 2.18.2 Required controls

At minimum, the benchmark must include:
- a no-topology baseline,
- a simpler structural-summary baseline,
- and any serious non-topological comparator judged necessary by the final benchmark design.

#### 2.18.3 Selective-reporting control

The project must declare which experiments are primary, which are secondary, and which are exploratory.

Primary conclusions may not depend on exploratory comparisons.

#### 2.18.4 Meaningful-improvement threshold

The project must define what counts as a meaningful gain in predictive performance, not merely a numerically nonzero gain.

#### 2.18.5 Stability of wins

A claimed win must survive:
- plausible seed variation,
- alternate admissible calibrations where specified,
- and major structural regimes.

---

### 2.19 Regime-conditioned analysis

#### 2.19.1 Structural regimes

The project must predefine structural regimes such as:

- sparse vs dense contact windows,
- fragmented vs cohesive windows,
- bridge-rich vs bridge-poor windows,
- loop-rich vs loop-poor windows,
- stable vs highly reconfigured windows.

#### 2.19.2 Risk regimes

Define low-risk, intermediate-risk, and high-risk windows using frozen thresholds on the estimand or its estimator.

#### 2.19.3 Transition regimes

If transition behavior is important, define transition windows explicitly rather than using informal language.

#### 2.19.4 Conditional evaluation

All regime-specific evaluations must state:
- regime definition,
- sample size within regime,
- metric definition within regime,
- and uncertainty procedure within regime.

#### 2.19.5 Heterogeneity-of-effect analysis

The project may ask whether topology-aware gains are concentrated in particular structural or risk regimes. Such analysis must be declared before evaluation to avoid cherry-picking.

---

### 2.20 Sensitivity and robustness program

#### 2.20.1 Epidemic-process sensitivity

The official benchmark does not vary epidemic family. SAIS is fixed.

Accordingly, epidemic-process sensitivity is limited to alternate admissible SAIS parameterizations and simulator conventions within the SAIS family, such as:

- alternate alerting-rate regimes,
- alternate awareness-decay regimes if formally admitted,
- alternate recovery-rate regimes,
- and alternate horizon lengths.

Sensitivity across SIS, SIR, SEIR, SEIS, or any non-SAIS epidemic class is out of scope for the official benchmark unless introduced by formal amendment.

#### 2.20.2 SAIS parameter sensitivity

Assess sensitivity to:
- \(\beta\),
- \(\beta_A\),
- \(\kappa\),
- \(\alpha\),
- \(\delta\),
- seeding assumptions,
- outbreak threshold \(z_{\mathrm{out}}\),
- burden threshold \(b^\star\) if used,
- and acceptance threshold \(q^\star\).

#### 2.20.3 Intervention sensitivity

Assess sensitivity to:
- intervention family \(v\),
- intervention-grid resolution,
- intervention intensity range \(\mathcal A_v\),
- targeted versus untargeted intervention rules,
- and any randomized intervention-selection mechanism.

#### 2.20.4 Windowing sensitivity

Assess sensitivity to:
- daily versus multi-day windows,
- overlapping versus disjoint windows,
- historical context length,
- and whether event-stream or window-summary access changes substantive conclusions.

#### 2.20.5 Topological-construction sensitivity

Assess sensitivity to:
- filtration choice,
- homology dimension,
- heat-kernel scale,
- random-feature dimension,
- structural depth activation,
- reference-object choice only when used as an explicit ablation,
- and whether the topology-aware term remains beneficial under the frozen structural-change pipeline.

#### 2.20.6 Metric sensitivity

Assess whether substantive conclusions depend on:
- primary intervention metric choice,
- containment metric choice,
- response-surface score,
- secondary Brier or log score on baseline slices,
- and calibration summary choice.

#### 2.20.7 Split sensitivity

Where feasible, assess alternate chronology-respecting split choices or rolling-origin variants.

#### 2.20.8 Robustness statement rule

A robustness claim is valid only relative to the frozen SAIS sensitivity grid. No broader statement is licensed.

---

### 2.21 Validation and external credibility

#### 2.21.1 Internal validation

Internal validation consists of chronology-respecting held-out evaluation under frozen design choices.

#### 2.21.2 Simulator realism checks

If the simulator is used to define labels, the project must specify any realism checks used to confirm that the induced target distribution is not pathological.

#### 2.21.3 Empirical alignment

The project must state which aspects of the design are empirically informed by:
- contact-network evidence,
- behavioral epidemiology literature,
- or public-health mechanism studies.

#### 2.21.4 External validation boundary

If no external validation exists, the project must state exactly what claims are withheld until such validation is performed.

---

### 2.22 Exclusion, invalidity, and failure rules

#### 2.22.1 Invalid label-generation run

A label-generation run is invalid if:
- simulator execution fails,
- output is degenerate beyond declared admissibility,
- required uncertainty estimates cannot be computed,
- or calibration/freeze invariants are violated.

#### 2.22.2 Invalid training run

A training run is invalid if:
- the objective becomes undefined,
- NaNs or infinities occur,
- chronology or leakage checks fail,
- or required topological objects cannot be computed.

#### 2.22.3 Invalid evaluation

An evaluation is invalid if:
- test support is insufficient under frozen rules,
- leakage is discovered,
- or required metrics cannot be computed in a well-defined way.

#### 2.22.4 Disclosure rule

Invalid runs must be logged and disclosed according to the reporting policy. Silent dropping is forbidden.

#### 2.22.5 Fail-fast consistency

These rules must match the global fail-fast philosophy of the project.

---

### 2.23 What this section must allow a statistician to conclude

After reading this section alone, a statistician or information theorist should be able to answer all of the following without consulting the implementation:

1. What is observed?
2. What is conditioned on?
3. What is random?
4. What is the primary estimand?
5. How are labels generated and how noisy are they?
6. What does the topological penalty mean statistically?
7. Which epidemic-relevant structural signals is topology supposed to encode?
8. Which scoring rules are primary and why?
9. How are uncertainty and multiple comparisons handled?
10. What sensitivity analyses are required before any claim is permitted?
11. What the project may claim, and what it may not claim.

---

### 2.24 Embedded ledgers for Section II

#### 2.24.A Estimand ledger

The official estimand ledger is:

- \(Y_t(v,a,\kappa)\): latent outbreak-probability response surface conditional on \(\mathcal W_t\); primary latent response object.
- \(\mu_t(v,a,\kappa)\): latent burden response surface conditional on \(\mathcal W_t\); secondary latent response object if activated.
- \(a_t^\star(v,\kappa)\): minimum intervention sufficiency; primary intervention estimand.
- \(\kappa_t^\star(v,a)\): minimum awareness sufficiency; primary intervention estimand.
- \(\mathcal C_t(v)\): containment region on the frozen intervention grid; primary intervention estimand.
- \(\pi_t^{\mathrm{trans}}(v,a,\kappa)\): regime-transition risk; primary or secondary intervention estimand depending on benchmark freeze.
- support-task estimands such as future contact probability or baseline outbreak probability: support estimands only.

#### 2.24.B Label-estimator ledger

The official label-estimator ledger is:

- simulator output \(Z_{t,H}^{(m;v,a,\kappa)}\): simulated epidemic burden draw under SAIS.
- Monte Carlo probability estimator:
  \[
  \widehat Y_t(v,a,\kappa)
  =
  \frac{1}{M_{t,v,a,\kappa}}
  \sum_{m=1}^{M_{t,v,a,\kappa}}
  \mathbf 1\{Z_{t,H}^{(m;v,a,\kappa)}\ge z_{\mathrm{out}}\};
  \]
  training target or evaluation target depending on batch identity.
- Monte Carlo burden estimator:
  \[
  \widehat \mu_t(v,a,\kappa)
  =
  \frac{1}{M_{t,v,a,\kappa}}
  \sum_{m=1}^{M_{t,v,a,\kappa}}
  Z_{t,H}^{(m;v,a,\kappa)};
  \]
  optional training or evaluation target.
- derived intervention label estimators:
  \[
  \widehat a_t^\star,\quad
  \widehat \kappa_t^\star,\quad
  \widehat{\mathcal C}_t;
  \]
  derived supervised labels for the primary benchmark task.

#### 2.24.C Topology-semantic ledger

The official topology-semantic ledger is:

- \(D_t\): topological state of the observed window; interpretable as static window topology.
- \(C_t=D_t-D_{t-\tau}\): signed structural-change object; interpretable as reorganization of connectivity, bridges, loops, fillings, and accessibility.
- \(\mathcal L_W\): geometric structural-alignment penalty on predicted versus observed structural change.
- \(\mathcal L_H\): harmonic-analytic structural-alignment penalty on predicted versus observed structural change.
- recursive higher-depth objects: compatible event-cluster geometry; secondary interpretation only.
- non-topological competitors: spectral summaries, degree summaries, density, clustering, contact totals, and other simpler structural covariates.

#### 2.24.D Metric ledger

The official metric ledger is:

- \(\mathrm{MAE}_{a^\star}\) or frozen equivalent: primary metric for intervention sufficiency.
- \(\mathrm{MAE}_{\kappa^\star}\) or frozen equivalent: primary metric for awareness sufficiency.
- \(\mathrm{IoU}_{\mathcal C}\) or frozen equivalent: primary metric for containment-region estimation.
- \(\mathrm{BS}_{\mathrm{surf}}\): primary surface-level score.
- baseline-slice Brier or log score: secondary metric only.
- support-task metrics: support-task reporting only.

#### 2.24.E Sensitivity ledger

The official sensitivity ledger is:

- SAIS parameters \((\beta,\beta_A,\kappa,\alpha,\delta)\): primary sensitivity axis.
- intervention family and intensity range: primary sensitivity axis.
- forecast horizon \(H\): primary sensitivity axis.
- outbreak threshold \(z_{\mathrm{out}}\): primary sensitivity axis.
- burden tolerance \(b^\star\): secondary sensitivity axis if burden-based sufficiency is used.
- windowing policy and history length: primary design sensitivity axis.
- filtration choice, homology dimension, heat-kernel scale, random-feature dimension: topology sensitivity axes.
- structural depth activation: secondary sensitivity axis.

#### 2.24.F Invalidity ledger

The official invalidity ledger is:

- label-generation invalidity: simulator failure, degenerate response surfaces beyond frozen admissibility rules, missing required Monte Carlo uncertainty objects, or calibration violation.
- training invalidity: undefined loss, NaNs or infinities, chronology failure, leakage, or failure to compute required topology objects.
- evaluation invalidity: insufficient support for the frozen intervention grid, leakage, or undefined primary intervention metrics.
- reporting invalidity: omission of uncertainty, omission of frozen metric definition, or silent dropping of invalid runs.

---

### 2.25 Section-closing principle

This section treats the project as a formally specified intervention-first predictive experiment under simulator-defined SAIS response targets, not as an informal risk-prediction exercise.

No topology-aware construction may be retained in the final design unless this section can state:

1. what epidemic-relevant structural signal it is intended to encode,
2. how that signal enters the intervention-response objective,
3. what non-topological alternative it must beat,
4. how its gain will be evaluated under uncertainty and sensitivity analysis,
5. and whether it supports the primary intervention task or only a secondary support task.

No primary scientific claim may be made unless it is expressible in terms of the frozen SAIS intervention-response estimands defined in this section.

---

## III. Machine Learning and Deep Learning

### 3.1 Purpose and governing machine-learning principle

This section gives the complete machine-learning, deep-learning, and implementation specification of the project.

Its purpose is to define, without omission:

1. the computational learning task;
2. the exact active model families;
3. the mandatory baseline suite;
4. the exact role of topology in learning;
5. the model architectures and internal components;
6. the optimization and training protocol;
7. the implementation stack and reproducibility contract;
8. the computational complexity and tractability limits;
9. the artifact, caching, and checkpoint policy;
10. the testing, invariant, and failure rules;
11. the hardware and execution environment;
12. the official benchmark harness and reporting requirements.

This section must be sufficient for a machine-learning researcher or deep-learning systems engineer to reproduce the benchmark without ambiguity.

No implementation, model choice, optimization choice, or library dependency may be introduced later unless it is declared here.

The governing principle is:

> Every model, every baseline, every optimization choice, and every implementation detail must be explicitly specified as a frozen design object.

No behavior may be left to implicit defaults.

---

### 3.2 Benchmark ontology and computational task

#### 3.2.2 Input object

The benchmark uses two formally distinct encoder interfaces:

1. **event-stream interface**
   \[
   \mathcal E_{t,h}
   =
   \{e_\ell:\ e_\ell \text{ occurs in } [t-h,t]\},
   \]
   used by event-based dynamic-graph models where supported;

2. **window-summary interface**
   \[
   \mathcal X_t
   =
   \mathsf X(\mathcal W_t^{\mathrm{evt}},G_t),
   \]
   used by tabular and heuristic baselines.

Every model family must declare which interface it uses.

For the primary intervention task, the encoder input alone is not the full benchmark input. The full benchmark input consists of:

- the encoder input,
- the intervention family \(v\),
- and the frozen intervention query grid or query slice attached to that instance.

No later section may define the input as only a window summary while omitting the intervention query object required by the primary output interface.

#### 3.2.3 Output object

All official benchmark models must support an intervention-first output interface.

The primary supervised object is the SAIS intervention-response surface on a frozen intervention grid:
\[
\widehat{\mathbf Y}_t(v)
=
\left\{
\widehat Y_t(v,a,\kappa):
(a,\kappa)\in\mathcal G_v
\right\},
\]
or, if the continuous burden surface is the directly supervised object,
\[
\widehat{\boldsymbol\mu}_t(v)
=
\left\{
\widehat \mu_t(v,a,\kappa):
(a,\kappa)\in\mathcal G_v
\right\}.
\]

Primary benchmark outputs are derived from this structured output object:

- minimum intervention sufficiency \(\widehat a_t^\star(v,\kappa)\),
- minimum awareness sufficiency \(\widehat \kappa_t^\star(v,a)\),
- containment region \(\widehat{\mathcal C}_t(v)\),
- and any frozen regime-transition quantity derived from the same response surface.

Secondary outputs may include:

- a baseline large-outbreak probability slice,
- a baseline attack-rate slice,
- support-task predictions such as link probability or structural-transition scores.

The benchmark must not describe these outputs as a vague “hierarchical task interface.” The official structure is a primary intervention-response output with optional support-task heads.

#### 3.2.4 Window-level adaptation layer

Because several named baseline models originate in temporal link prediction or node-level temporal tasks, the benchmark freezes a common adaptation layer:

- the encoder processes the admissible event-stream or window history up to time \(t\);
- the encoder produces event-level states, node-level states, graph-level states, or a temporal-memory state, depending on model family;
- these are aggregated by a frozen pooling operator
  \[
  \mathsf{Pool}_t,
  \]
  yielding a single window representation
  \[
  h_t^{\mathrm{win}};
  \]
- the primary intervention head maps
  \[
  h_t^{\mathrm{win}}
  \mapsto
  \widehat{\mathbf Y}_t(v),
  \]
  or to the corresponding burden surface if that is the directly supervised object.

Optional support heads may additionally map \(h_t^{\mathrm{win}}\) to:

- link-prediction targets,
- structural-transition targets,
- or baseline scalar epidemic-risk targets.

No baseline may be evaluated on a different primary supervision target or a different prediction granularity for official intervention-task comparison.

#### 3.2.5 Fairness rule

The benchmark must not compare:

- link-level outputs from one model,
- node-level outputs from another,
- scalar risk outputs from a third,
- and structured intervention outputs from a fourth,

without first passing all models through the same frozen intervention-response interface for the primary benchmark task.

Support-task comparisons are allowed, but they must be labeled explicitly as support-task evaluations and may not be conflated with primary intervention-task comparison.

#### 3.2.6 Learning objective

The total objective is
\[
\mathcal L_{\mathrm{total}}
=
\lambda_{\mathrm{int}}\mathcal L_{\mathrm{int}}
+
\lambda_{\mathrm{supp}}\mathcal L_{\mathrm{supp}}
+
\sum_{j=1}^{J}
\lambda_j
\mathcal L_j^{\mathrm{top}},
\]
where:

- \(\mathcal L_{\mathrm{int}}\) is the primary intervention-response loss,
- \(\mathcal L_{\mathrm{supp}}\) collects any activated support-task losses,
- and the \(\mathcal L_j^{\mathrm{top}}\) are frozen topology-aware losses.

The intervention term is mandatory. Support-task terms are optional but must be frozen before official evaluation. Every active term must be instantiated explicitly.

### 3.2A Official benchmark task structure

#### Primary task

The primary benchmark task is SAIS intervention-response estimation.

This includes:

- minimum intervention sufficiency,
- minimum awareness sufficiency,
- containment-region estimation,
- containment-boundary estimation if activated,
- and regime-transition quantities derived from the same response surface.

#### Support tasks

Support tasks are retained because they help evaluate whether the learned representation captures temporal structure relevant to the primary task.

Support structural tasks may include:

- dynamic link prediction,
- structural-transition forecasting.

Support epidemic tasks may include:

- baseline outbreak-probability prediction,
- baseline attack-rate prediction.

These support tasks may be used for auxiliary training, benchmarking, or ablation, but they do not define the primary leaderboard.

---

### 3.3 Exact role of topology in the ML pipeline

This must be absolutely frozen.

#### 3.3.1 Active topology role

The active role of topology is:

> **topology-as-loss / topology-as-constraint**

This is the official benchmark design.

More precisely, the official core benchmark uses topology-aware losses as auxiliary structural-alignment supervision attached to learned representations or auxiliary topology heads while the primary supervised target remains the SAIS intervention-response task.

#### 3.3.2 Explicitly inactive roles

The following are not active in the official core benchmark:

- topology as concatenated feature vector,
- topology as encoder input channel,
- topology as message-passing operator,
- topology as graph transformer token branch,
- topology as learned latent representation stream.

These may exist only as future exploratory work.

#### 3.3.3 Conditionally admissible but inactive role

Output-induced topology loss is mathematically admissible, but it is inactive by default in the official core benchmark unless a later subsection explicitly freezes:

- the output object,
- the induced filtration,
- the topology reference object,
- and the exact task role.

Accordingly, the official benchmark distinguishes:

- **active core role**: structural-alignment loss on the observed-window topology pipeline;
- **inactive but admissible role**: output-topology loss on model outputs or response surfaces.

#### 3.3.4 Justification

This freeze aligns:

- the mathematical formalism already developed,
- the statistical semantics already frozen,
- the intervention-first target structure,
- and the benchmark comparability requirements.

No fused-input PH branches are permitted in the official benchmark, and no empirical section may blur the distinction between active topology-as-loss and inactive admissible alternatives.

---

### 3.4 Exact active model families

This subsection defines the complete official model family suite.

All official neural model families share the same high-level architecture:

\[
\text{Encoder}
\rightarrow
\text{Window Representation } h_t^{\mathrm{win}}
\rightarrow
\text{Primary Intervention Head}
\]

Optional support heads and topology heads may be attached to the shared representation, but the primary task is always intervention-response estimation under SAIS.

---

#### 3.4.1 Family M0 — non-topological intervention model

This is the primary neural control family.

Architecture:
\[
\text{Encoder}
\rightarrow
\text{Primary Intervention Head}
\]

Optional support-task heads may be present if the benchmark freezes auxiliary support-task training, but no topology-aware loss is active.

Loss:
\[
\mathcal L_{\mathrm{int}}
\]
or
\[
\mathcal L_{\mathrm{int}} + \lambda_{\mathrm{supp}}\mathcal L_{\mathrm{supp}}
\]
if support-task multitask training is officially activated.

Purpose:
isolates the contribution of the encoder and intervention head without topology-aware constraints.

---

#### 3.4.2 Family M1 — Wasserstein topology-constrained intervention model

Architecture:
\[
\text{Encoder}
\rightarrow
\text{Primary Intervention Head}
\]
with an auxiliary topology head producing
\[
\widehat C_t.
\]

Loss:
\[
\mathcal L_{\mathrm{int}}
+
\lambda_{\mathrm{supp}}\mathcal L_{\mathrm{supp}}
+
\lambda_W \mathcal L_W
\]
with \(\lambda_{\mathrm{supp}}=0\) if support-task multitask training is inactive.

This is the official geometric topology-aware model.

Purpose:
tests whether direct geometric structural-alignment regularization improves intervention-response estimation.

---

#### 3.4.3 Family M2 — RKHS topology-constrained intervention model

Architecture:
\[
\text{Encoder}
\rightarrow
\text{Primary Intervention Head}
\]
with an auxiliary topology head producing
\[
\widehat C_t.
\]

Loss:
\[
\mathcal L_{\mathrm{int}}
+
\lambda_{\mathrm{supp}}\mathcal L_{\mathrm{supp}}
+
\lambda_H \mathcal L_H.
\]

This is the official harmonic-analytic topology-aware model.

Purpose:
tests whether smoothed structural-alignment regularization in the VPD-induced RKHS improves intervention-response estimation.

---

#### 3.4.4 Family M3 — combined Wasserstein-plus-RKHS topology model

Architecture:
\[
\text{Encoder}
\rightarrow
\text{Primary Intervention Head}
\]
with an auxiliary topology head producing
\[
\widehat C_t.
\]

Loss:
\[
\mathcal L_{\mathrm{int}}
+
\lambda_{\mathrm{supp}}\mathcal L_{\mathrm{supp}}
+
\lambda_W \mathcal L_W
+
\lambda_H \mathcal L_H.
\]

This family is allowed in the official benchmark only if the document explicitly declares whether the combined topology model is part of the main leaderboard or only a secondary comparison.

Purpose:
tests whether the geometric and harmonic-analytic topology constraints are complementary.

---

#### 3.4.5 Family M4 — higher-order recursive topology model

Architecture:
\[
\text{Encoder}
\rightarrow
\text{Primary Intervention Head}
\]
with the higher-depth recursive topology branch activated.

Loss:
\[
\mathcal L_{\mathrm{int}}
+
\lambda_{\mathrm{supp}}\mathcal L_{\mathrm{supp}}
+
\lambda_R \mathcal L_R.
\]

This family is optional and secondary.

It must not be included in the official core benchmark unless explicitly activated by amendment.

#### 3.4.6 Family-comparison rule

All official model-family comparisons must be made on the same frozen primary intervention task, the same intervention grid, the same SAIS simulator-defined labels, the same split design, and the same evaluation metrics.

No model family may be advantaged by being compared on a weaker scalar-risk target while another is compared on the full intervention-response task.

---

### 3.5 Mandatory baseline suite

This is the **official benchmark baseline suite** for the intervention-first SAIS benchmark.

Nothing may be removed without amendment, but official status must distinguish between:

- **primary intervention baselines**, which are eligible for the main leaderboard,
- **support-task baselines**, which are eligible only for support-task evaluation,
- and **exploratory comparators**, which are excluded from the official primary leaderboard unless fully adapted to the same intervention-response interface.

---

#### 3.5.1 Baseline B0 — tabular structural intervention regressor

Input:
- degree moments,
- spectral radius,
- clustering coefficient,
- connected components,
- window density,
- temporal event count,
- bridge proxies,
- and any other frozen non-topological window summaries listed in the structural-summary ledger.

Model:
**LightGBM**

Output:
the same frozen intervention-response target used by the neural models, either directly as a response surface or through a frozen intervention-label interface.

This is the canonical tabular intervention baseline.

---

#### 3.5.2 Baseline B1 — spectral intervention control

Input:
- leading eigenvalue,
- Laplacian spectrum summaries,
- algebraic connectivity,
- clustering statistics,
- assortativity,
- and any other frozen spectral controls.

Model:
**XGBoost**

Output:
the same frozen intervention-response target used by the neural models.

Purpose:
tests whether topology-aware gains reduce to spectral summaries alone.

---

#### 3.5.3 Baseline B2 — dynamic link-prediction support baseline

Frozen exact source model:
**EdgeBank** or another frozen link-prediction baseline if the benchmark later chooses a stronger alternative.

Official status:
- support-task comparator only by default,
- excluded from the official primary intervention leaderboard unless fully adapted to the same intervention-response interface,
- may appear in appendix or support-task sections without claiming primary-task comparability.

Because such models are natively link-prediction baselines rather than intervention-response estimators, they may not be presented as core intervention baselines unless their adaptation is fully frozen and benchmarked on the same primary targets.

---

#### 3.5.4 Baseline B3 — recency-persistence-popularity structural heuristic

Frozen exact heuristic:
**Recency-Persistence-Popularity Structural Heuristic**

This heuristic is retained because it is scientifically interpretable and provides a simple temporal-structure control.

For each window \(t\), define the frozen scalar summaries:

- **recency score**
  \[
  R_t
  =
  \sum_{(i,j)\in E_t}
  \exp\!\left(-\frac{\Delta_{ij}(t)}{\tau_R}\right),
  \]

- **persistence score**
  \[
  P_t
  =
  \sum_{(i,j)\in E_t}
  \frac{1}{h}
  \sum_{s=t-h+1}^{t}
  \mathbf 1\{(i,j)\in E_s\},
  \]

- **popularity/intensity score**
  \[
  U_t
  =
  \sum_{i\in V}
  \deg_t(i)^2.
  \]

The frozen scalar heuristic is
\[
H_t
=
\alpha_R R_t
+
\alpha_P P_t
+
\alpha_U U_t,
\]
with frozen nonnegative coefficients \((\alpha_R,\alpha_P,\alpha_U)\).

Official role:
- as a support-task or scalar-control baseline by default;
- as a primary intervention baseline only if the document explicitly freezes a map from \(H_t\) into the same intervention-response target used by all official primary baselines.

If the benchmark uses this heuristic only on a scalar support task, then the corresponding calibrated scalar output must be labeled support-task only and may not enter the primary intervention leaderboard.

If the benchmark promotes this heuristic to a primary intervention baseline, then it must define one of the following explicitly:

- a response-surface map
  \[
  (v,a,\kappa,H_t)\mapsto \widehat Y_t(v,a,\kappa),
  \]
- a derived intervention-label map
  \[
  H_t \mapsto \widehat a_t^\star,\ \widehat \kappa_t^\star,\ \widehat{\mathcal C}_t,
  \]
- or another frozen intervention-output construction.

No heuristic may remain on a weaker scalar-risk target while neural models are evaluated on the full intervention-response task.

This baseline is mandatory because it directly tests whether simple temporal persistence and hub-concentration heuristics already explain the benchmark signal.

---

#### 3.5.5 Baseline B4 — TGN

Frozen exact model:
**Temporal Graph Network (TGN)**

This is the canonical memory-based dynamic GNN baseline.

Mandatory.

---

#### 3.5.6 Baseline B5 — TGAT

Frozen exact model:
**Temporal Graph Attention Network (TGAT)**

Mandatory.

---

#### 3.5.7 Baseline B6 — DyGFormer

Frozen exact model:
**DyGFormer**

Mandatory.

This is the strongest modern transformer baseline.

---

#### 3.5.8 Baseline B7 — DyRep

Official status:
- **secondary neural baseline**,
- included in the extended benchmark suite,
- excluded from the minimum mandatory benchmark suite.

Minimum mandatory neural suite:
- TGN,
- TGAT,
- DyGFormer.

Extended neural suite:
- TGN,
- TGAT,
- DyGFormer,
- DyRep.

The document must not use phrases such as “recommended mandatory.” DyRep is either mandatory or secondary; in the official benchmark it is secondary.

---

#### 3.5.9 Baseline B8 — hypergraph higher-order control

Frozen choice:
**EpiDHGNN**

Official status:
- included in the extended benchmark suite,
- excluded from the official minimum mandatory suite,
- required in any section or table that makes higher-order structural claims.

Therefore:
- if a result is used to support only pairwise-contact-window claims, EpiDHGNN is not required in the primary table;
- if a result is used to support claims about higher-order structure, clustered event geometry, simplicial closure, or recursive topological depth, EpiDHGNN must be included in the corresponding comparator table.

This removes the ambiguous phrase “conditional mandatory” and replaces it with a claim-dependent reporting rule.

---

### 3.6 Exact encoder architecture freeze

This section leaves nothing unspecified.

---

#### 3.6.1 Encoder interface

Frozen official interface:
**event-stream temporal encoder**

Fallback:
window-sequence encoder only for models that cannot support event streams.

---

#### 3.6.2 Hidden dimension

Default:
\[
d_h = 128
\]

Search grid:
\[
\{64,128,256\}
\]

---

#### 3.6.3 Layer depth

Default:
\[
L=2
\]

Search grid:
\[
\{2,3,4\}
\]

---

#### 3.6.4 Dropout

Default:
\[
p=0.2
\]

Search grid:
\[
\{0.1,0.2,0.3\}
\]

---

#### 3.6.5 Activation

Frozen:
**GELU**

No alternatives unless explicitly tested.

---

#### 3.6.6 Normalization

Frozen:
**LayerNorm**

---

#### 3.6.7 Historical context

Default history:
\[
h=7
\]
windows

Search grid:
\[
\{3,7,14\}
\]

---

### 3.7 Exact topology-loss implementation freeze

---

#### 3.7.1 Wasserstein branch

Exact loss:
\[
\mathcal L_W
=
W_1(Z_1,Z_2)
\]

Matching:
recomputed per batch

Gradient convention:
fixed matching per backward pass

---

#### 3.7.2 RKHS branch

Exact theoretical object:
\[
\mathcal L_H
=
\|\Phi(Z_1)-\Phi(Z_2)\|_{\mathcal H}^2
\]

Practical implementation:
**Random Fourier Features**

Default dimension:
\[
m=512
\]

Grid:
\[
\{256,512,1024\}
\]

---

#### 3.7.3 Topology-loss schedule

The official benchmark computes topology-aware losses **every batch**.

No alternate frequency schedule is permitted in the official benchmark.

If computational stress tests later evaluate reduced-frequency schedules, those runs must be labeled exploratory and may not replace official results.

---

#### 3.7.4 Loss weights

Default:
\[
\lambda_W = 0.1
\]
\[
\lambda_H = 0.1
\]

Grid:
\[
\{0.01,0.05,0.1,0.5,1.0\}
\]

---

### 3.8 Optimization protocol

This must be fully explicit.

---

#### 3.8.1 Optimizer

Frozen:
**AdamW**

---

#### 3.8.2 Learning rate

Default:
\[
10^{-3}
\]

Grid:
\[
\{10^{-4},3\times10^{-4},10^{-3}\}
\]

---

#### 3.8.3 Weight decay

Default:
\[
10^{-4}
\]

---

#### 3.8.4 Scheduler

Frozen:
**cosine decay with linear warmup**

Warmup epochs:
\[
5
\]

---

#### 3.8.5 Gradient clipping

Frozen:
\[
\|\nabla\|_2 \le 1.0
\]

---

#### 3.8.6 Epoch budget

Maximum:
\[
200
\]

---

#### 3.8.7 Early stopping

Patience:
\[
20
\]

Selection metric:
validation Brier score

---

#### 3.8.8 Batch size

Default:
\[
64
\]

Grid:
\[
\{32,64,128\}
\]

---

#### 3.8.9 Mixed precision

Frozen:
**enabled**

---

### 3.9 Exact implementation stack

---

#### 3.9.1 Language

**Python 3.11**

---

#### 3.9.2 Core framework

**PyTorch 2.x**

---

#### 3.9.3 Graph framework

**PyTorch Geometric**

---

#### 3.9.4 Dynamic graph harness

**DyGLib-compatible harness**

Mandatory.

---

#### 3.9.5 TDA libraries

Frozen:

- **GUDHI**
- **Ripser.py**

---

#### 3.9.6 Tabular baselines

Frozen:

- **LightGBM**
- **XGBoost**

---

#### 3.9.7 Scientific stack

Frozen:

- NumPy
- SciPy
- pandas
- scikit-learn

---

### 3.10 Hardware environment

Everything must be logged.

---

#### 3.10.1 Required logging

- CPU
- GPU
- RAM
- OS
- CUDA
- PyTorch version
- PyG version

---

#### 3.10.2 Official benchmark tier

GPU:
**single RTX 4090 or equivalent**

---

#### 3.10.3 Smoke-test tier

CPU-safe minimal benchmark required.

---

### 3.11 Artifact and cache contract

---

#### 3.11.1 Mandatory saved artifacts

- raw windows
- graph objects
- topology objects
- simulator labels
- checkpoints
- validation logs
- test metrics
- plots
- tables

---

#### 3.11.2 Cache keys

Must include:

- dataset version
- window version
- topology version
- simulator version
- model family
- hyperparameter hash
- seed

---

#### 3.11.3 Silent cache reuse

Forbidden.

---

### 3.12 Reproducibility and seeds

---

#### 3.12.1 Seed set

Frozen:
\[
\{1,2,3,4,5\}
\]

Official results:
mean ± standard deviation

---

#### 3.12.2 Determinism

Deterministic operations required wherever supported.

---

### 3.13 Complexity and tractability

---

#### 3.13.1 Complexity ledger

Every model must report:

- preprocessing complexity
- per-step train complexity
- inference complexity
- memory complexity

---

#### 3.13.2 Topology complexity

Must separately report:

- PH computation cost
- Wasserstein cost
- RKHS approximation cost

---

### 3.14 Testing and invariants

Mandatory tests:

- chronology leakage
- shape
- dtype
- NaN/Inf
- cache consistency
- topology consistency
- simulator consistency

All required before official run.

---

### 3.15 Crash-on-logic-error policy

Immediate abort on:

- leakage
- NaN
- undefined topology
- stale cache
- invalid checkpoint
- invalid split
- unsupported depth

No silent fallback permitted.

---

### 3.16 Final official benchmark freeze

**Main families**
- M0
- M1
- M2

**Mandatory baselines**
- B0
- B1
- B2
- B3
- B4
- B5
- B6

**Conditional**
- B7
- B8

No deviations without amendment.
---

## IV. Network Science

### 4.1 Purpose and governing network-science principle

This section gives the complete network-science specification of the project.

Its purpose is to define, without omission:

1. the exact network objects under study;
2. the epidemic dynamical processes acting on those objects;
3. the temporal-network assumptions that determine their scientific meaning;
4. the classical graph-theoretic, spectral, and complex-network comparators;
5. the null and perturbation models required to test whether any topological signal is reducible to simpler structure;
6. the structural regimes in which spreading behavior and predictive value may differ;
7. the network-science interpretation of all topology-bearing constructions used elsewhere in the project.

The governing principle of this section is the following:

> The scientific object is not a graph in isolation, but an epidemic dynamical process coupled to a temporal interaction structure.

Accordingly, the primary network-science object is the pair
\[
(\mathcal G, \mathcal D),
\]
where

- \(\mathcal G\) is the observed or derived temporal interaction structure, and
- \(\mathcal D\) is the epidemic dynamical process acting on that structure.

No topological claim is admissible unless it is positioned against serious non-topological alternatives in this coupled sense.

This section overlaps with the Mathematics section but serves a different role. The Mathematics section defines the formal epidemic equations, persistence constructions, recursive depth, and harmonic-analytic machinery. The present section defines how those objects are interpreted from the standpoint of temporal networks, complex networks, spectral network theory, higher-order interactions, and network epidemiology.

---

### 4.2 Exact network object of study

#### 4.2.1 Node semantics

Each node represents one social entity capable of participating in contact events that may mediate transmission opportunity. In the active benchmark, a node represents an individual person unless the dataset-specific specification states otherwise.

The node set is denoted
\[
V=\{1,\dots,n\}.
\]

All node inclusion, exclusion, and identifier-normalization rules must be frozen before any structural analysis is performed.

#### 4.2.2 Event semantics

The raw observed network data are temporal contact events. A contact event is an ordered or unordered tuple
\[
e_\ell = (i_\ell, j_\ell, s_\ell, u_\ell, w_\ell, m_\ell),
\]
where:

- \(i_\ell, j_\ell \in V\) are the participants;
- \(s_\ell\) is the event start time;
- \(u_\ell\) is the event end time or stop time;
- \(w_\ell\) is an optional duration or intensity weight;
- \(m_\ell\) is optional metadata.

If the underlying data only record contact timestamps but not durations, then \(u_\ell\) is omitted and contact is treated as instantaneous.

#### 4.2.3 Contact interpretation

A contact event is interpreted as an observed opportunity for transmission, not necessarily a transmission itself. This distinction is fundamental.

The project does not assume that every contact corresponds to equal biological transmission probability. Rather, the contact network is treated as a structural substrate on which epidemic dynamics are defined.

#### 4.2.4 Edge semantics

Edges exist in several derived representations:

1. **Event-level contact relation**  
   An edge exists at time \(t\) if there is an active contact event involving nodes \(i\) and \(j\).

2. **Window-level aggregated edge**  
   An edge exists in window \(W_t\) if at least one contact event between \(i\) and \(j\) occurs during that window.

3. **Weighted window-level edge**  
   Edge weight may represent:
   - contact count,
   - total contact duration,
   - average duration,
   - recency-weighted count,
   - or another frozen function of the event history in the window.

The exact choice must be frozen in the dataset-specific benchmark specification.

#### 4.2.5 Directionality

The primary contact-network representation is undirected unless the data contain a scientifically meaningful direction of transmission opportunity. If directionality is present in the raw data but dropped in the benchmark, this choice must be documented and justified.

#### 4.2.6 Repeated contacts

Repeated contacts are not discarded. They are represented in one of the following ways, depending on the derived object:

- as repeated events in the event stream;
- as cumulative edge weight in a window;
- as duration aggregation in a window;
- as temporal persistence of an edge across windows.

The design must state explicitly which of these are preserved in each representation.

#### 4.2.7 Derived network object family

The project uses the following network object family:

1. **Raw event stream**
   \[
   \mathcal E = \{e_\ell\}_{\ell=1}^L.
   \]

2. **Temporal network**
   \[
   \mathcal G^{\mathrm{temp}} = (V,\mathcal E).
   \]

3. **Windowed graph sequence**
   \[
   \{\mathcal G_t\}_{t=1}^T,
   \]
   where each \(\mathcal G_t=(V,E_t,W_t)\) is the aggregated contact graph for window \(t\).

4. **Clique-complex lift**
   For a graph \(\mathcal G_t\), the project may construct its clique complex \(X_t\) or a filtered clique complex derived from weighted edges.

5. **Topology-bearing structural object**
   Persistence diagrams, virtual persistence diagrams, and recursive higher-depth constructions derived from \(X_t\).

These objects must never be conflated. They are distinct network-science representations of the same underlying contact process.

---

### 4.3 Epidemic dynamics on networks

This subsection defines how epidemic processes are interpreted on the network objects above.

#### 4.3.1 Population-level baseline models

The classical well-mixed SIS model is
\[
\frac{dS}{dt} = -\beta SI + \gamma I,
\qquad
\frac{dI}{dt} = \beta SI - \gamma I.
\]

The classical well-mixed SIR model is
\[
\frac{dS}{dt} = -\beta SI,
\qquad
\frac{dI}{dt} = \beta SI - \gamma I,
\qquad
\frac{dR}{dt} = \gamma I.
\]

These equations are included only as population-level baselines. They do not represent contact heterogeneity, temporal ordering, or structural constraints.

#### 4.3.2 Node-level static-network epidemic equations

For a static adjacency matrix \(A=(A_{ij})\), a node-level SIS approximation may be written as
\[
\frac{dp_i}{dt}
=
\beta (1-p_i)\sum_{j=1}^n A_{ij}p_j
-
\gamma p_i,
\]
where \(p_i(t)\) is the probability node \(i\) is infected at time \(t\).

This formalizes how local network structure enters epidemic dynamics.

#### 4.3.3 Static-network epidemic process

For a static graph \(G=(V,E)\), the epidemic process is defined by:

- node states in \(\{S,I\}\), \(\{S,I,R\}\), or another frozen compartment set;
- transmission along edges \((i,j)\in E\);
- recovery, immunity loss, or progression rules according to the chosen epidemic class.

The exact process family is frozen in the mathematical and statistical sections; the present section interprets its network consequences.

#### 4.3.4 Temporal-network epidemic process

For a temporal event stream or graph sequence, transmission may occur only along active contacts.

The project must state which of the following is active:

1. **Event-driven process**  
   Transmission can occur only during active event intervals.

2. **Window-driven process**  
   Transmission is defined on an aggregated graph within a window.

3. **Hybrid process**  
   A window is the observed structural unit, but event ordering within the window still affects transmission.

The active choice must be frozen and used consistently across label generation, interpretation, and evaluation.

#### 4.3.5 Time-scale relation

The project must explicitly state the relative scale of:

- contact dynamics,
- epidemic progression,
- awareness or behavior dynamics if included,
- windowing dynamics.

These may lie in distinct regimes:

- **fast-switching network regime**: contacts fluctuate much faster than epidemic state changes;
- **slow-switching network regime**: network structure is quasi-static during epidemic progression;
- **intermediate regime**: network and epidemic coevolve on comparable time scales.

This distinction is scientifically central because threshold behavior and effective mixing depend on it.

#### 4.3.6 Concurrency

Concurrency is defined as the extent to which a node participates in overlapping or near-simultaneous contacts or partnerships. The exact concurrency metric must be frozen dataset-by-dataset.

Concurrency is treated as a first-class structural variable because it changes path availability and epidemic reachability in temporal networks.

#### 4.3.7 Burstiness

Burstiness refers to irregular inter-event timing, often characterized by heavy-tailed inter-contact intervals or clustered event activity. If used, the project must fix a mathematical burstiness measure, such as a coefficient based on the mean and standard deviation of inter-event times.

Burstiness is scientifically important because epidemic spread may differ dramatically between regular and bursty contact schedules even when static aggregate graphs are similar.

#### 4.3.8 Edge persistence and repeated-contact retention

Edge persistence refers to whether the same pair of nodes remains in contact across multiple windows or repeatedly reappears.

This quantity is treated as distinct from simple edge weight because:
- one long repeated relationship and
- many independent one-off contacts

may have different epidemiological meaning despite similar aggregate counts.

---

### 4.4 Temporal granularity and windowing

#### 4.4.1 Window policy

The benchmark must specify the exact windowing rule:
- daily,
- rolling daily,
- disjoint multi-day,
- overlapping multi-day,
- or another explicit policy.

No implicit default is allowed.

#### 4.4.2 Window span

Let the window span be denoted \(\Delta_w\). This must be frozen.

#### 4.4.3 Window indexing

Let
\[
\mathcal W_t
\]
denote the set of events assigned to window \(t\), and
\[
\mathcal G_t
\]
the graph or weighted graph derived from \(\mathcal W_t\).

#### 4.4.4 Intra-window aggregation

The project must explicitly state how events in \(\mathcal W_t\) are aggregated:

- binary aggregation:
  \[
  A_{ij}^{(t)} = \mathbf 1\{\text{at least one event between }i,j\text{ in } \mathcal W_t\},
  \]
- count aggregation:
  \[
  A_{ij}^{(t)} = \#\{\text{events between }i,j\text{ in } \mathcal W_t\},
  \]
- duration aggregation:
  \[
  A_{ij}^{(t)} = \sum_{\ell: (i_\ell,j_\ell)=(i,j),\, e_\ell\in \mathcal W_t} \mathrm{dur}(e_\ell),
  \]
- or another frozen rule.

#### 4.4.5 Inter-window dependence

The benchmark must specify whether windows are treated as:

- independent observational units for a simplified benchmark, or
- a linked temporal sequence where dependence is retained.

The active design must agree with Sections II and III.

#### 4.4.6 Preserved and discarded temporal information

For the chosen windowing policy, the section must explicitly state what is preserved and what is discarded.

Possible preserved information:
- event existence,
- event count,
- duration totals,
- edge persistence across windows,
- partial temporal order.

Possible discarded information:
- precise within-window order,
- exact inter-event intervals,
- sub-window concurrency,
- path timing details.

This declaration is mandatory because scientific interpretation depends on it.

---

### 4.5 Classical complex-network comparator families

This subsection defines serious graph ensembles and historical comparator families from network science.

#### 4.5.1 Erdős–Rényi random graphs

The \(G(n,p)\) family is used as a homogeneous random-contact comparator. It tests whether a phenomenon depends on broad nonrandom structure beyond mean degree.

#### 4.5.2 Lattice and geometric-neighbor graphs

Lattice or local geometric comparators are used to test strongly local propagation regimes:
- nearest-neighbor spread,
- spatially constrained spread,
- localized clustering without hub dominance.

These are scientifically important because epidemic spreading on lattices historically provides a contrasting regime to well-mixed or random-network dynamics.

#### 4.5.3 Small-world graphs

Watts–Strogatz or comparable small-world families are used to represent:
- strong local clustering,
- short characteristic path length,
- intermediate structure between lattice and random graph.

#### 4.5.4 Scale-free graphs

Barabási–Albert or other heavy-tailed degree models are used to represent:
- hub-dominated connectivity,
- strong degree heterogeneity,
- unusual threshold behavior.

#### 4.5.5 Configuration models

Degree-preserving randomization models are used to isolate the contribution of the degree sequence from other structural features.

#### 4.5.6 Spatial random geometric graphs

If spatial interpretation is relevant to the dataset or scientific claim, spatial random geometric graphs may be used as locality-preserving comparators.

#### 4.5.7 Higher-order graph families

If higher-order structural claims are made, the benchmark must acknowledge:
- hypergraph comparators,
- simplicial-complex comparators,
- and projected pairwise controls.

These are essential if the project claims to capture genuinely group-level interaction effects rather than only pairwise graph properties.

---

### 4.6 Standard network-science structural summaries

This subsection defines the strong non-topological structural alternatives that topology-aware methods must compete against.

#### 4.6.1 Degree-based summaries

The project must define and, where appropriate, compute:

- mean degree,
- degree variance,
- maximum degree,
- degree skewness,
- degree quantiles,
- heavy-tail diagnostics.

#### 4.6.2 Component and connectivity summaries

The project must define:

- number of connected components,
- giant-component fraction,
- component-size distribution,
- bridge count,
- articulation-point count.

#### 4.6.3 Path-based summaries

The project must define:

- average shortest path length,
- diameter,
- reachability fraction,
- eccentricity summaries.

#### 4.6.4 Clustering and closure summaries

The project must define:

- local clustering coefficient,
- global clustering coefficient,
- transitivity,
- triangle count,
- clique count where appropriate.

#### 4.6.5 Centrality summaries

The project must define:

- betweenness centrality,
- closeness centrality,
- eigenvector centrality,
- k-core or coreness,
- bridge-centrality or articulation-based measures where used.

#### 4.6.6 Community summaries

The project must define:

- modularity,
- conductance,
- cut ratio,
- inter-community edge fraction,
- community-size distribution if relevant.

#### 4.6.7 Temporal summaries

The project must define, where applicable:

- burstiness,
- edge persistence,
- repeated-contact retention,
- inter-event interval dispersion,
- temporal reachability or temporal path summaries.

#### 4.6.8 Spectral summaries

The project must define:

- adjacency spectral radius \(\lambda_1(A)\),
- algebraic connectivity,
- Laplacian spectrum summaries,
- other frozen spectral proxies if used.

These summaries are treated as primary comparators, not afterthoughts.

---

### 4.7 Topology versus graph statistics

This is an empirical question, not a slogan.

#### 4.7.1 Formal comparison question

The section must define the following benchmark question:

> Do persistence-based and VPD-based summaries capture predictive structural information beyond standard network-science summaries?

#### 4.7.2 Reduction-to-simpler-structure question

The benchmark must test whether apparent topology-aware gains reduce to:
- degree heterogeneity,
- clustering and closure,
- bridge or articulation structure,
- community structure,
- spectral structure,
- temporal persistence,
- or burstiness.

#### 4.7.3 Comparator classes that topology must beat

No topology-aware claim is admissible unless comparison is made against at least:

1. degree/sparsity summaries,
2. clustering/closure summaries,
3. bridge/component summaries,
4. community/modularity summaries,
5. spectral summaries,
6. temporal persistence/burstiness summaries.

#### 4.7.4 Interpretation discipline

If topology-aware models win only in settings where graph-statistic baselines are weak, this must be stated plainly. If graph-statistic baselines erase the gain, that must also be stated plainly.

---

### 4.8 Spectral structure and epidemic relevance

This subsection positions spectral network science as the strongest non-topological alternative.

#### 4.8.1 Static SIS threshold relation

For static-network SIS dynamics, the project recognizes the classical threshold relation
\[
\tau_c \approx \frac{1}{\lambda_1(A)},
\]
where \(\lambda_1(A)\) is the adjacency spectral radius and \(\tau_c\) is the epidemic threshold under the relevant approximation.

#### 4.8.2 SIR-related spectral relevance

For SIR-like spreading, spectral structure remains informative but is not interpreted identically to SIS threshold theory. The section must distinguish threshold relevance from final-size or severity relevance.

#### 4.8.3 Temporal spectral proxies

For temporal networks, the benchmark must freeze how spectral structure is summarized:
- time-averaged adjacency spectral radius,
- window-by-window spectral sequence,
- supra-adjacency or multilayer spectral proxy,
- or another explicit temporal spectral object.

#### 4.8.4 Limits of spectral sufficiency

The section must explicitly state that spectral radius and related threshold surrogates are strong but not universally sufficient summaries of epidemic risk. Structural context, temporal ordering, community organization, and higher-order effects may alter their usefulness.

#### 4.8.5 Spectral-vs-topological benchmark

One of the central questions of the project is:

> Does topology add predictive or structural information beyond spectral structure?

This must be treated as a primary benchmark question.

---

### 4.9 Dynamic-network assumptions

#### 4.9.1 Instantaneous versus durational contact

The project must state whether contacts are treated as:
- point events,
- duration intervals,
- or both.

#### 4.9.2 Repeated-contact handling

The project must state whether repeated contact:
- increases weight only,
- defines persistence,
- remains as a separate event history,
- or all three.

#### 4.9.3 Edge persistence across windows

The section must define whether edges are reconstructed independently in each window or whether persistence across windows is a first-class structural variable.

#### 4.9.4 Temporal ordering retained

The project must define whether the network interpretation retains:
- complete event order,
- within-window order,
- cross-window order only,
- or no order after aggregation.

#### 4.9.5 Memory effects

The benchmark must state which of the following memory effects are scientifically in scope:
- memory in edge recurrence,
- memory in contact timing,
- memory in node interaction preference,
- memory in behavior-induced rewiring.

#### 4.9.6 Adaptive rewiring and behavior-driven contact change

If the project uses behaviorally meaningful structural interpretations, this section must state whether behavior is interpreted as acting through:
- edge deletion,
- reduced effective edge weight,
- selective contact avoidance,
- rewiring toward safer contacts,
- or no explicit network-topology change.

This is necessary because many epidemic-behavior models distinguish behavior acting on transmission parameters from behavior acting on contact topology, and the present project may need both interpretations.

---

### 4.10 Network misspecification and proxy risk

#### 4.10.1 Missing-contact risk

Observed contact events may omit actual transmission-relevant interactions.

#### 4.10.2 Timestamp censoring

Event times may be coarsened, censored, or truncated.

#### 4.10.3 Duration distortion

Recorded duration may not equal biologically relevant exposure.

#### 4.10.4 Node subsampling

The observed network may exclude relevant nodes or populations.

#### 4.10.5 Measurement bias

Certain contact types may be systematically under-observed or over-observed.

#### 4.10.6 Proxy mismatch

The observed temporal contact network may be only a proxy for effective transmission opportunity. This limitation must be stated whenever strong structural claims are made.

#### 4.10.7 Aggregation distortion

Windowing may induce structural artifacts:
- loops may appear or disappear due to aggregation,
- bridge structure may be exaggerated or hidden,
- concurrent events may collapse into static closure.

This is particularly important for the topology-aware interpretation.

---

### 4.11 Null and perturbation models

This subsection defines the executable null-model program used to test whether topology-aware gains are reducible to simpler structure.

For every null family below, the benchmark must freeze:

- whether the null acts on raw events or window graphs,
- the exact algorithm,
- the number of null replicates per analyzed object,
- the random seed policy,
- the summary statistics preserved by construction.

#### 4.11.1 Degree-preserving rewiring null

Object acted on:
window graph \(\mathcal G_t\).

Algorithm:
perform a frozen number \(K_{\mathrm{rewire}}\) of double-edge swaps that preserve the degree sequence and reject self-loops or duplicate edges unless explicitly allowed.

Preserves:
- node set,
- degree sequence.

Destroys:
- community structure,
- clustering specifics,
- bridge geometry,
- topological organization not implied by degrees.

Purpose:
tests whether signal reduces to degree heterogeneity.

#### 4.11.2 Timestamp-shuffle null

Object acted on:
raw event stream within the benchmark horizon.

Algorithm:
permute event timestamps among fixed contact pairs, preserving the multiset of contact times and the multiset of interacting pairs.

Preserves:
- static aggregate graph,
- pair frequencies.

Destroys:
- temporal order,
- concurrency structure,
- timing alignment.

Purpose:
tests whether temporal ordering matters beyond static structure.

#### 4.11.3 Pair-shuffle null

Object acted on:
raw event stream.

Algorithm:
permute node-pair labels among fixed timestamps, preserving the timestamp multiset while randomizing which pair received which event.

Preserves:
- event-time pattern.

Destroys:
- original structural organization.

Purpose:
tests whether structure matters beyond timing.

#### 4.11.4 Duration-preserving null

Object acted on:
durational event stream where durations are available.

Algorithm:
preserve the multiset of event durations while randomizing their assignment to contact pairs or timestamps according to a frozen rule.

Purpose:
tests whether duration heterogeneity alone explains the signal.

#### 4.11.5 Burst-preserving null

Object acted on:
pairwise event histories or whole-event stream, depending on the dataset.

Algorithm:
preserve inter-event interval distributions according to a frozen burst-preserving reshuffle while destroying higher-order alignment across pairs.

Purpose:
tests whether burstiness alone explains the signal.

#### 4.11.6 Spectrally matched null

Object acted on:
window graph \(\mathcal G_t\).

Algorithm:
construct or approximate a rewired control whose leading spectral quantity is within a frozen tolerance
\[
|\lambda_1(\mathcal G_t)-\lambda_1(\mathcal G_t^{\mathrm{null}})|\le \varepsilon_\lambda.
\]

Purpose:
tests whether topology contributes beyond the strongest spectral comparator family.

#### 4.11.7 Community-preserving / bridge-randomizing null

Object acted on:
window graph \(\mathcal G_t\) together with a frozen community assignment rule.

Algorithm:
approximately preserve community sizes or modularity while randomizing cross-community bridge placement.

Purpose:
tests whether bridge geometry and leakage drive the observed topology-aware gain.

#### 4.11.8 Official null hierarchy

The required official nulls are:

1. degree-preserving rewiring,
2. timestamp shuffle,
3. spectral-matched null where computationally feasible.

Recommended nulls:
- pair shuffle,
- duration-preserving null,
- community-preserving / bridge-randomizing null.

Exploratory nulls:
- burst-preserving and higher-order nulls if computationally feasible.

No null family may be added post hoc to rescue or attack a result after benchmark outcomes are already known.

#### 4.11.3 Pair-shuffle null

Shuffle contact pairs while preserving the timing pattern.

Purpose:
tests whether structure matters beyond timing.

#### 4.11.4 Duration-preserving randomization

Preserve contact durations while randomizing allocation or ordering.

Purpose:
tests whether duration heterogeneity alone explains the effect.

#### 4.11.5 Burst-preserving randomization

Preserve inter-event interval or burst structure while destroying higher-level structure where feasible.

Purpose:
tests whether burstiness alone explains the signal.

#### 4.11.6 Community-preserving / bridge-randomizing null

Approximately preserve community structure while randomizing bridge placement.

Purpose:
tests whether bridge geometry and leakage patterns are central to the signal.

#### 4.11.7 Spectrally matched but topologically altered null

Construct or approximate nulls with similar leading spectral quantities but different structural organization.

Purpose:
tests whether topology captures information beyond spectral structure.

#### 4.11.8 Higher-order nulls

If higher-order structures are used, define nulls that:
- preserve pairwise projections,
- but destroy genuine group interactions.

#### 4.11.9 Official null hierarchy

The final benchmark must freeze which nulls are:
- required,
- recommended,
- exploratory.

No null may be used opportunistically after looking at results.

---

### 4.12 Structural regimes

This subsection defines the structural stratification of the benchmark.

#### 4.12.1 Density regimes

The benchmark must define:
- sparse windows,
- moderate-density windows,
- dense windows.

#### 4.12.2 Connectivity regimes

The benchmark must define:
- fragmented regime,
- weakly connected regime,
- highly connected regime.

#### 4.12.3 Bridge regimes

The benchmark must define:
- bridge-rich,
- bridge-poor,
- articulation-dominated,
- bottlenecked.

#### 4.12.4 Closure regimes

The benchmark must define:
- low-clustering,
- high-clustering,
- high-filling / highly closed regimes where appropriate.

#### 4.12.5 Community regimes

The benchmark must define:
- strong modularity,
- weak modularity,
- high leakage,
- low leakage.

#### 4.12.6 Temporal regimes

The benchmark must define:
- bursty vs regular,
- persistent-edge vs rapidly reconfigured,
- high concurrency vs low concurrency.

#### 4.12.7 Transition regimes

The benchmark must define windows or periods in which structural indicators are changing rapidly, since these may be especially informative for topology-aware constraints.

#### 4.12.8 Freeze rule

Every regime definition must be fixed before any regime-conditioned performance claims are made.

---

### 4.13 Network-science interpretation of topology-bearing objects

This subsection forces a scientific interpretation of the topological objects.

#### 4.13.1 Ordinary persistent-homology interpretation

For clique-complex-based persistent homology:

- \(H_0\) is interpreted as fragmentation, component accessibility, bridge-mediated connectivity, and merging structure.
- \(H_1\) is interpreted as loop or circulation structure arising from overlapping local contacts.
- fillings via triangles or higher simplices are interpreted as cohesive closure that can suppress or modify open-cycle structure.

#### 4.13.2 Internal window-comparison interpretation

When persistence objects from multiple windows are compared, the resulting signed structural-change object is interpreted as describing:
- bridge creation,
- bridge destruction,
- loop creation,
- loop snapping,
- filling creation,
- filling removal,
- fragmentation and reconnection.

#### 4.13.3 Repeated-contact persistence interpretation

If topological comparisons across windows are used, they may encode repeated-contact retention and structural persistence rather than merely static topology.

#### 4.13.4 RKHS / harmonic-analytic interpretation

When VPD comparisons are passed through the heat-kernel / RKHS semimetric, the resulting object is interpreted as a smoothed structural-change signature, not merely a raw geometric difference. This may be viewed as a candidate latent structural-risk functional, but this remains an empirical claim that must be benchmarked.

#### 4.13 Higher-order interactions, simplicial structure, and hypergraph alternatives

The official benchmark makes the following frozen distinction:

1. **pairwise contact structure** is the primary observed substrate;
2. **clique-complex lifts of pairwise graphs** are derived topology-bearing summaries of that pairwise substrate;
3. **genuine higher-order interactions** require direct hypergraph or simplicial data and are not assumed to be fully identified by clique-complex lifts.

Accordingly, the project may claim that its PH machinery summarizes higher-order structural organization **derived from pairwise contacts**, but it may not claim that this is identical to direct higher-order interaction data unless such data are explicitly present and benchmarked.

#### 4.13.6 Comparability requirement

Higher-depth objects are only scientifically meaningful when the compatibility or containment relations among events are respected. Not all structural events are combinable in a meaningful way.

#### 4.13.7 Restriction rule

If a higher-depth object does not admit a convincing network-science interpretation, then empirical claims involving it must be restricted accordingly, even if the mathematics remains valid.

---

### 4.14 Higher-order interactions, simplicial structure, and hypergraph alternatives

#### 4.14.1 Pairwise versus higher-order interaction distinction

The section must explicitly distinguish:

- pairwise contact structure,
- clique-complex lifts of pairwise graphs,
- genuine higher-order interactions represented directly as simplices or hyperedges,
- and recursive interval-of-interval constructions defined on topological event objects.

These are not the same scientific object.

#### 4.14.2 Simplicial-complex interpretation

The official clique-complex branch is interpreted as a higher-order summary derived from pairwise contact closure under the frozen filtration. It is not automatically equivalent to direct group-interaction data.

#### 4.14.3 Hypergraph interpretation

When direct hypergraph or simplicial interaction data exist, they represent genuine higher-order interaction more directly than clique-complex lifts. Such data must be treated as scientifically distinct from pairwise-derived closures.

#### 4.14.4 Recursive higher-depth network-science interpretation

If the recursive higher-depth branch is activated, the network-science interpretation must be frozen explicitly.

The official admissible interpretation is as follows:

- first-order persistence events are treated as atomic structural events;
- admissible compatible collections of such events define hyperedges in an event hypergraph;
- inclusion or refinement relations among compatible event collections define a directed acyclic graph on those hyperedges;
- recursive topology is then interpreted as topology on the geometry of compatible event-clusters rather than only on the original contact graph.

This interpretation is scientifically admissible only when the compatibility or containment relations are frozen explicitly and when empirical claims are restricted to what those relations actually encode.

#### 4.14.5 Relation to the project’s topology

The project must state, for each active branch, whether the PH machinery is:

- a higher-order summary of pairwise interactions,
- a partial proxy for genuinely higher-order contact organization,
- or a recursive summary of clustered structural events in the derived topological state space.

#### 4.14.6 Comparator requirement

If the project foregrounds higher-order structural claims, then at least one higher-order non-topological comparator family must appear in Section III, or the claims must be restricted to derived-topology interpretation rather than direct higher-order-contact identification.

---

### 4.15 Claim boundary for network science

#### 4.15.1 Admissible claims

This section may support claims about:
- structural organization of contact networks,
- predictive relevance of structural summaries,
- comparison of topology with graph-statistic and spectral controls,
- network regimes where topology-aware methods help,
- and, when justified, higher-order structural hypotheses.

#### 4.15.2 Inadmissible claims

This section does not by itself support:
- causal intervention claims,
- public-health policy recommendations,
- or claims that observed contact structure is a perfect proxy for transmission.

#### 4.15.3 Required takeaway

After reading this section alone, a network scientist should be able to answer:

1. What exact temporal network object is being studied?
2. What epidemic dynamics operate on it?
3. Which structural summaries are serious non-topological competitors?
4. Which null models test whether topology is redundant?
5. Which structural regimes matter?
6. Which higher-order constructions are genuinely network-scientific and which remain primarily mathematical?

---

### 4.16 Embedded ledgers for Section IV

#### 4.16.A Network object ledger

The official network object ledger is:

- raw event stream \(\mathcal E\),
- event window \(\mathcal W_t^{\mathrm{evt}}\),
- weighted window graph \(G_t=(V,E_t,w_t)\),
- clique-complex filtration \(\mathcal F_t\),
- persistence object \(P_t\),
- VPD state \(D_t\),
- signed structural-change object \(C_t=D_t-D_{t-\tau}\),
- higher-order network object if direct hypergraph or simplicial data are present,
- recursive topological object if higher-depth activation is used.

#### 4.16.B Structural summary ledger

The official structural-summary ledger must list every non-topological summary actually used, including at minimum any active members of:

- degree moments,
- density,
- connected components,
- clustering summaries,
- assortativity,
- temporal event counts,
- bridge proxies,
- recency, persistence, and popularity summaries,
- and any other tabular covariates admitted into the benchmark.

#### 4.16.C Spectral ledger

The official spectral ledger must list every spectral comparator actually used, including at minimum any active members of:

- leading eigenvalue,
- algebraic connectivity,
- Laplacian spectrum summaries,
- and any other explicitly benchmarked spectral control.

#### 4.16.D Null-model ledger

The official null-model ledger must list every active null or perturbation model and, for each one:

- what object it perturbs,
- what invariants it preserves,
- what structure it destroys,
- and what scientific redundancy question it addresses.

#### 4.16.E Regime ledger

The official regime ledger must list every structural regime used in analysis, including exact definitions for any active members of:

- sparse versus dense,
- fragmented versus cohesive,
- bridge-rich versus bridge-poor,
- loop-rich versus loop-poor,
- stable versus highly reconfigured,
- and any regime keyed to intervention difficulty or containment-boundary proximity.

#### 4.16.F Higher-order interpretation ledger

A frozen ledger listing every higher-order or recursive empirical object and the strongest admissible network-science interpretation the project is willing to defend.

---

### 4.17 Section-closing principle

No topology-aware result may be presented as scientifically interesting unless it survives comparison against:

- strong graph-statistic controls,
- strong spectral controls,
- temporal null models,
- and, where applicable, higher-order interaction alternatives.

The burden of proof in this section is not to show that topology is mathematically elegant. The burden is to show that it is network-scientifically nonredundant.

## 4.18 Official dataset freeze

### Dataset scope and benchmark-use policy

### Dataset scope and benchmark-use policy

Primary epidemiological datasets are frozen as the official benchmark datasets for main scientific conclusions:

Core SocioPatterns:
- hospital ward,
- primary school,
- high school,
- workplace,
- conference / Hypertext / SFHH.

Secondary epidemiological datasets may be used only for secondary analyses, robustness checks, or transfer-style sensitivity studies if the document later freezes their preprocessing and admissibility rules:

- flights,
- household networks,
- Kenyan households.

General ML datasets may be used exclusively for model-family maturity checks, implementation stress testing, or support-task engineering validation:

- Wikipedia,
- Reddit,
- MOOC,
- Enron,
- UCI.

Primary scientific conclusions, primary leaderboard results, and main intervention-task claims may only use the core epidemiological datasets unless this section is amended formally.

No later section may mix:

- primary epidemiological conclusions,
- secondary epidemiological sensitivity analyses,
- and general ML maturity checks

without labeling them separately and unambiguously.

---

## V. Public Health

### 5.1 Purpose and governing public-health principle

This section gives the complete public-health and epidemiological interpretation of the intervention-first SAIS benchmark.

Its purpose is to define, without omission:

1. the applied public-health problem class;
2. the disease-system classes to which the benchmark can and cannot speak;
3. the surveillance and operational use-cases to which the benchmark is relevant;
4. the epidemiological meaning of the simulator-derived intervention-response targets;
5. the intervention and prevention mechanisms that are in scope;
6. the public-health meaning of behavior, contact structure, and structural change;
7. the interpretation of contact data as a transmission substrate;
8. the burden types the benchmark approximates and the burden types it does not;
9. the public-health meaning of calibration, uncertainty, and miscalibration;
10. the domain validity threats, generalizability limits, and translational constraints;
11. the evidence ladder required before stronger applied claims are permitted;
12. the ethical, equity, and risk-communication boundaries of use.

The governing principle of this section is the following:

> This section must translate the project from mathematical, statistical, and network-science language into domain-valid public-health language, while sharply restricting any claim not supported by the benchmark’s simulator-conditional evidence structure.

The role of this section is distinct from the earlier sections:

- Section I defines the epidemic mathematics and topology-related formalism.
- Section II defines the SAIS intervention-response estimands, labels, and statistical design.
- Section III defines the machine-learning benchmark and implementation contract.
- Section IV defines the temporal-network object, structural comparators, and network-science interpretation.
- Section V defines what all of that can and cannot mean in infectious-disease and public-health terms.

To prevent domain slippage, the section uses the following public-health exactness taxonomy:

- **Domain object**: a real public-health quantity or process.
- **Model-based proxy**: a simulator-defined or benchmark-defined quantity used to approximate a domain-relevant concept.
- **Operational signal**: a quantity plausibly usable in surveillance, prioritization, or preparedness.
- **Non-operational research quantity**: useful scientifically but not yet suitable for deployment.

The benchmark is not a field trial, not a surveillance system, and not a policy-effect estimator. All applied interpretation in this section must remain conditional on the frozen SAIS simulator, the frozen intervention family, and the empirical contact-data scope defined elsewhere in the document.
- **Unsupported applied claim**: a statement the project is not permitted to make.

No public-healthly meaningful quantity, use-case, restriction, or interpretive boundary may be introduced later unless it is formally defined here.

OKAY YOU WERE VAGUE SO HERE: Primary claims concern intervention-relevant epidemic quantities under behavior-aware simulation.

The benchmark may claim:

- improved threshold estimation
- improved containment prediction
- improved intervention sufficiency estimation

The benchmark may not claim real-world policy efficacy without external validation.

---

### 5.2 Public-health problem statement

The applied objective of the project is to understand and predict epidemic-risk-related quantities from temporal contact structure in settings relevant to infectious-disease spread.

Stated in domain language, the project asks:

> Given observed patterns of interpersonal contact over time, can one identify windows of contact structure that are more likely to support substantial disease transmission under a declared epidemic model?

The central public-health motivation is that outbreaks are shaped not only by pathogen biology, but also by who contacts whom, when, how often, for how long, and under what behavioral and environmental conditions. In many real settings, contact structure may provide useful signal before full clinical burden is observable. A benchmark that predicts simulator-defined outbreak risk from contact structure may therefore be relevant to early structural risk assessment, situational awareness, or preparedness-oriented analysis, provided its limits are stated honestly.

This benchmark is not, by default:

- a causal intervention model,
- a clinically validated outbreak forecasting system,
- a hospitalization-demand model,
- or a deployment-ready public-health decision tool.

It is a scientifically structured benchmark for **contact-structured epidemic-risk prediction under explicit assumptions**.

---

### 5.3 Disease-system classes and public-health use-cases

The project must not treat epidemic model classes as mere mathematical containers. Each model class corresponds to a disease-system class with distinct public-health meaning.

#### 5.3.1 Acute immunizing infections

This class includes diseases in which infection is followed by recovery and at least temporary protection against reinfection. In this setting, SIR-like dynamics are substantively appropriate.

Public-healthly relevant features include:

- finite infectious period,
- epidemic waves,
- attack rate and final-size relevance,
- outbreak suppression through prevention or isolation,
- possibly strong value of early warning.

Examples of suitable use-cases include:
- outbreak-risk monitoring in acute community spread,
- structural analysis of superspreading opportunity,
- preparedness for wave-like transmission.

#### 5.3.2 Recurrent or endemic infections

This class includes diseases or settings in which recovery does not produce durable immunity and reinfection remains possible. In this setting, SIS-like dynamics are substantively appropriate.

Public-healthly relevant features include:

- ongoing prevalence,
- recurrence,
- endemic circulation,
- re-entry into susceptibility,
- sustained rather than one-wave burden.

Examples of suitable use-cases include:
- persistent transmission environments,
- chronic circulation settings,
- repeated-risk surveillance.

#### 5.3.3 Latent or incubating infections

This class includes diseases in which infection and infectiousness are separated by an exposed or latent stage, with possible delay between infection, symptom onset, and detection. In this setting, SEIR- or SEIS-like models may be substantively required.

Public-healthly relevant features include:

- delayed symptom recognition,
- latent transmission chains,
- presymptomatic or asymptomatic spread,
- mismatch between observed illness and actual infection progression.

Examples of suitable use-cases include:
- early-warning contexts where observed case counts lag true transmission,
- settings where exposure and infectiousness are temporally separated.

#### 5.3.4 Behavior-sensitive infections

This class includes diseases for which human behavior substantially changes exposure or effective transmission. Relevant mechanisms include:

- distancing,
- masking,
- protective equipment,
- sexual behavior modification,
- risk avoidance,
- contact substitution,
- adherence fatigue,
- awareness-driven changes.

Examples of suitable use-cases include:
- respiratory outbreaks with self-protection and contact adaptation,
- sexually transmitted infections with partner-behavior change,
- awareness-sensitive community outbreaks.

#### 5.3.5 Resource-sensitive or care-sensitive infections

This class includes settings in which medical resources, treatment access, care-seeking, isolation capacity, or healthcare response materially alter progression, detection, or control.

Examples of suitable use-cases include:
- healthcare-associated transmission settings,
- resource-limited outbreak response,
- settings where care access changes effective recovery or onward spread.

#### 5.3.6 Public-health use-case mapping

The benchmark may be relevant to one or more of the following use-cases, depending on the simulator family and validation status:

- early structural risk warning,
- situational awareness,
- outbreak investigation prioritization,
- preparedness analysis,
- contact-structure-based vulnerability assessment,
- structural comparison across windows or settings.

It is not automatically relevant to:
- direct clinical diagnosis,
- patient-level prognosis,
- causal intervention evaluation,
- hospital operations forecasting,
- field attack-rate estimation,
- or public communication without expert mediation.

#### 5.3.7 Out-of-scope disease classes

Unless specifically extended, the benchmark is not assumed to validly represent:

- vector-borne transmission systems where human contact structure is not the primary transmission substrate,
- strongly environmentally mediated transmission systems where recorded contact is a weak proxy,
- pathogen systems requiring detailed within-host or clinical progression modeling beyond the current simulator class,
- settings where contact structure is secondary to institutional, environmental, or healthcare-system transmission routes.

---

### 5.4 Epidemiological quantities of interest in public-health language

This subsection defines the public-health interpretation of the benchmark targets and related quantities.

#### 5.4.1 Attack rate

Attack rate is the proportion of the relevant population infected over the specified epidemic horizon under the declared simulator.

In public-health language, this is a burden-like quantity reflecting how much spread occurs in the modeled scenario.

#### 5.4.2 Large-outbreak probability

The large-outbreak probability is the probability, under the simulator and parameter regime, that a window’s contact structure generates an outbreak exceeding the benchmark-defined threshold.

This is a **model-based outbreak-risk quantity**, not an observed field frequency.

#### 5.4.3 Horizon-specific outbreak risk

The benchmark may define risk at a specific horizon \(H\), meaning the risk that substantial spread occurs by that horizon under the chosen model assumptions.

Short horizons are more relevant to immediate surveillance and triage. Longer horizons are more relevant to scenario-oriented preparedness or structural sensitivity analysis.

#### 5.4.4 Transition risk

Transition risk refers to the possibility that a contact-structural regime moves from relatively low epidemic potential to substantially higher epidemic potential across windows or over time.

This is especially relevant to structural monitoring and may capture regime shifts even before large observed burden appears.

#### 5.4.5 Transmission-opportunity risk

Some benchmark quantities are best interpreted not as direct disease burden, but as **transmission-opportunity proxies**. These reflect whether the observed contact structure is permissive of spreading under the simulator, regardless of whether such spreading is realized in field data.

#### 5.4.6 Burden distinctions

The project must distinguish clearly among the following public-health quantities:

- transmission opportunity,
- infections,
- detected cases,
- symptomatic burden,
- severe disease burden,
- hospitalization burden,
- mortality burden,
- public-health operational burden.

The present benchmark primarily approximates a simulator-defined structural epidemic-risk object. It does **not** automatically estimate hospitalization burden, mortality burden, or operational healthcare demand unless explicitly extended.

#### 5.4.7 Additional quantities if activated

If the design later includes them, the following must be explicitly distinguished:

- peak timing,
- peak prevalence,
- outbreak duration,
- persistence of elevated risk,
- resource-sensitive burden proxies.

No such quantity may be implied without explicit definition.

---

### 5.5 Epidemic model classes in substantive language

This subsection provides the public-health meaning of the epidemic model families referenced elsewhere.

#### 5.5.1 SIR in public-health terms

SIR represents diseases or outbreak settings where:
- susceptible individuals become infected,
- infected individuals recover,
- recovered individuals are no longer susceptible over the modeled horizon.

Its substantive interpretation is appropriate when immunity, removal, or effective non-return to susceptibility matters over the analysis window.

This model family supports public-health interpretation of:
- finite outbreak waves,
- attack rate,
- large-outbreak probability,
- outbreak termination.

#### 5.5.2 SIS in public-health terms

SIS represents diseases or settings where:
- susceptible individuals become infected,
- infected individuals recover,
- recovered individuals return to susceptibility.

Its substantive interpretation is appropriate when reinfection or non-durable protection matters.

This model family supports interpretation of:
- endemic persistence,
- repeated local flare-ups,
- long-run prevalence risk,
- recurrence.

#### 5.5.3 SEIR and SEIS in public-health terms

SEIR and SEIS add an exposed or latent stage between susceptibility and infectiousness. These are appropriate when:
- infection does not immediately imply infectiousness,
- symptom onset is delayed,
- incubation matters,
- transmission may occur after or before symptoms depending on parameterization.

This class is important in public-health terms because observed cases often lag the true infection process.

#### 5.5.4 Awareness-coupled or behavior-coupled epidemic systems

Behavior-coupled epidemic models incorporate the idea that epidemic conditions affect behavior, and behavior in turn affects spread.

Public-healthly meaningful mechanisms include:
- increased self-protection under high risk,
- reduced contact under awareness,
- adherence fatigue,
- imitation of preventive practices,
- awareness campaigns or communication effects.

The benchmark must never invoke such models without clarifying whether behavior acts through:
- reduced contact,
- reduced susceptibility,
- reduced transmissibility,
- altered recovery or care-seeking,
- or another explicit pathway.

#### 5.5.5 Intervention-sensitive interpretations

Different epidemic model classes are more or less appropriate depending on the disease and intervention setting:

- SIR-like: acute immunizing outbreaks, many classic respiratory-wave interpretations.
- SIS-like: recurrent or endemic settings.
- SEIR/SEIS-like: latency-sensitive diseases, delayed symptom or detection contexts.
- awareness-coupled / behavior-coupled: risk-responsive populations or communication-sensitive outbreaks.

#### 5.5.6 Disease-theory selection rule

The project must always justify epidemic model choice in substantive disease terms, not only in mathematical or coding convenience terms.

---

### 5.6 Contact networks as transmission substrates

This subsection translates the network object into public-health terms.

#### 5.6.1 Contact is not transmission

The observed contact network is **not** the transmission chain. It is a representation of potential exposure opportunities.

This distinction must appear every time the benchmark target is interpreted.

#### 5.6.2 Exposure interpretation

The project must state what features of contact are treated as epidemiologically relevant:
- existence of contact,
- duration of contact,
- repeated contact,
- timing of contact,
- clustering of contact,
- overlap or concurrency.

#### 5.6.3 Equal-contact versus weighted-contact interpretation

If all contacts are treated equally, that is a strong assumption and must be stated clearly.

If contacts are weighted, the section must specify what the weight represents:
- duration,
- intensity,
- repeated exposure,
- or another proxy.

#### 5.6.4 Contact context distinctions

The benchmark should state whether it can distinguish or ignores:
- household versus nonhousehold contact,
- workplace versus community contact,
- school versus general contact,
- healthcare exposure versus routine exposure,
- close contact versus casual proximity.

If the dataset does not encode such distinctions, the section must say so plainly.

#### 5.6.5 Latent modifiers omitted by the contact graph

The benchmark generally does not directly observe:
- pathogen dose,
- biological susceptibility,
- prior immunity,
- ventilation or environment,
- symptom status,
- mask use,
- testing behavior,
- healthcare seeking,
- treatment availability.

These omitted variables limit how strongly contact structure can be interpreted as true transmission opportunity.

#### 5.6.6 Transmission-substrate adequacy

The section must state the project’s best public-health interpretation of the contact data:
- a plausible transmission substrate,
- a partial exposure proxy,
- or a weak structural indicator.

The strength of this claim must match the dataset and context.

#### 5.6.7 Public-health language for this benchmark

The preferred public-health phrasing is that the benchmark uses **contact-structured epidemic-risk proxies** or **contact-structured transmission-opportunity risk functionals**, not direct measures of field transmission or disease burden.

---

### 5.7 Surveillance role and operational timeline

This subsection freezes the first plausible public-health user and action interface for the benchmark.

#### 5.7.1 Primary intended expert user

The primary intended expert user is:
- a surveillance analyst,
- outbreak investigation lead,
- or preparedness-oriented public-health analyst,

not a clinician, member of the public, or automated operational decision system.

#### 5.7.2 Primary plausible use

The first plausible use-case supported by this benchmark is:
**expert-facing structural risk prioritization of windows, settings, or periods for closer monitoring or investigation**.

This means the benchmark output may support questions such as:
- which windows appear structurally more permissive of spread under the simulator?
- which periods merit closer epidemiological review?
- which settings may deserve heightened structural surveillance?

#### 5.7.3 Uses not yet supported

The benchmark does **not** yet support, absent external validation:

- direct intervention triggering,
- hospital resource activation,
- public-facing warning issuance,
- school/workplace closure decisions,
- patient-level risk communication,
- autonomous policy action.

#### 5.7.4 Operational horizon

The official benchmark is interpreted as a **short-horizon structural risk signal** unless amended. If the horizon is changed, the corresponding operational interpretation must also be changed and refrozen.

#### 5.7.5 Target-to-action mismatch rule

Every public-health discussion of the benchmark target must explicitly state:

- the nearest plausible action it could inform,
- the nearest invalid over-interpretation,
- the validation level required before moving from one to the other.

The benchmark therefore functions, at most, as an expert-facing research-stage structural surveillance aid unless stronger evidence is produced.

#### 5.7.2 Information available at prediction time

The section must specify what information is assumed available when a forecast is made:
- contact structure only,
- contact structure plus recent contact history,
- contact structure plus simulated epidemic context,
- or another explicit information set.

#### 5.7.3 Operational horizon

The benchmark must define whether it is oriented to:
- immediate response,
- next few days,
- next week,
- or longer-term scenario analysis.

#### 5.7.4 Workflow relevance

The official benchmark is relevant only to the following research-stage expert workflow:

- structural prioritization of windows or settings for closer review by surveillance or outbreak-analysis experts.

It is not officially relevant to:
- automated operational triage,
- direct intervention triggers,
- hospital operations,
- public-facing warning systems.

This is the only workflow interpretation licensed by the frozen benchmark unless amended.

#### 5.7.5 Non-equivalence with case surveillance

The section must distinguish contact-structure-based forecasting from:
- case surveillance,
- symptom surveillance,
- hospitalization surveillance,
- genomic surveillance,
- mortality surveillance.

These are complementary but not interchangeable.

#### 5.7.6 Non-claims about deployment

Unless externally validated, the project must state that it is not yet a field-deployed surveillance tool.

---

### 5.8 Intervention taxonomy in public-health terms

This subsection defines the intervention landscape that contextualizes the benchmark.

#### 5.8.1 Pharmaceutical interventions

These include:
- vaccination,
- prophylaxis,
- treatment,
- post-exposure prophylaxis.

Unless explicitly modeled, the benchmark does not estimate their causal effects.

#### 5.8.2 Nonpharmaceutical interventions

These include:
- distancing,
- masking,
- isolation,
- quarantine,
- venue closure,
- mobility restriction,
- contact reduction.

The benchmark may reflect some of these structurally if they change the contact network or effective transmission substrate.

#### 5.8.3 Information and awareness interventions

These include:
- alerting,
- public messaging,
- risk communication,
- behavior nudges,
- awareness campaigns,
- social-media-mediated information diffusion.

These are epidemiologically relevant when they alter contact behavior, protective behavior, or care-seeking.

#### 5.8.4 Resource and care interventions

These include:
- treatment access,
- medical resource allocation,
- isolation support,
- healthcare capacity,
- detection infrastructure.

These may alter recovery, diagnosis, behavior, or effective spread and must not be casually collapsed into generic “behavior.”

#### 5.8.5 Contact-topology versus transmission-parameter interventions

The project must distinguish interventions acting through:
- contact topology,
- transmission probability,
- susceptibility,
- recovery,
- or awareness persistence.

This distinction is essential for honest public-health interpretation.

#### 5.8.6 Benchmark status relative to interventions

The current benchmark may encode intervention-like effects only insofar as they are represented through the simulator assumptions or structural-contact changes. It does not, by default, evaluate real intervention efficacy.

---

### 5.9 Behavioral epidemiology and public-health semantics

This subsection translates structural and topological language into behaviorally meaningful public-health language.

#### 5.9.1 General principle

Behavior matters epidemiologically only insofar as it changes:
- exposure opportunity,
- susceptibility,
- transmissibility,
- recovery,
- detection,
- or care-seeking.

#### 5.9.2 Behavior classes relevant to the benchmark

Potentially relevant behavior classes include:
- distancing,
- repeated-contact clustering into stable groups,
- contact substitution,
- selective avoidance,
- sustained self-protection,
- loss of vigilance,
- mobility adaptation,
- communication-responsive behavior.

#### 5.9.3 Structural interpretation of behavior

The section should translate structural changes into public-healthly meaningful categories:

- bridge destruction: reduced mixing between otherwise connected subgroups;
- bridge formation: new cross-group exposure opportunity;
- stable dense closure: stronger local clustering, potentially household or workgroup concentration;
- reconfiguration of contact pathways: behaviorally or institutionally altered mixing;
- repeated structural persistence: sustained exposure patterns or stable social grouping.

#### 5.9.4 Awareness and information pathways

Behavior may be driven by:
- direct observation of local illness,
- public information,
- peer information,
- rumors or distorted beliefs,
- general risk communication,
- accumulated alertness or concern.

The benchmark must not overstate which of these pathways it captures.

#### 5.9.5 Decision-theoretic versus descriptive behavior

The project must state whether it assumes:
- threshold-triggered behavior,
- continuous awareness response,
- utility-driven self-protection,
- imitation or social learning,
- or no explicit behavior microfoundation.

#### 5.9.6 Behavioral processes not modeled

If the benchmark does not directly model:
- fatigue,
- misinformation,
- healthcare provider recommendation,
- social norms,
- treatment-seeking,
- or subgroup-specific behavior,
then that omission must be made explicit.

---

### 5.10 Meaning of the benchmark target in public-health language

#### 5.10.1 Plain-language definition

The primary target is a simulator-derived epidemic-risk quantity conditional on observed contact structure under a declared epidemic model.

#### 5.10.2 Public-health phrasing

The preferred phrasing is one of:

- model-based structural outbreak-risk proxy,
- contact-structured epidemic-risk indicator,
- simulator-defined transmission-opportunity risk functional.

#### 5.10.3 Not an observed field probability

This is not a directly measured real-world probability of outbreak.

#### 5.10.4 Not a causal policy effect

This is not an intervention effect estimate.

#### 5.10.5 Not automatically a severe-outcome forecast

Unless explicitly extended, the target is not a hospitalization or mortality forecast.

#### 5.10.6 Interpretation ladder

The benchmark target should be interpreted as:
1. a research quantity,
2. potentially a structural surveillance proxy,
3. not yet an externally validated operational forecast.

---

### 5.11 What may and may not be claimed

#### 5.11.1 Permitted claims

The project may claim:
- comparative predictive performance within the benchmark,
- structural signal under the modeled assumptions,
- robustness or fragility to simulator and structural assumptions,
- possible public-health relevance of certain structural regimes.

#### 5.11.2 Forbidden claims

The project may not claim:
- true field outbreak probability,
- real intervention efficacy,
- causal effect of topology,
- direct operational readiness,
- clinically validated surveillance utility,
- superiority in real populations absent external validation.

#### 5.11.3 Evidence escalation rule

Stronger applied claims require stronger evidence:
- simulator plausibility,
- external dataset transfer,
- epidemiological validation,
- workflow validation,
- and eventually prospective operational testing.

---

### 5.12 Calibration, uncertainty, and action in public-health terms

#### 5.12.1 Public-health meaning of calibration

Calibration matters because poorly calibrated risk can:
- falsely reassure,
- induce unnecessary alarm,
- misallocate scarce resources,
- distort surveillance prioritization,
- reduce trust in analytic tools.

#### 5.12.2 Public-health meaning of uncertainty

Uncertainty affects:
- how strongly a forecast should influence action,
- what contingency planning is justified,
- how much confidence can be placed in prioritization decisions.

#### 5.12.3 Threshold-based decision risk

If a forecast is used near action thresholds, miscalibration is particularly dangerous. This must be stated explicitly.

#### 5.12.4 Interval communication

Uncertainty intervals or risk bands must be explained in public-health language rather than only statistical language.

#### 5.12.5 Recalibration and updating

If recalibration or model updating is performed, it should be interpreted as part of responsible public-health use, not merely as a technical improvement.

---

### 5.13 Burden types, decision contexts, and mismatch risks

#### 5.13.1 Burden taxonomy

The section must distinguish:
- transmission-opportunity burden,
- infection burden,
- detected case burden,
- symptomatic burden,
- severe burden,
- healthcare-demand burden,
- operational public-health burden.

#### 5.13.2 Current benchmark location

The current benchmark primarily targets a **contact-structured epidemic-risk object**. It sits much closer to transmission-opportunity risk than to hospital-demand or mortality burden.

#### 5.13.3 Decision-context mismatch

A model that predicts simulator-defined outbreak risk well may still be poorly matched to:
- ICU planning,
- mortality mitigation,
- school closure policy,
- staffing forecasts,
- treatment allocation.

This mismatch must be stated explicitly.

#### 5.13.4 Operational non-equivalence

Good structural risk prediction does not automatically imply good healthcare-demand forecasting.

---

### 5.14 Domain validity threats

This subsection enumerates the major threats to public-health validity.

#### 5.14.1 Contact-data representativeness

Observed contacts may not represent the broader population or true exposure structure.

#### 5.14.2 Reporting and measurement bias

Some contact types may be over-recorded or under-recorded.

#### 5.14.3 Latent infection states

True infection states may be unobserved, delayed, asymptomatic, or partially measured.

#### 5.14.4 Behavioral adaptation during outbreaks

People change behavior in response to epidemic conditions in ways the benchmark may only partially capture.

#### 5.14.5 Intervention drift

Real policies and social practices change over time, potentially invalidating static interpretations.

#### 5.14.6 Care-seeking and testing distortion

Detected burden is not identical to true burden.

#### 5.14.7 Pathogen and setting heterogeneity

Public-health meaning may vary by disease, institution, region, and community.

#### 5.14.8 Contact-versus-transmission mismatch

Observed contact is not identical to infectious exposure.

#### 5.14.9 Simulator misspecification

The epidemic simulator may embody wrong biological or behavioral assumptions.

---

### 5.15 Public-health sensitivity analyses in domain terms

The broader robustness program must be translated into domain terms.

#### 5.15.1 Higher transmissibility

Interpret as a more contagious or more aggressive spread scenario.

#### 5.15.2 Lower transmissibility

Interpret as a milder spread scenario.

#### 5.15.3 Faster recovery

Interpret as shorter infectiousness or more rapid control.

#### 5.15.4 Slower recovery

Interpret as prolonged infectiousness and greater opportunity for spread.

#### 5.15.5 Immunity loss or reinfection

Interpret as recurrence-prone or endemic settings.

#### 5.15.6 Longer or shorter horizon

Interpret as immediate operational awareness versus longer-term preparedness.

#### 5.15.7 Alternate outbreak thresholds

Interpret as different public-health definitions of what counts as a consequential outbreak.

#### 5.15.8 Behavioral-response sensitivity

If behavior is modeled, interpret altered alertness, distancing, or rewiring sensitivity in substantive public-health terms.

#### 5.15.9 Resource and care sensitivity

If resource-sensitive assumptions are modeled, interpret them as changes in care access, recovery, or response capacity.

---

### 5.16 Surveillance, validation, and external credibility ladder

This subsection defines the evidentiary ladder for public-health interpretation.

#### 5.16.1 Level 1: internal benchmark validity

The model performs well on the frozen benchmark under the declared assumptions.

#### 5.16.2 Level 2: simulator validity

The simulator and calibration regime are judged epidemiologically plausible.

#### 5.16.3 Level 3: structural plausibility

The contact-network substrate and structural-signal interpretations are epidemiologically credible.

#### 5.16.4 Level 4: external transfer

The approach generalizes to additional contact datasets or settings.

#### 5.16.5 Level 5: epidemiological validation

Outputs relate to real outbreak, case, or burden observations in some justified way.

#### 5.16.6 Level 6: workflow validation

The signal improves a genuine surveillance or preparedness workflow.

#### 5.16.7 Level 7: deployment readiness

Only after repeated external and operational validation.

#### 5.16.8 Claim ladder

Every stronger public-health claim must be tied to the minimum required validation level.

---

### 5.17 Equity, heterogeneity, and subgroup considerations

#### 5.17.1 Heterogeneous exposure

Different groups may have very different contact opportunities and structural risk.

#### 5.17.2 Heterogeneous vulnerability

Exposure risk and severe-outcome risk are not equivalent across populations.

#### 5.17.3 Heterogeneous detectability

Different groups may be observed, tested, or represented differently in the data.

#### 5.17.4 Structural-risk inequity

Apparent structural risk may reflect institutional or social inequities, not merely neutral network structure.

#### 5.17.5 Benchmark responsibility

The official benchmark does **not** include a dedicated subgroup-fairness analysis in its core form.

This is a declared limitation.

Accordingly, no claim of subgroup validity, equitable performance, or fairness-aware deployment is licensed by the official benchmark.

#### 5.17.6 Fairness boundary

The project is not a fairness or equity study in its official frozen form. Any future fairness extension requires:
- explicit subgroup definitions,
- explicit performance stratification,
- explicit data-representation analysis,
- and explicit amendment of this section.

---

### 5.18 Risk communication and interpretability for public-health users

#### 5.18.1 Intended output form

The benchmark output must be described as one of:
- research-only latent index,
- probabilistic structural risk score,
- alert band,
- scenario-dependent risk signal.

The final design must freeze the intended interpretation.

#### 5.18.2 User classes

Potential user classes include:
- surveillance analysts,
- outbreak investigators,
- public-health planners,
- researchers.

The section must state whether the output is suitable only for expert interpretation.

#### 5.18.3 Communication cautions

The project must state what the output should **not** be communicated as:
- a field outbreak certainty,
- a causal policy recommendation,
- a clinical risk score,
- a severity forecast unless validated.

#### 5.18.4 Decomposition and interpretability

The official public-health interpretability stance is the following:

- the benchmark output is primarily a scalar structural epidemic-risk score;
- secondary interpretation may refer to bridge-related, closure-related, persistence-related, or fragmentation-related structural signals only when those signals are explicitly computed and reported in the corresponding analysis;
- no implicit decomposition may be claimed from the scalar score alone.

This freezes interpretability as:
- scalar first,
- structural decomposition only when explicitly produced,
- no hand-waved explanation language.

#### 5.18.5 Public dissemination boundary

Unless validated and communication-tested, the output should not be treated as ready for direct public dissemination.

---

### 5.19 Ethical and translational caution

#### 5.19.1 Benchmark versus deployment

Success in this benchmark does not imply safe or responsible deployment in public-health decision systems.

#### 5.19.2 Overstatement prohibition

The project is prohibited from claiming operational readiness absent external and workflow validation.

#### 5.19.3 Ethical misuse risks

Potential misuse includes:
- overinterpreting structural risk as causal blame,
- acting on poorly calibrated scores in high-stakes settings,
- ignoring subgroup inequities,
- treating model-based proxies as observed disease burden.

#### 5.19.4 Responsible next-step principle

Any translational step must involve:
- epidemiological validation,
- stakeholder review,
- context-specific assessment,
- calibration review,
- and communication review.

---

### 5.20 What a public-health expert should be able to conclude from this section

After reading Section V alone, a public-health or infectious-disease expert should be able to answer:

1. What public-health problem class this benchmark belongs to.
2. Which disease-system classes are represented and which are not.
3. What the observed contact data are assumed to mean epidemiologically.
4. What epidemic and behavioral mechanisms are in scope.
5. What kind of burden or risk the benchmark target approximates.
6. Which surveillance or preparedness use-cases it may be relevant to.
7. What calibration and uncertainty mean in operational terms.
8. What the validity threats and generalizability limits are.
9. What evidence would be needed before stronger applied claims are justified.
10. What the project may and may not claim in public-health language.

---

### 5.21 Embedded ledgers for Section V

#### 5.21.A Disease-system ledger

A frozen table mapping:
- epidemic model class,
- substantive disease assumptions,
- reinfection status,
- latency status,
- behavioral sensitivity,
- and public-health use-case.

#### 5.21.B Burden ledger

A frozen table distinguishing:
- transmission-opportunity risk,
- infection burden,
- detected burden,
- severe burden,
- healthcare-demand burden,
- and whether the benchmark targets each of them.

#### 5.21.C Intervention ledger

A frozen table listing:
- intervention class,
- mechanism of action,
- whether it is structurally represented,
- whether it is only implicit in the simulator,
- and whether it is out of scope.

#### 5.21.D Behavior ledger

A frozen table mapping:
- modeled structural or behavioral changes,
- public-health interpretation,
- whether the interpretation is direct or indirect,
- and what remains unmodeled.

#### 5.21.E Validity-threat ledger

A frozen table listing each domain validity threat and its consequence for interpretation.

#### 5.21.F Validation ladder ledger

A frozen table mapping claim strength to required evidence level.

#### 5.21.G Communication ledger

A frozen table specifying:
- intended users,
- intended output interpretation,
- forbidden framings,
- and expert-interpretation requirements.

---

### 5.22 Section-closing principle

No public-health interpretation may exceed the level justified by:

- the contact-data substrate,
- the epidemic model assumptions,
- the simulator calibration regime,
- the benchmark evaluation evidence,
- and the external validation status.

The function of this section is not merely to warn against overclaiming. Its function is to define exactly what kind of public-health object this project is, what kind it is not, and what evidentiary path would be required to move from one to the other.

---

## VI. Literature and benchmark-positioning specification

### 6.1 Purpose of this section

This section records the literature that directly motivates, constrains, or challenges the project. Its role is not to provide a generic bibliography. Its role is to answer five concrete questions:

1. What problem class does the project actually belong to in the existing literature?
2. What tasks do prior papers study on temporal contact or temporal graph data?
3. What do those papers already do well that the present project must match or exceed?
4. What do those papers not do, or not do together, that creates room for the present project?
5. Which open questions raised in the attached conversations must be explicitly answered by the final design?

The literature section must therefore track:
- paper identity,
- contribution type,
- task formulation,
- data assumptions,
- model assumptions,
- evaluation assumptions,
- and the exact gap between that paper and the present project. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

---

### 6.2 Core literature question inherited from the attached conversation

The first attached conversation raises a central synthesis question that this project must answer explicitly:

> Given temporal graph data of contacts, what are all the tasks someone might want to do in epidemic theory, which quantities are part of the observed setup, which quantities are later model choices or hyperparameters, and what are the prediction versus prescription tasks? :contentReference[oaicite:2]{index=2}

That question is not peripheral. It is one of the main organizing questions for the literature review because it determines whether the project is framed primarily as:
- epidemic forecasting,
- structural risk prediction,
- temporal graph learning,
- disease–behavior modeling,
- intervention design,
- or some combination of these. :contentReference[oaicite:3]{index=3}

The conversation already correctly identifies one major hierarchy that the literature supports: the temporal contact data are the primary substrate, while epidemic parameters such as infection rate, recovery rate, immunity structure, awareness rate, and related simulator settings are later model choices that determine which disease process is run on that substrate. The same conversation also correctly distinguishes epidemic-theory hyperparameters from machine-learning hyperparameters. :contentReference[oaicite:4]{index=4}

---

### 6.3 Task inventory implied by the literature

Across the papers supplied so far, the literature supports a broad task inventory for temporal contact data.

#### 6.3.1 Contact-level temporal graph learning tasks

The temporal graph benchmark papers focus primarily on:
- future link prediction,
- dynamic node classification,
- dynamic node property prediction,
- and related sequence-sensitive temporal representation learning tasks.  
This is explicit in **Temporal Graph Benchmark for Machine Learning on Temporal Graphs**, which emphasizes realistic and reproducible edge-level and node-level evaluation, and in **Towards Better Dynamic Graph Learning: New Architecture and Unified Library**, which standardizes dynamic link prediction and dynamic node classification pipelines through DyGLib. :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}

The conversation summary correctly identifies these as lower-level tasks that can feed epidemic modeling but are not themselves epidemic tasks yet. :contentReference[oaicite:7]{index=7}

#### 6.3.2 Epidemic-state and epidemic-risk prediction tasks

The epidemic-theory papers imply tasks such as:
- predicting whether infection will grow or die out,
- predicting node infection state,
- estimating outbreak probability,
- estimating attack rate or final size,
- estimating threshold regimes,
- identifying containment regions under behavioral adaptation,
- and quantifying how behavior changes effective epidemic strength.  
This is especially visible in **On the existence of a threshold for preventive behavioral responses to suppress epidemic spreading**, which studies coupled SIS-style epidemic and preventive behavior and analytically derives parameter regions where infection is suppressed, and in **Coupled disease–behavior dynamics on complex networks: A review**, which explicitly frames disease and behavior as a coupled nonlinear system on networks. :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}

The conversation summary also already extracted this task distinction correctly: predicting future contacts, predicting infection states, and designing interventions are different tasks even when they use the same temporal contact substrate. :contentReference[oaicite:10]{index=10}

#### 6.3.3 Prescription and control tasks

Several papers move beyond prediction into prescription:
- vaccination decisions,
- distancing policies,
- awareness interventions,
- optimal control,
- resource allocation,
- mobility control,
- and metapopulation-level intervention design.  
This is explicit in **Optimal Control of Endemic Epidemic Diseases With Behavioral Response**, which formulates intervention design as an optimal control problem for a coupled epidemic–behavior ODE system, and in **The effect of information-driven resource allocation on the propagation of epidemic with incubation period**, which studies information-driven protection and resource allocation in a layered SEIS setting. :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}

#### 6.3.4 Coupled awareness / information / behavior tasks

A large portion of the supplied epidemic literature is not just about disease spread alone, but about coevolution among:
- epidemic state,
- awareness or information diffusion,
- behavior,
- mobility,
- resources,
- and higher-order social reinforcement.  
This is explicit in:
- **Coupled disease–behavior dynamics on complex networks: A review**,
- **Behavioural change models for infectious disease transmission: a systematic review (2010–2015)**,
- **Coupled Epidemic-Information Propagation With Stranding Mechanism on Multiplex Metapopulation Networks**,
- **Modeling coupled epidemic and awareness spreading with heterogeneous recovery and simplicial complexes**. :contentReference[oaicite:13]{index=13} :contentReference[oaicite:14]{index=14} :contentReference[oaicite:15]{index=15} :contentReference[oaicite:16]{index=16}

This matters because the present project cannot pretend that “epidemic modeling on contact graphs” means only SIR or SIS on a static network. The literature already treats coevolving behavior and information as part of the serious task landscape.

---

### 6.4 What the literature says about the setup hierarchy

One of the user’s key questions in the first conversation was whether the temporal contact data come first and the SIR/SIS parameters later. The literature supports a qualified version of that view.

#### 6.4.1 What is primary

Primary data objects include:
- nodes,
- contact events,
- timestamps,
- durations,
- repeated-contact structure,
- attributes,
- and any derived temporal graph representation.  
The temporal graph ML papers treat these as the native learning substrate. :contentReference[oaicite:17]{index=17} :contentReference[oaicite:18]{index=18}

#### 6.4.2 What is later but scientifically essential

Later epidemic-model choices include:
- epidemic model class: SIS, SIR, SEIR, SEIS, SAIS, multiplex epidemic–awareness models, metapopulation variants;
- biological parameters: infection rate, recovery rate, incubation rate, immunity loss;
- behavior parameters: alerting rate, adoption strength, awareness decay, imitation intensity, risk perception coupling;
- resource parameters: protective efficacy, allocation rules, recovery enhancement;
- control parameters: vaccination intensity, distancing level, mobility intervention.  
These are not “afterthoughts”; they are the epidemic model layer placed on top of the contact substrate. This is exactly the distinction made in the conversation synthesis and repeatedly reflected in the supplied disease–behavior papers. :contentReference[oaicite:19]{index=19} :contentReference[oaicite:20]{index=20} :contentReference[oaicite:21]{index=21}

#### 6.4.3 What the project must say clearly

The final paper must explicitly distinguish:
- observed temporal contact data,
- derived structural summaries,
- epidemic-model parameters,
- behavior parameters,
- calibration choices,
- and machine-learning hyperparameters.  
The attached conversation already warned that these are easy to conflate and that the literature treats them as different layers. :contentReference[oaicite:22]{index=22}

---

### 6.5 Literature cluster A: epidemic-behavior theory

#### 6.5.1 *On the existence of a threshold for preventive behavioral responses to suppress epidemic spreading* (Sahneh, Chowdhury, Scoglio, 2012)

This paper is foundational for the project because it gives an analytically tractable coupled epidemic–behavior model and shows that preventive behavior can create a region of parameter space where infection dies out even above the classical epidemic threshold. It explicitly categorizes behavior models into:
- change in system state,
- change in system parameters,
- change in contact topology. :contentReference[oaicite:23]{index=23}

What it has that the present project needs to respect:
- explicit threshold theory,
- explicit parameter-region analysis,
- an analytically defensible coupling of behavior and epidemic spread,
- a clear distinction among different ways behavior enters the model. :contentReference[oaicite:24]{index=24}

What it does not provide for the present project:
- modern temporal graph learning baselines,
- topology-aware machine learning,
- benchmark-style reproducible evaluation on temporal contact windows,
- a direct treatment of persistence-based topological structure.

What the present project must learn from it:
- any use of behavior must specify **how** behavior enters the epidemic system;
- the project must not use vague language like “behavior matters” without clarifying whether it changes topology, transmission rate, state transitions, or awareness. :contentReference[oaicite:25]{index=25}

#### 6.5.2 *Coupled disease–behavior dynamics on complex networks: A review* (Wang, Andrews, Wu, Wang, Bauch, 2015)

This review is one of the most important framing papers for the project. It makes clear that disease and behavior form a coupled nonlinear system and that network structure matters materially relative to homogeneous mixing. It also surveys vaccination, distancing, decision-making, and information-driven responses on complex networks. :contentReference[oaicite:26]{index=26}

What it has that the present project needs:
- domain legitimacy for coupled disease–behavior modeling on networks,
- a strong warning that well-mixed models are insufficient in many settings,
- a taxonomy of behavior-sensitive mechanisms,
- a broader systems view linking disease, behavior, and network topology. :contentReference[oaicite:27]{index=27}

What it still leaves open for the present project:
- no modern temporal graph benchmark framing,
- no topology-aware learning benchmark,
- no persistent-homology or VPD machinery,
- no unified ML/network/public-health benchmark design.

What the present project must take from it:
- the project should not frame itself as “just dynamic graph learning”;
- it belongs in the coupled disease–behavior / network epidemiology literature as much as in temporal graph ML. :contentReference[oaicite:28]{index=28}

#### 6.5.3 *Behavioural change models for infectious disease transmission: a systematic review (2010–2015)* (Verelst, Willem, Beutels, 2016)

This paper is essential because it is explicitly a review of behavior change models and it highlights a major weakness in the literature: many models are theoretically interesting but lack real data, representative validation, or robust validation processes. It classifies behavior models by how information becomes preventive action and notes that only a minority use real-life data for parametrization or validation. :contentReference[oaicite:29]{index=29}

What it has that the present project needs:
- a behavior-model taxonomy,
- an explicit validation critique,
- a warning against purely theoretical modeling without representative data,
- support for individual-level or heterogeneous models. :contentReference[oaicite:30]{index=30}

What the present project must do in response:
- state clearly whether it is predictive, descriptive, mechanistic, or prescriptive;
- acknowledge that simulator-based evidence is not field validation;
- avoid presenting a purely theoretical mechanism as though it were already epidemiologically validated. :contentReference[oaicite:31]{index=31}

---

### 6.6 Literature cluster B: temporal graph learning and benchmark design

#### 6.6.1 *Temporal Graph Benchmark for Machine Learning on Temporal Graphs* (Huang et al., 2023)

This is the benchmark paper the present project must take most seriously on the ML side. It argues that temporal graph learning needs realistic, reproducible, and robust evaluation, and it emphasizes that performance can vary drastically across datasets and that simple methods can outperform temporal graph models in some node-level tasks. :contentReference[oaicite:32]{index=32}

What it has that the present project needs:
- benchmark discipline,
- realistic and reproducible evaluation mindset,
- explicit awareness of over-optimistic evaluation,
- a benchmark-centered rather than architecture-centered perspective. :contentReference[oaicite:33]{index=33}

What it does not supply:
- epidemic-risk targets,
- public-health interpretation,
- simulator-defined outbreak-risk labels,
- topology-aware constraint learning.

What the present project must learn from it:
- if the benchmark is not brutally standardized, the empirical story will not be credible;
- one dataset and one narrow baseline family are not enough. :contentReference[oaicite:34]{index=34}

#### 6.6.2 *Towards Better Dynamic Graph Learning: New Architecture and Unified Library* (Yu et al., 2023)

This paper matters for two reasons: DyGFormer itself, and DyGLib as a unified implementation library. It argues that previous methods often failed to capture correlations or long-term temporal dependencies and that inconsistent training pipelines led to poor reproducibility. :contentReference[oaicite:35]{index=35} :contentReference[oaicite:36]{index=36}

What it has that the present project needs:
- a strong modern transformer-style baseline,
- a unified pipeline mindset,
- a direct warning that diverse implementations can produce inconsistent baseline findings. :contentReference[oaicite:37]{index=37}

What it does not do:
- epidemic theory,
- behavioral epidemiology,
- outbreak-risk labels,
- public-health interpretation,
- topology-as-loss on temporal contact windows.

What the present project must take from it:
- implementation inconsistency is not a side issue; it can invalidate conclusions;
- the project must either adopt or emulate a DyGLib-like level of implementation discipline. :contentReference[oaicite:38]{index=38}

#### 6.6.3 *On the Power of Heuristics in Temporal Graphs* (Cornell et al., 2025)

This paper is one of the strongest warnings for the benchmark. It shows that simple heuristics exploiting recency and popularity can match or outperform sophisticated neural models under standard evaluation protocols. :contentReference[oaicite:39]{index=39}

What it has that the present project needs:
- a benchmark humility principle,
- recency/popularity as serious comparator mechanisms,
- explicit motivation for strong heuristic baselines. :contentReference[oaicite:40]{index=40}

What the present project must do in response:
- include strong heuristic baselines,
- explicitly test whether topology-aware gains reduce to recency, persistence, or hub concentration,
- not assume that a neural baseline is automatically harder to beat than a heuristic one. :contentReference[oaicite:41]{index=41}

#### 6.6.4 *TGB-Seq Benchmark: Challenging Temporal GNNs With Complex Sequential Dynamics* (Yi et al., 2025)

This paper argues that many current temporal GNN benchmarks overemphasize repeated edges and underemphasize genuine sequential dynamics. It shows that current methods can degrade badly when repeated-edge shortcuts are reduced and more demanding sequential structure is required. :contentReference[oaicite:42]{index=42}

What it has that the present project needs:
- a warning that many temporal graph methods can succeed by memorization rather than real sequence learning,
- support for treating repeated-contact persistence and nontrivial sequential structure as distinct benchmark properties. :contentReference[oaicite:43]{index=43}

What it means for the present project:
- a contact-window epidemic benchmark must distinguish repeated-edge/persistence signal from richer sequential structural signal;
- if the benchmark is too recurrence-heavy, topology-aware results may be confounded by memorization-friendly structure. :contentReference[oaicite:44]{index=44}

---

### 6.7 Literature cluster C: incubation, resources, multiplexity, metapopulations, and higher-order interactions

#### 6.7.1 *The effect of information-driven resource allocation on the propagation of epidemic with incubation period* (Zhu et al., 2022)

This paper matters because it goes beyond simple SIS/SIR by explicitly modeling incubation, information diffusion, and resource allocation in a three-layer framework. It highlights exposed but asymptomatic states, information-driven resource seeking, and topology-dependent threshold behavior. :contentReference[oaicite:45]{index=45}

What it has that the present project may still be missing:
- incubation-period realism,
- exposed-state dynamics,
- resource-coupled epidemic interpretation,
- explicit multilayer coupling of epidemic, information, and resources. :contentReference[oaicite:46]{index=46}

Implication for the present project:
- if the project wants to claim modernity in epidemiological modeling, it should at least position itself relative to incubation and resource-aware epidemic frameworks, even if the official core benchmark remains SIR/SIS.

#### 6.7.2 *Coupled Epidemic-Information Propagation With Stranding Mechanism on Multiplex Metapopulation Networks* (An et al., 2024)

This paper contributes a multiplex metapopulation perspective with mobility, information propagation, and adaptive behavior. It emphasizes population-size dynamics and spatiotemporal characteristics, not just local contact edges. :contentReference[oaicite:47]{index=47}

What it has that the present project may still be missing:
- metapopulation interpretation,
- mobility-sensitive epidemic structure,
- patch-level or spatial-population framing,
- epidemic-information coupling beyond a single contact layer. :contentReference[oaicite:48]{index=48}

Implication:
- the project must be clear that it is a temporal contact-network benchmark, not yet a full metapopulation mobility benchmark.

#### 6.7.3 *Optimal Control of Endemic Epidemic Diseases With Behavioral Response* (Zino, Parino, Rizzo, 2024)

This paper is critical for the prescription side of the literature. It couples epidemic dynamics with a behavioral-response game and studies cost-effective control policies. :contentReference[oaicite:49]{index=49}

What it has that the present project may still be missing:
- an explicit control-theoretic prescription layer,
- optimization over interventions,
- cost-effectiveness framing,
- healthcare and socioeconomic control objectives. :contentReference[oaicite:50]{index=50}

Implication:
- the present project is not yet an optimal intervention design paper unless it explicitly adds a control layer;
- if it stays predictive, it should say so and not blur into prescription rhetoric.

#### 6.7.4 *Modeling coupled epidemic and awareness spreading with heterogeneous recovery and simplicial complexes* (Kan, Zhang, Wang, 2026)

This paper matters because it introduces heterogeneous alertness recovery and simplicial-complex-based awareness reinforcement. It explicitly uses simplicial complexes to capture higher-order social reinforcement and heterogeneous awareness persistence. :contentReference[oaicite:51]{index=51}

What it has that the present project may still be missing:
- a direct higher-order epidemic–awareness model,
- simplicial reinforcement as a domain object rather than only as a topological summary,
- heterogeneous awareness decay or recovery. :contentReference[oaicite:52]{index=52}

Implication:
- if the present project claims higher-order social or awareness semantics, it should position those claims against papers that model higher-order interaction structure directly rather than only summarizing pairwise contact graphs through topology.

---

### 6.8 What these papers collectively have that the present project must still include or justify

The literature taken together reveals several categories of things that the present project must either include, justify omitting, or defer honestly.

#### 6.8.1 Strong benchmark discipline

The temporal graph literature already expects:
- reproducible evaluation,
- realistic splits,
- strong baselines,
- and unified implementations.  
The project must therefore meet that standard, not merely gesture toward it. :contentReference[oaicite:53]{index=53} :contentReference[oaicite:54]{index=54}

#### 6.8.2 Strong heuristic baselines

The modern temporal graph literature already shows that simple heuristics can be highly competitive. Therefore heuristic controls are not optional extras. :contentReference[oaicite:55]{index=55}

#### 6.8.3 Explicit behavior mechanism language

The disease–behavior literature already distinguishes behavior acting through:
- state transitions,
- transmission parameters,
- contact topology,
- information response,
- imitation,
- resource allocation,
- awareness persistence.  
The project must use that level of specificity. :contentReference[oaicite:56]{index=56} :contentReference[oaicite:57]{index=57} :contentReference[oaicite:58]{index=58}

#### 6.8.4 Validation humility

The behavior-model review already warns that many models are theoretically rich but weakly validated. The present project must not repeat that failure by overstating simulator-only results. :contentReference[oaicite:59]{index=59}

#### 6.8.5 Higher-order alternatives

The newer epidemic-awareness papers already use multiplex, hypergraph, or simplicial constructions. Therefore the project cannot imply that pairwise temporal GNNs exhaust the plausible comparator space if it foregrounds higher-order semantics. :contentReference[oaicite:60]{index=60} :contentReference[oaicite:61]{index=61} :contentReference[oaicite:62]{index=62}

#### 6.8.6 Prescription is a different problem class

The optimal-control literature already shows what it means to genuinely solve a prescription problem. The present project should therefore distinguish prediction from prescription and not promise more than it currently does. :contentReference[oaicite:63]{index=63}

---

### 6.9 What the present project appears to contribute that these papers do not combine

The literature supplied so far suggests that the present project’s most plausible niche is not any one of the following alone:
- not just a temporal graph benchmark,
- not just a disease–behavior model,
- not just a network-science paper,
- not just a public-health warning-score paper,
- and not just a TDA paper.

Its plausible contribution is instead the conjunction of all of these:

1. a chronology-respecting temporal contact benchmark with simulator-defined epidemic-risk targets;
2. a topology-as-loss framework built on VPD/Wasserstein/RKHS machinery rather than topology-as-input;
3. explicit comparison against serious graph-statistic, spectral, heuristic, and temporal-graph baselines;
4. a network-science and public-health interpretation layer rather than a purely methodological story. :contentReference[oaicite:64]{index=64} :contentReference[oaicite:65]{index=65} :contentReference[oaicite:66]{index=66} :contentReference[oaicite:67]{index=67}

What remains essential is to show that this conjunction is nonredundant and not just a stack of disconnected ingredients.

---

### 6.10 Literature questions the final paper must answer explicitly

The attached conversation raised questions that the final literature section must answer directly, not leave implicit.

#### 6.10.1 What are all the tasks one might want to do with temporal contact data?

The literature says the answer includes at least:
- future contact prediction,
- node infection-state prediction,
- outbreak-risk prediction,
- attack-rate estimation,
- threshold estimation,
- intervention comparison,
- awareness modeling,
- resource allocation,
- mobility-aware spread modeling,
- optimal control,
- and higher-order interaction analysis. :contentReference[oaicite:68]{index=68} :contentReference[oaicite:69]{index=69} :contentReference[oaicite:70]{index=70} :contentReference[oaicite:71]{index=71}

#### 6.10.2 What comes first: data or epidemic parameters?

The literature-supported answer is:
- contact data and temporal structure come first as the substrate;
- epidemic and behavior parameters come next as model assumptions on that substrate;
- machine-learning hyperparameters are yet another separate layer. :contentReference[oaicite:72]{index=72}

#### 6.10.3 Are we modeling one task or many?

The literature suggests that many adjacent tasks exist, but the present project must freeze one primary task. The design so far points to:
- primary task: simulator-defined outbreak-risk prediction on temporal contact windows;
- secondary tasks: structural comparison, robustness, and scientific interpretation;
- non-primary tasks: direct optimal control, real-world intervention efficacy, and clinical burden forecasting. :contentReference[oaicite:73]{index=73} :contentReference[oaicite:74]{index=74}

#### 6.10.4 What are all the hyperparameters?

The literature and conversation together imply three distinct classes:
- epidemic-process hyperparameters: \(\beta,\gamma,\tau\), horizon, seeding, immunity, incubation, awareness coupling, resource efficacy;
- behavior hyperparameters: alerting rate, awareness decay, imitation or response intensity, topology-change strength;
- machine-learning hyperparameters: hidden dimension, depth, learning rate, history horizon, topology-loss weight, random-feature dimension, batch size. :contentReference[oaicite:75]{index=75} :contentReference[oaicite:76]{index=76} :contentReference[oaicite:77]{index=77}

The final paper must not blur these into one undifferentiated set of “hyperparameters.”

---

### 6.11 Literature-grounded deficits the project must still avoid

This section exists to make explicit what the literature warns against.

The project must avoid:
- benchmark fragility through narrow baselines,
- implementation inconsistency,
- overreliance on recurrence-heavy datasets,
- behavior language without mechanism,
- simulator-only overclaiming,
- higher-order claims without higher-order comparators,
- and slipping from prediction rhetoric into prescription rhetoric without adding control machinery. :contentReference[oaicite:78]{index=78} :contentReference[oaicite:79]{index=79} :contentReference[oaicite:80]{index=80} :contentReference[oaicite:81]{index=81} :contentReference[oaicite:82]{index=82} :contentReference[oaicite:83]{index=83}

---

### 6.12 Section-closing principle

The purpose of this literature section is not to show that the project is “related” to many papers. Its purpose is to show that the project sits at the intersection of:
- temporal graph benchmark design,
- network epidemic theory,
- disease–behavior dynamics,
- higher-order interaction modeling,
- and public-health interpretation.

The final paper must therefore demonstrate four things simultaneously:

1. it understands what the temporal graph literature already demands;
2. it understands what the epidemic-behavior literature already models;
3. it understands what those literatures still do not combine;
4. and it can state exactly what the present project adds without exaggeration. :contentReference[oaicite:84]{index=84} :contentReference[oaicite:85]{index=85} :contentReference[oaicite:86]{index=86}

# VI. Corrected Benchmark Architecture and Experimental Maturity Upgrade
## (Intervention-First, SAIS-Centered, Multi-Tier Dataset Design)

### 6.1 Scientific correction to benchmark objective

The benchmark is no longer centered on outbreak-risk prediction as the primary scientific endpoint.

Outbreak-risk prediction is formally reclassified as an **intermediate predictive task layer**.

The primary scientific objective is now:

**to estimate, compare, and improve intervention-relevant epidemic quantities on temporal contact networks under behavior-aware epidemic dynamics.**

This includes:

- epidemic threshold estimation
- containment-region estimation
- intervention sufficiency
- awareness-rate sufficiency
- vaccination allocation sensitivity
- regime-transition forecasting
- behavior-induced suppression dynamics

This correction aligns the benchmark with modern epidemic-theory maturity.

---

### 6.2 Primary epidemic model class (official correction)

The official primary epidemic model is:

**SAIS (Susceptible–Alert–Infected–Susceptible)**

This replaces SIS/SIR as the principal scientific model family.

SIS and SIR are retained only as lower-order comparators.

The reason is scientific maturity:

SAIS explicitly models intervention-relevant behavioral adaptation through alertness dynamics.

This allows direct estimation of:

- alerting threshold
- awareness sufficiency
- behavioral containment
- intervention response sensitivity
- threshold lifting under public-health action

These are closer to gold-standard epidemic tasks than plain state forecasting.

Official parameter family:

- transmission rate: \(\beta\)
- recovery rate: \(\delta\)
- alerting rate: \(\kappa\)
- alert-state transmission: \(\beta_A\)

Primary derived quantities:

- \(\tau_{c1}\)
- \(\tau_{c2}\)
- \(\kappa_c\)

These become official benchmark targets.

---

### 6.3 Official task hierarchy (fully explicit)

#### Tier I — temporal graph learning support tasks

These are machine-learning support tasks.

They are NOT the main scientific endpoint.

##### Task I.A — dynamic link prediction

Predict future contacts:

\[
(i,j,t+\Delta t)
\]

This evaluates temporal structural learning.

##### Task I.B — node affinity prediction

Predict future contact distribution:

\[
P(j \mid i, t:t+k)
\]

This measures future transmission opportunity structure.

##### Task I.C — sequential transmission-chain forecasting

Predict novel contact chains and new pathways.

This specifically tests:

- high-surprise regimes
- novel-contact emergence
- superspreader chain formation

---

#### Tier II — intermediate epidemic prediction tasks

These are secondary scientific tasks.

##### Task II.A — infection state prediction

Predict:

- S
- A
- I

node state over future horizon.

##### Task II.B — outbreak-risk prediction

Simulator-derived probability:

\[
P(\text{large outbreak} \mid G_{1:t})
\]

This remains in the benchmark but is explicitly secondary.

##### Task II.C — final size / attack rate

Predict:

\[
\mathbb{E}[Z_T \mid G_{1:t}]
\]

where \(Z_T\) is final epidemic size.

---

#### Tier III — primary intervention tasks

This is the primary benchmark layer.

##### Task III.A — epidemic threshold estimation

Estimate:

\[
R_0,\quad \lambda_1,\quad \tau_c
\]

including topology-sensitive threshold lifting.

---

##### Task III.B — cushion-region estimation

Estimate:

\[
[\tau_{c1}, \tau_{c2}]
\]

This is the primary containment task.

---

##### Task III.C — minimum awareness sufficiency

Estimate:

\[
\kappa_c
\]

This becomes a primary benchmark endpoint.

---

##### Task III.D — intervention simulation

Explicit “what-if” policy tasks:

- mask uptake
- awareness campaign
- distancing
- targeted isolation
- targeted vaccination
- hub vaccination
- bridge vaccination

This is the gold-standard task layer.

---

### 6.4 Official dataset hierarchy (fully explicit and selected)

The benchmark is now explicitly split.

---

## Tier A — primary epidemiological contact datasets

These are official core datasets.

### A1. SocioPatterns hospital ward

Primary intervention dataset.

Reason:
- direct infection pathway relevance
- HCW-patient structure
- ideal for intervention analysis

:contentReference[oaicite:1]{index=1}

---

### A2. SocioPatterns primary school

Primary transmission-cluster dataset.

Reason:
- dense community structure
- class closure interventions
- ideal for SAIS threshold study

:contentReference[oaicite:2]{index=2}

---

### A3. SocioPatterns high school

Primary adolescent transmission / community bridging dataset.

Reason:
- repeated contacts
- class clusters
- friendship overlay

:contentReference[oaicite:3]{index=3}

---

### A4. SocioPatterns workplace

Primary adult repeated-contact / intervention dataset.

Reason:
- repeated contacts
- community linkers
- hub-target vaccination task

:contentReference[oaicite:4]{index=4}

---

### A5. SocioPatterns conference / Hypertext / SFHH

Primary burst-event and superspreader dataset.

Reason:
- short horizon
- bursty transient contact regime
- high novelty

:contentReference[oaicite:5]{index=5}

---

### A6. Flights

Primary long-range epidemic transport substrate.

Used for spatial transmission sensitivity.

---

## Tier B — maturity validation datasets

Used for temporal graph ML maturity.

- Wikipedia
- Reddit
- MOOC
- LastFM
- Enron
- UCI

These are NOT primary epidemiological datasets.

They exist for model maturity stress testing.

---

### 6.5 Topological task integration (corrected)

Topology-aware loss is officially evaluated only on tasks where it has direct semantic meaning.

Primary topology-sensitive tasks:

- threshold estimation
- bridge-risk detection
- community leakage
- superspreader burst transitions
- intervention sensitivity
- containment boundary estimation

This is where topology is strongest.

It is NOT treated as universally useful across all tasks.

---

### 6.6 Official difficulty strata

All results must be stratified.

#### Regime A — repeated contact

Memorization-heavy.

#### Regime B — novel contact

High surprise.

#### Regime C — burst / superspreader

Short-time structural shock.

#### Regime D — intervention-sensitive

Behavioral threshold regime.

#### Regime E — bridge-critical

Community leakage / bridge formation.

---

### 6.7 Primary scientific claim (fully corrected)

The paper’s main claim is now:

**topology-aware temporal graph learning improves estimation of intervention-relevant epidemic thresholds and containment quantities under behavior-aware epidemic dynamics on high-resolution contact networks.**

This is substantially more mature than plain outbreak-risk prediction.

## VI. Cross-domain integration

### 6.1 End-to-end object flow

The project must define the full flow of objects:
observed contact events to temporal windows, temporal windows to topological constructions, topological constructions to virtual persistence diagram objects, virtual persistence diagrams to exact or approximate feature representations, feature representations to predictive models, predictive models to probabilistic outputs, and probabilistic outputs to evaluation and interpretation.

### 6.2 Mapping exact mathematical objects to code artifacts

Every exact mathematical object used by the project must have a corresponding implementation-level representation. This subsection defines that mapping and marks where approximation enters.

### 6.3 Mapping estimands to outputs and metrics

Every reported number in the final project must map back to a formally defined estimand, prediction target, or evaluation functional. This section states that mapping.

### 6.4 Mapping model components to scientific roles

Each architectural component, simulator component, or evaluation component must be tied to the scientific role it plays. No component should exist in the project without a declared scientific purpose.

---

## VII. Experimental design specification

### 7.1 Official dataset scope

The official benchmark dataset set is:
\[
\mathcal D_{\mathrm{official}}=\{D_1,\dots,D_K\},
\]
where every included dataset must satisfy the following inclusion criteria:

- timestamped contact data of sufficient granularity for temporal-window construction;
- stable node-identifier normalization;
- enough windows after preprocessing to support chronology-respecting train/validation/test splits;
- no unresolved provenance or licensing issues;
- no unresolved logical failures in simulator label generation.

Any dataset failing these conditions is excluded from the official benchmark and may appear only in exploratory analysis.

### 7.2 Chronological split design

For each official dataset, the split is frozen as:
- earliest \(60\%\) of windows: training,
- next \(20\%\): validation,
- final \(20\%\): test,

unless a dataset-specific amendment documents a different chronology-respecting split forced by support constraints.

No random shuffling is permitted.

### 7.3 Official benchmark tiers

The benchmark tiers are frozen as:

- **smoke tier**: minimal run, small simulation count, correctness/testing only;
- **pilot tier**: reduced search and reduced simulation count for design iteration;
- **official tier**: frozen benchmark configuration used for reported internal results;
- **submission tier**: official tier plus all required sensitivity, null-model, and regime analyses.

### 7.4 Official ablation matrix

The official ablation matrix is exactly the one defined in Section III and no larger unless amended.

### 7.5 Official sensitivity matrix

The official sensitivity axes are:

- epidemic model class: \(\{\mathrm{SIR},\mathrm{SIS}\}\),
- transmission and recovery parameters: frozen default \((\beta_0,\gamma_0)\) plus explicit alternate grid,
- outbreak threshold \(\tau\): frozen default plus explicit alternate grid,
- horizon length \(H\): frozen default plus explicit alternate values,
- window policy: frozen default plus explicit alternate policies,
- topology-loss family: Wasserstein versus RKHS,
- structural-depth activation: depth \(0\) versus depth \(1\) if the recursive branch is activated,
- feature-approximation dimension: \(m\in\{256,512,1024\}\),
- chronology split variation: exactly one alternate chronology-respecting split rule defined in advance.

Each axis has:
- a frozen default,
- a frozen alternate set,
- and a frozen reporting status.

The phrase “where feasible” is forbidden in the official sensitivity specification.

### 7.6 Official reporting standard

Every official reported result must include:
- point estimate,
- uncertainty summary,
- number of seeds,
- number of simulation draws for labels,
- comparator family,
- metric definition,
- whether the result is primary, secondary, or exploratory.

### 7.7 Reproducibility standard

Every official experiment must record:
- dataset version,
- simulator version,
- calibration version,
- topology version,
- code version,
- seed set,
- hardware tier.

No result lacking this provenance may enter the final benchmark tables.

### 7.4 Ablation matrix

The official ablation matrix is frozen as follows:

1. **A0: no-topology control**
   - encoder + prediction head only;

2. **A1: Wasserstein-topology constraint**
   - encoder + prediction head + Wasserstein topological loss;

3. **A2: RKHS-topology constraint**
   - encoder + prediction head + RKHS/heat-kernel topological loss;

4. **A3: higher-depth topological constraint**
   - encoder + prediction head + recursive higher-depth penalty, only if this branch is activated in the official benchmark;

5. **A4: non-topological structural-summary controls**
   - tabular or spectral baselines defined in Section III;

6. **A5: epidemic-model sensitivity ablations**
   - SIR/SIS/other frozen simulator families;

7. **A6: structural-summary reduction ablations**
   - degree-only, spectral-only, clustering-only, bridge/community-only comparator families;

8. **A7: temporal-order and structural-depth ablations**
   - first-order versus higher-order admissible topological comparison families, where activated.

The following are **not official ablation arms** unless introduced by amendment:

- topology as input,
- topology as fused input-plus-constraint,
- topology as encoder-native feature stream.

This subsection must agree with Section III’s freeze that topology is active only as a training-time constraint in the main benchmark.

### 7.5 Sensitivity matrix

This subsection defines the exact grid of sensitivity analyses across epidemic assumptions, network assumptions, feature approximations, and optimization randomness.

### 7.6 Statistical reporting standard

The project must specify how metrics are summarized, what intervals are reported, how pairwise model comparisons are performed, what effect-size language is used, and what plotting standards are allowed.

### 7.7 Reproducibility standard

This subsection must specify seeds, rerun counts, nondeterministic components, package version capture, and the criteria by which an experiment is considered reproducible.

---

## VIII. Amendment history and residual unresolved issues

The benchmark is assumed to be moving toward freeze. Therefore this section no longer functions as a generic placeholder for future decisions.

### 8.1 Amendment history

Every substantive change after the first frozen draft must be logged here with:

- amendment identifier,
- date,
- affected sections,
- exact change,
- rationale,
- whether previously reported results are invalidated.

### 8.2 Residual unresolved issues

Any issue still listed here is an explicit blocker to full freeze.

For each unresolved issue, the document must state:
- issue identifier,
- affected section,
- allowed options,
- decision owner,
- evidence required,
- deadline for resolution.

### 8.3 Freeze rule

If any issue remains unresolved in this section, the document is not fully frozen and no final claims section may be treated as final.

---

## IX. Final frozen claims

This section must be completed before submission-grade freeze. No claim may appear in the paper unless it is licensed here.

### 9.1 Mathematical claims

The project may claim only the following mathematical categories:

- exact construction of the virtual persistence diagram group state space and its lifted \(W_1\) metric;
- exact finite-case harmonic-analytic and RKHS pipeline on the frozen VPD state space;
- theorem-backed uniformly discrete infinite extension only under the explicitly stated assumptions;
- exact admissible official topology-aware loss families and any explicitly demoted secondary families;
- exact smooth-stratum or stratum-local differentiation rules as specified in Section I;
- and exact recursive higher-depth constructions only as mathematical objects, not automatically as empirically validated scientific mechanisms.

Every mathematical claim must state:

- the exact subsection on which it depends,
- the assumption set on which it depends,
- whether it is exact, theorem-backed, approximate, engineering-level, or deferred-semantic,
- and whether it is active in the official benchmark or merely mathematically admissible.

No mathematical claim may be rhetorically upgraded into an empirical, network-scientific, or public-health claim without separate licensing elsewhere in the document.

### 9.2 Methodological claims

The project may claim only the following methodological categories:

- topology-as-loss / topology-as-constraint is the active benchmark role of topology;
- the official benchmark is intervention-first and is defined relative to frozen SAIS intervention-response estimands;
- the benchmark compares topology-aware structural-alignment losses against serious non-topological controls on the same primary task;
- the benchmark is chronology-respecting, simulator-defined, and evaluation-separated with distinct training-label and evaluation-label Monte Carlo batches;
- the benchmark includes explicit graph-statistic, spectral, heuristic, and support-task controls;
- the benchmark distinguishes clearly between primary intervention-task results and support-task results.

No claim is licensed about:

- topology-as-input,
- native topological encoders,
- topology-aware message passing as an official design feature,
- deployment-ready architecture,
- or universal superiority of topology across all tasks

unless those claims are added by formal amendment and frozen elsewhere in the document.

### 9.3 Empirical claims

Every empirical claim must state explicitly:

- dataset scope,
- whether the claim concerns the primary intervention task or a support task,
- the exact target object,
- the exact metric,
- the comparator family,
- the uncertainty summary,
- the sensitivity scope,
- and whether the claim is global or regime-specific.

A valid empirical claim must satisfy all of the following:

- it is evaluated on the frozen task interface appropriate to that claim;
- it survives all required robustness, null-model, and scope-consistency checks frozen elsewhere in the document;
- it does not rely on exploratory comparisons while being phrased as a primary conclusion;
- and it does not substitute support-task success for primary-task success.

No support-task win may be presented as though it were a primary intervention-task win.

No scalar-risk win may be presented as though it were automatically a containment or intervention-sufficiency win.

No regime-specific win may be presented as though it were a global benchmark win unless the claim explicitly states that it is regime-restricted.

### 9.4 Applied claims

Every applied claim must be phrased in benchmark-valid language only. Admissible forms include:

- simulator-conditional intervention-response signal under the frozen SAIS benchmark,
- contact-structured epidemic-risk proxy under declared SAIS assumptions,
- comparative predictive utility for intervention sufficiency, awareness sufficiency, or containment estimation under declared assumptions,
- and structural signal relevance within the frozen benchmark scope.

By default, the following stronger claims are inadmissible:

- real intervention efficacy,
- field outbreak probability,
- healthcare burden in the target population,
- operational deployment readiness,
- policy recommendation,
- causal intervention effect in the real world,
- or transportable public-health effect outside the benchmark population and simulator regime.

Such claims are forbidden unless the required validation ladder is satisfied explicitly and the supporting evidence is frozen elsewhere in the document.

No section may convert:

- a simulator-conditional benchmark win into a policy claim,
- a contact-network benchmark result into a disease-wide population claim,
- or a support-task gain into an applied intervention claim

by rhetorical drift, vague language, or omission of the relevant assumptions.

---

## X. Appendices within Design.md

### Appendix A. Global notation ledger

This appendix is the official notation-control ledger for the benchmark.

It must contain a table with the following columns:

- symbol,
- section of first definition,
- mathematical/statistical/network/public-health role,
- overloaded elsewhere? yes/no,
- replacement notation if ambiguity exists.

At minimum, this appendix must resolve all high-risk notation collisions involving:

- \(\tau\) as temporal lag versus any threshold-like notation,
- \(\kappa\) as alerting parameter versus any awareness-sufficiency output or function,
- \(\mathcal W_t\) versus \(\mathcal W_t^{\mathrm{evt}}\),
- \(G_t\), \(\mathcal F_t\), \(P_t\), \(D_t\), and \(C_t\),
- \(Y_t(v,a,\kappa)\), \(\mu_t(v,a,\kappa)\), and their Monte Carlo estimators,
- \(\widehat a_t^\star\), \(\widehat \kappa_t^\star\), and \(\widehat{\mathcal C}_t\),
- and all symbols used both in mathematical and public-health sections.

This appendix must also identify any notation that is permitted to remain overloaded and explain why that overloading is harmless.

No symbol may remain multiply overloaded without an explicit note here.

### Appendix B. Assumption ledger

This appendix is the official dependency-control ledger for assumptions.

It must list every numbered or explicitly frozen assumption in the document, with columns:

- assumption identifier,
- exact text,
- section,
- domain,
- exact / theorem-backed / engineering / deferred-semantic status,
- active in official benchmark? yes/no,
- and which claims depend on it.

At minimum, this appendix must include:

- all mathematical assumptions,
- all simulator assumptions,
- all calibration assumptions,
- all intervention-family assumptions,
- all topology-semantics assumptions,
- all dataset-scope assumptions,
- and all public-health interpretation assumptions.

No claim may appear in the paper unless the assumptions on which it depends are traceable through this appendix.

### Appendix C. Invariant ledger

This appendix is the official runtime-invariant control ledger.

It must list every runtime invariant, with columns:

- invariant identifier,
- description,
- checked at config parse / preprocessing / training / evaluation,
- hard-fail on violation? yes/no.

At minimum, this appendix must include invariants for:

- chronology-respecting splits,
- train/validation/test information firewalls,
- distinct training-label and evaluation-label Monte Carlo batches,
- frozen intervention-grid consistency,
- topology-object cache validity,
- calibration-object consistency,
- metric computability,
- and support-task versus primary-task labeling consistency.

No implementation may silently relax an invariant listed here.

### Appendix D. Artifact ledger

This appendix is the official reproducibility-artifact ledger.

It must list every persisted artifact, with columns:

- artifact name,
- generating section,
- cache key,
- versioning rule,
- invalidation rule,
- whether required for final reproducibility.

At minimum, this appendix must include:

- split definitions,
- calibration outputs,
- simulator seed bundles,
- Monte Carlo label artifacts,
- topology-state artifacts,
- model checkpoints,
- evaluation outputs,
- and final reporting tables or figures derived from frozen benchmark runs.

No artifact required for final reproducibility may be omitted from this appendix.

### Appendix E. Complexity ledger

This appendix is the official tractability ledger for the benchmark.

It must list, for every major benchmark component:

- object or algorithm,
- preprocessing complexity,
- train-time complexity,
- inference complexity,
- memory complexity,
- approximation status.

At minimum, this appendix must include:

- event-window construction,
- graph aggregation,
- filtration construction,
- persistence computation,
- VPD construction,
- Wasserstein-loss computation,
- RKHS-loss computation,
- response-surface label generation,
- model training,
- and evaluation-time intervention-response prediction.

If any complexity claim is only heuristic or implementation-dependent, that status must be stated explicitly.

### Appendix F. Exact-versus-approximate ledger

This appendix is the official object-status ledger.

It must list every major object with columns:

- object,
- exact / theorem-backed / approximate / engineering / deferred semantic,
- section of definition,
- active in official benchmark? yes/no.

At minimum, this appendix must include:

- the VPD group,
- lifted \(W_1\),
- the finite RKHS branch,
- the infinite uniformly discrete extension,
- random-feature approximations,
- simulator-defined intervention-response surfaces,
- Monte Carlo estimators,
- topology-aware losses,
- support-task targets,
- and recursive higher-depth constructions.

No object may be used in the benchmark without a status entry here.

### Appendix G. Failure-mode ledger

This appendix is the official fail-fast control ledger.

It must list every hard-fail condition with:

- failure identifier,
- subsystem,
- trigger,
- stage,
- mandatory logging behavior.

At minimum, this appendix must include hard-fail conditions for:

- invalid configuration,
- chronology violation,
- leakage detection,
- invalid calibration,
- degenerate label generation,
- topology-computation failure,
- undefined loss,
- NaNs or infinities,
- invalid evaluation metric computation,
- and report-generation inconsistency.

No hard-fail rule may exist in implementation without being listed here, and no listed hard-fail rule may be downgraded silently in code.

### Appendix H. Claim-boundary ledger

This appendix is the official claim-boundary control ledger.

For each domain, it must list:

- admissible claim forms,
- inadmissible claim forms,
- evidence required for stronger claims,
- and the section where the corresponding evidence ladder is defined.

At minimum, it must distinguish separately among:

- mathematical admissibility claims,
- statistical predictive claims,
- machine-learning benchmark claims,
- network-science nonredundancy claims,
- public-health interpretation claims,
- and real-world operational or policy claims.

This appendix must also identify:

- which claims are permitted only for the primary intervention task,
- which claims are permitted only for support tasks,
- which claims are simulator-conditional only,
- and which claims are forbidden unless external validation is added by amendment.

No claim appearing in the paper, benchmark report, abstract, figure caption, or conclusion may exceed the strongest admissible form listed for its domain.