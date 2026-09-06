# Mathematical technical report

This report separates the released Prism operator from historical hypotheses. Statements below are algebraic or conditional proxy results unless explicitly labeled empirical. No equation implies universal task improvement.

## 1. Released operator: partial Euclidean spectral projection

Let $W\in\mathbb R^{m\times n}$, $W=U\Sigma V^\top$, and $U_r$ contain the leading $r$ left singular vectors. Set

$$
P_r=U_rU_r^\top,\qquad D=(I-P_r)W.
$$

Define $\alpha=0$ for $D=0$ or $W=0$. Otherwise,

$$
\alpha=\min\left(1,\frac{0.98\varepsilon\|W\|_F}{\|D\|_F}\right),\qquad
T(W)=W-\alpha D.
$$

Since $P_r(I-P_r)=0$,

$$
\langle P_rW,(I-P_r)W\rangle_F=0,
$$

$$
\|T(W)\|_F^2=\|P_rW\|_F^2+(1-\alpha)^2\|D\|_F^2.
$$

This operator generally **reduces** total weight energy. It is not the energy-preserving NPSR operator. Its error bound is

$$
\|T(W)-W\|_F=\alpha\|D\|_F\leq0.98\varepsilon\|W\|_F.
$$

The leading singular components are unchanged and the tail is attenuated. For $0\leq\alpha<1$, the left multiplier $P_r+(1-\alpha)(I-P_r)$ is invertible, so rank is preserved in exact arithmetic. Full projection is the separate case $\alpha=1$.

For a fixed hidden vector $h$,

$$
\|\Delta y\|_2=\|\Delta W h\|_2\leq\|\Delta W\|_2\|h\|_2\leq\|\Delta W\|_F\|h\|_2.
$$

This does not control an autoregressive trajectory: later hidden states and token choices can change. A small weight edit need not produce a proportionally small change in accuracy, reasoning length, or termination.

**Prism settings:** block 23 FFN down only, $r=512$, $\varepsilon=0.12$, source F16, no high-dimensional feature metric, no readout protection. The encoded result must be checked independently because $\operatorname{round}_{F16}(T(W))\neq T(W)$ in general.

## 2. Earlier quantization-channel hypothesis

For $A\in\mathbb R^{r\times n}$ and $B\in\mathbb R^{m\times r}$,

$$
BA=\sum_j b_j a_j^\top,\qquad t_j=\operatorname{vec}(b_ja_j^\top),
$$

$$
\langle t_i,t_j\rangle=(b_i^\top b_j)(a_i^\top a_j),\qquad
K=(B^\top B)\odot(AA^\top).
$$

This lifts a channel into an interaction space without explicitly allocating each $mn$-dimensional vector. Under a fixed-coefficient, zero-mean, second-order-uncorrelated noise model, an operator reconstruction risk can be written

$$
R(c)=(c-\mathbf1)^\top S(c-\mathbf1)+c^\top Nc.
$$

A minimizing solution is

$$
c^*=\mathbf1-(S+N)^\dagger N\mathbf1.
$$

The observed quantized weights do not reveal the true clean $S$ or noise $N$. Plug-in estimates, correlations induced by block quantization, and coefficients dependent on noisy observations invalidate an unconditional improvement guarantee. An exactly representable original operator is a counterexample to blindly assuming it needs denoising. The synthetic precursor is not the released GGUF recipe.

## 3. v0.1 GPEP

The first real-GGUF integration used a proxy path projector:

$$
B'=B-\eta(I-QQ^\top)B.
$$

The archived implementation combines weights from a gated FFN and normalization under a surrogate input model. This defines an implementable edit, not a semantic identification of harmful directions. Its real-model smoke tests did not establish improvement.

## 4. v0.2 NPSR: norm-preserving redistribution

For an orthogonal projector $P$, define

$$
e=\frac{\|PW\|_F^2}{\|W\|_F^2},\qquad V=(P-eI)W.
$$

Projector orthogonality gives

$$
\langle W,V\rangle_F=0,\qquad \|V\|_F^2=e(1-e)\|W\|_F^2.
$$

Therefore

$$
T_\lambda(W)=\frac{W-\lambda V}{\sqrt{1+\lambda^2e(1-e)}}
$$

preserves $\|W\|_F$ in exact arithmetic. Its relative displacement is

$$
\frac{\|T_\lambda(W)-W\|_F}{\|W\|_F}
=\sqrt{2-\frac{2}{\sqrt{1+\lambda^2e(1-e)}}}.
$$

Positive and negative $\lambda$ redistribute energy in opposite directions. Norm preservation does not preserve the allocation of energy among output coordinates. The broad v0.2 edits showed mixed-task regression and long reasoning loops despite satisfying the intended local geometry.

## 5. v0.3: row-tangent updates

For row $w_i$ and proposed projected row $p_i$, remove the radial component:

$$
v_i=p_i-\frac{\langle p_i,w_i\rangle}{\|w_i\|_2^2}w_i.
$$

Then, with zero-row/tangent branches handled separately,

$$
w_i'=\cos\theta_i\,w_i+\sin\theta_i\,\|w_i\|_2\frac{v_i}{\|v_i\|_2}.
$$

This yields $\|w_i'\|_2=\|w_i\|_2$ and relative displacement $2\sin(\theta_i/2)$. A per-row budget and a sum of relative tensor budgets constrain the edit but are not network Lipschitz bounds. Independent row rotations do not imply a rank-preservation theorem. In v0.3, the tested two-layer edit recovered a small regression baseline, not a capability advantage.

## 6. v0.4: cubic interaction-feature geometry

For $x\sim\mathcal N(0,I)$ as a **proxy**, absorb the learned normalization scales into gate/up vectors and write

$$
h_j(x)=\operatorname{SiLU}(g_j^\top x)(u_j^\top x),\quad s_j=\|g_j\|_2,\quad v_j=g_j/s_j.
$$

For $Z\sim\mathcal N(0,1)$ define Hermite coefficients

$$
a_j=\mathbb E[\operatorname{SiLU}(s_jZ)],\quad
b_j=\mathbb E[Z\operatorname{SiLU}(s_jZ)],\quad
c_j=\frac{\mathbb E[(Z^2-1)\operatorname{SiLU}(s_jZ)]}{\sqrt2}.
$$

The implementation uses 48-point Gauss-Hermite quadrature, not an exact finite formula for the original nonlinear integrals. The surrogate is

$$
\widehat h_j(x)=\left[a_j+b_jv_j^\top x+\frac{c_j}{\sqrt2}\big((v_j^\top x)^2-1\big)\right](u_j^\top x).
$$

Let $t_j=v_j^\top u_j$, $m_j=b_jt_j$, $L_j=a_ju_j+\sqrt2c_jt_jv_j$, and let matrices $V,U$ have rows $v_j^\top,u_j^\top$. Define $R=VV^\top$, $S=UU^\top$, and $T=VU^\top$. Orthogonal Gaussian chaos decomposition gives the surrogate moment identity

$$
\begin{aligned}
K=\mathbb E[\widehat h\widehat h^\top]
={}&mm^\top+LL^\top\\
&+(bb^\top)\odot(R\odot S+T\odot T^\top)\\
&+(cc^\top)\odot\big((R\odot R)\odot S+2R\odot T\odot T^\top\big).
\end{aligned}
$$

This is exact for the chosen polynomial surrogate coefficients, not for actual model activations. In particular, RMS-normalized residual states are not independent Gaussian variables. Higher-order features here are analytical interaction features, not additional deployed layers.

## 7. Weighted reader-space approximation

Construct a positive-definite reader metric $H$ from normalized downstream weight rows plus a ridge term. With $C=H^{1/2}$ and $A=CW$,

$$
M=AKA^\top,\quad P_r=Q_rQ_r^\top,\quad W_r=C^{-1}P_rCW,
$$

where $Q_r$ spans the leading eigenspace of $M$. For fixed $H\succ0$ and $K\succeq0$, this realizes a solution of

$$
\min_{\operatorname{rank}(Z)\leq r}\|C(Z-W)K^{1/2}\|_F^2.
$$

This is an approximation objective, not a task-loss objective. The actual reader proxy omits much of the downstream network and is not its complete Jacobian. Setting $H=I,K=I$ recovers the Euclidean control used by Prism.

## 8. Readout-contrast protection: a conditional statement

Let $O$ be the output head and $\Gamma$ the final RMSNorm scale. For a fixed reference token $j_0$ and selected token set, form rows

$$
A_j=(O_j-O_{j_0})\Gamma.
$$

Let $Q_A$ be an orthonormal basis for the column space of $A^\top$. Project a proposed update into its orthogonal complement:

$$
\Delta W_\perp=(I-Q_AQ_A^\top)\Delta W,\qquad A\Delta W_\perp=0.
$$

If **only the final FFN down matrix changes**, the same input prefix gives the same pre-FFN activation $h$, and thus $A\Delta W_\perp h=0$. Selected logit-contrast numerators are unchanged. Scalar RMS normalization can still rescale all contrasts; probabilities are not invariant. Other tokens can win, the next prefix can diverge, and stopping is not guaranteed. Quantization and finite-precision leakage further weaken the ideal statement.

The combined experimental operator is

$$
W'=W-\alpha(I-Q_AQ_A^\top)\left[W-H^{-1/2}P_rH^{1/2}W\right].
$$

Its protected variants did not pass development screening. They must not be credited for Prism's observed internal results.

## 9. What no formula above establishes

None proves that low-energy directions are useless, that arbitrary GGUF architectures can be edited safely, or that accuracy must improve. The identity baseline is already unbeatable on a perfectly solved task. Claims must specify checkpoint, task distribution, inference policy, resource budget, and uncertainty.

The evaluated objective is better expressed operationally as

$$
F_\theta(B)=\Pr(\text{correct final delivery within generation budget }B).
$$

Weight geometry is a way to propose candidates. Only matched, held-out behavioral evaluation can test whether a candidate improves $F_\theta(B)$ in a stated domain.

## References and relation to prior work

The project uses established tools: SVD and low-rank approximation, orthogonal projections, Gaussian/Hermite moments, and positive-definite metrics. The integration is a research hypothesis, not a priority claim. Related training-free spectral editing: [LASER](https://arxiv.org/abs/2312.13558). Implementation format: [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md). Base model: [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B).

## Completed empirical follow-up

The released Euclidean operator has now been evaluated on full paired GSM8K and IFEval splits under the same Think-on protocol. GSM8K flexible-extraction accuracy increased from 70.43% to 72.71%; IFEval prompt-level strict accuracy decreased from 72.64% to 68.58%, and all other IFEval metrics also declined. This is consistent with the report's separation of matrix-space properties from task guarantees. See [completed benchmark report](BENCHMARK_RESULTS.md) for all metrics, conditions, and limitations. No formula or model weight was changed for this publication.
