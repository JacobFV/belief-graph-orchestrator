# math

## state

\[
S_t = (\mathcal{E}_{\le t}, \{\mathcal{G}^{(\ell)}_t\}_{\ell=0}^{5}, \mathcal{R}_t, \{\mathcal{Q}^{(k,\ell)}_t\}_{k,\ell}, \hat p_t, \mathcal{B}_t, \Xi_t)
\]

## scale coordinates

\[
\sigma = (\tau,\rho,\eta,\kappa,\chi)
\]

## node embeddings

\[
\phi_v = (z_v^{\mathrm{obj}}, z_v^{\mathrm{dyn}}, z_v^{\mathrm{belief}}, z_v^{\mathrm{value}})
\]

and complex/scale read

\[
q_v^{(k,\ell)} = F_{k,\ell}(\phi_v, z_t^{(k)}, \tau_t)
\]

## association

\[
s_{ij} = \alpha \,\mathrm{sim}_{\mathrm{vis}}(i,j) + \beta \,\mathrm{IoU}(i,j) + \gamma \,\mathrm{sim}_{\mathrm{text}}(i,j) + \delta \,\mathrm{roleCompat}(i,j) + \epsilon \,\mathrm{temporalContinuity}(i,j)
\]

## hotness

\[
h_t(v) = w_r r_t(v) + w_f f_t(v) + w_b b_t(v) + w_m m_t(v) + w_h h^{\mathrm{hist}}_t(v) + w_a a_t(v)
\]

## query aperture

\[
\mathcal{Q}^{(k,\ell)}_t = \mathrm{TopK}(s^{(k,\ell)}_{\mathrm{front}} + s^{(k,\ell)}_{\mathrm{spatial}} + s^{(k,\ell)}_{\mathrm{causal}} + s^{(k,\ell)}_{\mathrm{branch}} + s^{(k,\ell)}_{\mathrm{semantic}} + s^{(k,\ell)}_{\mathrm{hist}} + s^{(k,\ell)}_{\mathrm{analog}})
\]

## contracts

\[
z^{(\ell)}_{t+1} = F_\ell(z^{(\ell)}_t, C_{\ell+1\downarrow}, M_{\ell-1\uparrow}, \mathcal{Q}^{(\ell)}_t, \xi_t^{(\ell)})
\]

## pointer dynamics

\[
p_{t+1} = f_\theta(p_t, u_t, \Delta t_t) + \epsilon_t
\]
\[
o_t \sim g_\psi(p_t, x_t) + \nu_t
\]

control target energy:

\[
E_{\mathrm{target}}(p) = -\log \pi_t^{\mathrm{target}}(p)
\]

controller objective:

\[
J = \sum_{s=t}^{t+H} E_{\mathrm{target}}(\hat p_s) + \lambda_u \|u_s\|^2 + \lambda_\Delta \|u_s-u_{s-1}\|^2 + \lambda_\Sigma \, \mathrm{tr}(\Sigma_s)
\]

## branches

\[
\mathcal{B}(a_t)=\{b_{t,1},\dots,b_{t,K}\}
\]

with posterior update

\[
P(b_{t,k} \mid \Delta \mathcal{E}_{t+1:t+\ell}) \propto P(\Delta \mathcal{E}_{t+1:t+\ell} \mid b_{t,k})\pi_{t,k}
\]

## value-of-retrospection

\[
\mathrm{VoR}_{k,\ell}(t) = \mathbb{E}[\Delta U \mid \text{retrieve history}] - \lambda_T\Delta T - \lambda_C\Delta C
\]

frontier hazard

\[
H_t = a_1 u_t^{\mathrm{ptr}} + a_2 \phi_t^{\mathrm{frag}} + a_3 \mathcal{H}(\mathcal{B}_t) + a_4 \tau_t^{\mathrm{timeout}}
\]

## scheduler objective

\[
\max_{b_{k,\ell}\ge 0} \sum_{k,\ell} b_{k,\ell}(t)(U_{k,\ell}(t)-\lambda_C C_{k,\ell}(t)-\lambda_D D_{k,\ell}(t))
\]

## losses

\[
\mathcal{L} = \lambda_1\mathcal{L}_{\mathrm{OCR}} + \lambda_2\mathcal{L}_{\mathrm{pseudoStruct}} + \lambda_3\mathcal{L}_{\mathrm{sameEntity}} + \lambda_4\mathcal{L}_{\mathrm{role}} + \lambda_5\mathcal{L}_{\mathrm{pointer}} + \lambda_6\mathcal{L}_{\mathrm{branch}} + \lambda_7\mathcal{L}_{\mathrm{verify}} + \lambda_8\mathcal{L}_{\mathrm{target}} + \lambda_9\mathcal{L}_{\mathrm{recovery}} + \lambda_{10}\mathcal{L}_{\mathrm{retrieval}}
\]

reference implementation currently instantiates:

\[
\mathcal{L}_{\mathrm{verify}} = \mathrm{CE}(\hat y, y)
\]

and pairwise target ranking

\[
\mathcal{L}_{\mathrm{target}} = \max(0, m - s^+ + s^-)
\]
