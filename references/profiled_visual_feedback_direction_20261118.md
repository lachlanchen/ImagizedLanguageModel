# Next candidate: feature learning with a solved output layer

The direct readout refit reduces actual nonlinear native-coordinate error from
.542497 to .271039 in 27.085 monitored seconds, with the feature map unchanged.
The remaining error is substantially above the .016775 exact-query identity
floor and still fails the fixed .10 requirement. More unchanged stochastic
epochs and lambda tuning are not selected. The next candidate retains the
same raw query inputs and learns the nonlinear expansion while solving its
linear output globally at each outer step.

This follows established variable projection, including
[Pereyra, Scherer and Wong](https://www.csrc.sdsu.edu/research_reports/CSRCR2007-12.pdf)
and [Newman et al.](https://arxiv.org/abs/2007.13171). Those works support the
separable-optimization method, not this visual language model, corpus or GPU
quality claim. Applying it to the native-feedback interface is an engineering
hypothesis, not a new physical law or a demonstrated language algorithm.

For fixed corpus coefficients c_i and native targets y_i=e_i-mu, define

\[
 F(A)=\min_K\left\{\sum_i c_i\|\phi_A(x_i)K-y_i\|^2
                  +\lambda\|K\|_F^2\right\}.
\]

Keep LayerNorm, the source affine statistics, pixel encoder, queries, context
and writer fixed. Only the nonlinear expansion A/bias is an outer variable;
the affine/residual output coefficients are the solved K. Fix lambda from the
initial feature map before outer updates. Recomputing lambda as feature energy
changes would require its additional derivative and would define another
objective. Positive fixed ridge gives a unique K. In exact arithmetic, the
profiled gradient is the partial feature gradient at that solved K; inverse-
matrix differentiation need not be retained in the backward graph.

Before a real trial, compare that partial gradient with differentiation through
the ridge solve on independent numerical fixtures. Require full-source gradient
accumulation with the original masses, not one clipped step per frequency-
weighted unique-row minibatch. Bind common FP32 feature/readout execution for
the fixed-feature control and learned-feature arm, with FP64 sufficient
statistics and solve. Existing BF16 results stay unchanged. A common new
precision baseline must be measured; it cannot be credited as a feature-learning
effect. Record ideal objective, stored-weight execution and every line-search
failure separately.

A prospective finite execution should use a fixed number of outer steps,
bounded backtracking and a decreasing profiled objective, with no development
data or source selection. First qualify real complete-record solves, gradient
direction, fitting/reset and resources; only then bind the full-support head
comparison. Preserve the .10 coordinate gate and the complete-language criteria.
A coordinate pass still requires separately registered live-image conditional
writing, complete autonomous answers/books and untouched-source confirmation.
It does not admit a full language fit by itself.

The exact-query slot conflict remains a separate .016775 floor in this input
representation; adding nonlinear features cannot resolve identical inputs.
That floor does not exclude the .10 target. Do not silently add slot residuals
or change image/query sources during the proposed feature-learning comparison.
No new variable-projection fit has been implemented, registered or executed by
this direction note. Live ownership remains in the private handoff. The full
independent image-language goal remains active.
