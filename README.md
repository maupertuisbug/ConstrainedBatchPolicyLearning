This paper provides a way to do off-policy learning using contraints. As mentioned in the paper, this problem can be formulated as a constrained optimization problem. The main solution is to solve this as a Lagragian Dual where the strong duality holds. 

Given the primal problem:

$$
\begin{aligned}
\text{minimize} \quad & f(x) \\
\text{subject to} \quad & g_i(x) \leq 0, \quad i = 1, \dots, m \\
& h_j(x) = 0, \quad j = 1, \dots, p
\end{aligned}
$$

The Lagrangian is defined as:

$$
\mathcal{L}(x, \lambda, \nu) = f(x) + \sum_{i=1}^m \lambda_i g_i(x) + \sum_{j=1}^p \nu_j h_j(x)
$$

The Lagrangian dual function is:

$$
g(\lambda, \nu) = \inf_x \mathcal{L}(x, \lambda, \nu)
$$

The Lagrangian dual problem becomes:

$$
\begin{aligned}
\text{maximize} \quad & g(\lambda, \nu) \\
\text{subject to} \quad & \lambda \geq 0
\end{aligned}
$$


For the experiments in this code, my goal is to show that the primal dual gap reduces as the optimization is done. 