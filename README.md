# group-testing-rl
## Current Problem Statement
I am on a gameshow, and the host has an unknown $\vec{x}$. I am trying to uncover that $\vec{x}$. He samples one row $ \vec{w_1}^T=[w_{11}, w_{12}, ..., w_{1n}]$ from the Walsh matrix without replacement and multiplies it with that $\vec{x}$ to get $\vec{y_1}$, which is now a scalar, since he sampled one row. I try to find all $\tilde{x}$ such that $\vec{w_1}\tilde{x}=\vec{y_1}$ is satisfied. I must end up with at most $_{n}C_{k}$ $\tilde{x}$'s, including one that is actually equal to $x$. I keep these all in a pool. Then, the host samples another row, forming the matrix:
$\hat{W}=\begin{bmatrix} \vec{w_1}^T \\ \vec{w_2}^T \end{bmatrix}$
Select the $\tilde{x}$'s from the set we have that are consistent with $\hat{W}{x}={\vec{y_2}}$. We should have less than $_{n}C_{k}$ solutions. However, once the host samples $\frac{klog_{2}(N)}{log_{2}(K)}$ rows, we should only have one unique solution, which is the correct $x$. If we don't, this is an unsuccessful run.
Later, develop a deep policy/reinforcement learning network that can cleverly sample rows of the Walsh matrix such that $x$ is uncovered at the optimal time.
Note: Solution will not scale well.
