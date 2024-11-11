# Many-body-Hussimi-function-on-spin-chain
Hussimi function for many-body spin-1/2 systems evolving with various Hamiltonians

Hamiltonians:
1. One-Axis Twisting: $\hat{H} = \frac{1}{4}\sum_{kl}\hat{\sigma}^z_k\hat{\sigma}^z_l$
2. Two-Axis Coutner-Twisting: $\hat{H} = \frac{1}{4}\sum_{kl}(\hat{\sigma}^z_k\hat{\sigma}^z_l-\hat{\sigma}^y_k\hat{\sigma}^y_l)$, $\Delta = \frac{1}{2}$
3. Heisenberg XXZ: $\hat{H} = \sum_k \hat{\sigma}^x_k\hat{\sigma}^x_{k+1} +\hat{\sigma}^y_k\hat{\sigma}^y_{k+1} + \Delta\hat{\sigma}^z_k\hat{\sigma}^z_{k+1})$

Time-evolved state
$|\psi(t)\rangle = e^{-it\hat{H}}|-\pi/2,0\rangle$, where $|\theta,\phi\rangle = e^{-i\phi\frac{\pi}{2}\sum_k\hat{\sigma}^z_k}e^{-i\theta\pi\sum_k\hat{\sigma}^x_k}|0\rangle\^{\otimes L}$

Hussimi function is defined as:
$Q(\theta,\phi;|\psi\rangle) = |\langle\theta,\phi|\psi|^2$.
