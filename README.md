# fastismore

Fast importance sampling for model robustness evaluation. Tools for DES 3x2pt extensions.

## Installation

```sh
pip install git+https://github.com/des-science/fastismore
```

## How to use

1. Run your favorite sampler in cosmosis with

```ini
extra_output = ... sigma_crit_inv_lens_source/sigma_crit_inv_1_1 sigma_crit_inv_lens_source/sigma_crit_inv_1_2 ... data_vector/2pt_theory#639
```
where `#639` should be replaced with the size of your data vector after scale cuts, and the `sigma_crit_inv_i_j` factors should include all combinations of lens bin `i` with source bin `j`.

2. Run `fastis-sample` to compute importance weights for the new data vector.

3. Plot results using `fastis-plot` or your favorite script.

If you prefer to work in a notebook environment, chains can be loaded as in the following example:

```python
import fastismore
import fastismore.plot

baseline = fastismore.Chain('baseline_chain.txt')
contaminated = fastismore.ImportanceChain('importance_weights.txt', baseline)

fastismore.plot.plot_2d(param1, param2, [baseline, contaminted], truth, labels, sigma=0.3))
```

For more use cases, check the examples directory.
