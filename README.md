# fastismore

Importance sampling tools for DES Extensions.

## How to use

1. Run your favorite sampler in cosmosis with

```ini
extra_output = ... sigma_crit_inv_lens_source/sigma_crit_inv_1_1 sigma_crit_inv_lens_source/sigma_crit_inv_1_2 ... data_vector/2pt_theory#639
```

where `#639` should be replaced with the size of your data vector after scale cuts, and the `sigma_crit_inv_i_j` factors should include all combinations of lens bin `i` and source bin `j`.

3. Run `importance_sampling.py` to compute importance weights for the new data vector.

4. Plot results using `post_process.py` or your favorite script.

