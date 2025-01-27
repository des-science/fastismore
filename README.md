# fastismore

Importance sampling tools for DES Extensions

## How to use

1. Run your favorite sampler in cosmosis with

```ini
extra_output = ...  data_vector/2pt_theory#639 sigma_crit_inv_lens_source/sigma_crit_inv_1_1 ...
```
where `#639` indicates the size of your scale cut datavector, and the `sigma_crit_i_j` factors should include all combinations of lens bins `i` and source bins `j`.

3. Run importance_sampling.py to compute importance weights for the new data vector.

4. Plot results using post_process.py or your favorite script

