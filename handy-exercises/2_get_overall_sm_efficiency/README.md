# get_overall_sm_efficiency

## Usage 

```sh
# filename is the program which we are about to profile.
python3 get_overall_sm_efficiency.py filename
```

PS: Remeber to `export PATH=$PATH:/path/to/ncu`

## Metrics we use and how we calculate

We collect these information with **Nsight Compute**.

| metric name | description | in formula below |
|--|--| -- |
|smsp__cycles_active.avg.pct_of_peak_sustained_elapsed|of cycles with at least one warp in flight| $efficiency$ |
|gpu__time_active.avg|total duration in nanoseconds| $duration $ |

$$
overallSmEfficiency = \Sigma_i^n efficiency_i \times \frac{duration_i}{\Sigma_j^n duration_j}
$$
