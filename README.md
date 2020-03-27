# Globalized Constrained Gauss Newton
We present a Globalization Scheme of Equality Constrained Gauss Newton Methods based on [the work of O. Knoth 1989](https://link.springer.com/article/10.1007/BF01396345). The framework [CasADi](https://web.casadi.org/) is used to compute derivatives effectively using Automatic Differentiation.

## Testing

### Volleyball Example

We provide a test set of simulated two dimensional points obtained from two cameras filming a three dimensional volleyball trajectory. The algorithm is used to estimate the time step between the pictures. In another approach we implemented the estimation of the gravitational force.

### How to use the test

1. Call volleyball_ggn_h.m
   - This will run the constrained volleyball problem and estimate the time step.
2. Call volleyball_ggn_g.m
   - This will run the constrained volleyball problem and estimate the gravity.

### Comparison to an existing solver

1. Call test_lsq_ggn_h.m
   - This will run the unconstrained volleyball problem and estimate h and compare it to LSQNONLIN.
2. Call test_lsq_ggn_g.m
   - This will run the unconstrained volleyball problem and estimate g and compare it to LSQNONLIN. 

## Acknowledgement

This project was done under the supervision of Professor Dr. Moritz Diehl as part of the course in Numerical Optimization. Sincere thanks are extended to Florian Messerer for his continuous guidance throughout our work on the project, and to Katrin Baumgrtner for providing us with the data that was used in the example. We would like to express our great appreciation to them for their valuable assistance and advice. 