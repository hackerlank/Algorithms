###PALD - Stochastic Multi-objective Optimization

The PAreto Local Desccent (PALD) is an algorithm for solving
multi-objective optimization problems. PALD can efficiently find a
Pareto-optimal solution for problems involving noisy convex objectives
and constraints:

  min<sub>x</sub> E<sub>x</sub> [ f<sub>1</sub>(x), f<sub>2</sub>(x),
  ..., f<sub>k</sub>(x) ]

s.t. E<sub>x</sub>[f<sub>i</sub>(x) <= r<sub>i</sub>


###Features
* Much smaller sample complexity than evolutionary algorithm based
approaches
* Resistance to noise in objectives and constraints
* Provable convergence for convex problems

###Basic Usage
