### PALD - Stochastic Multi-objective Optimization

The PAreto Local Desccent (PALD) is an algorithm for solving
multi-objective optimization problems. PALD can efficiently find a
Pareto-optimal solution for problems involving noisy convex objectives
and constraints:

argmin<sub>x</sub> E<sub>x</sub> [ f<sub>1</sub>(x),
f<sub>2</sub>(x), ..., f<sub>k</sub>(x) ]

s.t. E<sub>x</sub>[f<sub>i</sub>(x)] <= r<sub>i</sub>,

where f<sub>i</sub>(x) are scalar functions, and r<sub>i</sub> are constants.
The minimization is in the Pareto-improving sense.

### Features
* Much smaller sample complexity than evolutionary algorithm based
approaches
* Resistance to noise in objectives and constraints
* Provable convergence for convex problems

### Basic Usage

#### C++ Interface

#### Matlab Interface
The min\_pald.m implements the min\_pald() function, which takes the following parameters:
* @func: a vector function F: R<sup>n</sup> -> R<sup>k</sup>, the output vector represents [ f<sub>1</sub>(x),f<sub>2</sub>(x), ...,f<sub>k</sub>(x) ]
* X0: an initial point, x.
* niter: number of iterations
* bs: minibatch size
* beta: perturbation scaling factor
* alpha: step size paremeter
* lambda: regularization factor
* tau: kernel bandwidth
* R: constant vector representing r<sub>i</sub>
