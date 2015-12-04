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

### Applications
PALD is used in [Tempo](../DistributedSys/Tempo) for solving optimal resource allocation problems. The [paper](http://arxiv.org/pdf/1512.00757.pdf) includes the analysis of PALD.

### Basic Usage
#### Example
See test.cpp.

#### C++ Interface
The C++ interface is provided in pald.hpp:
``` c++
class pald {
public:
    struct problem {
	// Project the proposed decision vector @x to the nearest
	// point The @x will be replaced with the new point.
	virtual void proj(gsl_vector *x) = 0;
	// Evaluate the value of the objective vector at @x, and store
	// the objective vector in @y.
	virtual void eval(const gsl_vector *x, gsl_vector *y) = 0;
    };
};

```
Here, pald::problem is the abstract base class of an optimization
problem. For this problem, the user implements the following two
methods:
* proj(gsl_vector *x): map (in-place modification) x to a feasible
point in the domain.
* eval(const gsl_vector *x, gsl_vector *y): compute the objective
vector y with the given input vector x.

#### Matlab Interface
The pald.m implements the pald() function, which takes the following parameters:
* @func: a vector function F: R<sup>n</sup> -> R<sup>k</sup>, the output vector represents [ f<sub>1</sub>(x),f<sub>2</sub>(x), ...,f<sub>k</sub>(x) ]
* X0: an initial point, x.
* niter: number of iterations
* bs: minibatch size
* beta: perturbation scaling factor
* alpha: step size paremeter
* lambda: regularization factor
* tau: kernel bandwidth
* R: constant vector representing r<sub>i</sub>

### Requirements
* ulib: https://code.google.com/p/ulib/
* gsl: http://www.gnu.org/software/gsl/gsl.html
* glpk: http://www.gnu.org/software/glpk/
