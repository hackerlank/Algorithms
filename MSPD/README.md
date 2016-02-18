MSPD is an implementation of the stochastic primal-dual algorithm[[1](http://papers.nips.cc/paper/4942-stochastic-convex-optimization-with-multiple-objectives)]. While MSPD and [PALD](https://github.com/ZilongTan/Algorithms/tree/master/PALD) are both multi-objective optimization algorithms, MSPD does NOT find a Pareto-optimal solution. MSPD minimizes one single objective function subject to the constraints on other objectives, that is:

argmin<sub>x</sub> E<sub>x</sub> [ f<sub>1</sub>(x) ]

s.t. E<sub>x</sub>[f<sub>i</sub>(x)] <= r<sub>i</sub>, for i = 2,3,...,k

To see why the above problem does not guarantee a Pareto-optimal solution, suppose that x<sub>1</sub> and x<sub>2</sub> yield the same f<sub>1</sub>(x) but different f<sub>2</sub>(x) values: f<sub>2</sub>(x<sub>1</sub>) < f<sub>2</sub>(x<sub>2</sub>). Thus, x<sub>2</sub> is a solution to the above problem but not a Pareto-optimal one. By constrast, [PALD](https://github.com/ZilongTan/Algorithms/tree/master/PALD) gives a Pareto-optmal solution.

### Usage
Both the Matlab implementation and the C++ implementation are provided, of which the interfaces are the same as [PALD](https://github.com/ZilongTan/Algorithms/tree/master/PALD).
