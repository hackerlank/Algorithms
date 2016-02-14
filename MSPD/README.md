MSPD is an implementation of the stochastic primal-dual algorithm[[1](http://papers.nips.cc/paper/4942-stochastic-convex-optimization-with-multiple-objectives)]. While MSPD and [PALD](https://github.com/ZilongTan/Algorithms/tree/master/PALD) are both multi-objective optimization algorithms, MSPD does NOT find a Pareto-optimal solution. MSPD minimizes one single objective function subject to the constraints on other objectives, that is:

argmin<sub>x</sub> E<sub>x</sub> [ f<sub>1</sub>(x) ]

s.t. E<sub>x</sub>[f<sub>i</sub>(x)] <= r<sub>i</sub>, for i = 2,3,...,k




