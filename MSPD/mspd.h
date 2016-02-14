/* One implementation of the Mahdavi's algorithm
 * Copyright (c) 2016 Zilong Tan (eric.zltan@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, subject to the conditions listed in the Colossal
 * LICENSE file. These conditions include: you must preserve this
 * copyright notice, and you cannot mention the copyright holders in
 * advertising related to the Software without their permission.  The
 * Software is provided WITHOUT ANY WARRANTY, EXPRESS OR IMPLIED. This
 * notice is a summary of the Colossal LICENSE file; the license in
 * that file is legally binding.
 */

// Implementing the stochastic primal-dual algorithm described in
// Mahdavi et al., Stochastic Convex Optimization with Multiple
// Objectives, NIPS 2013.

#ifndef _MSPD_H
#define _MSPD_H

#include <stdint.h>
#include <gsl/gsl_matrix.h>

class mspd {
public:
    struct problem {
	// Project the proposed decision vector @x to the nearest
	// point The @x will be replaced with the new point.
	virtual void proj(gsl_vector *x) = 0;
	// Evaluate the value of the objective vector at @x, and store
	// the objective vector in @y.
	virtual void eval(const gsl_vector *x, gsl_vector *y) = 0;
    };

    // Minimize y[0] subject to <y[1],...,y[k]> <= vub, where
    // y[0],...,y[k] are the objectives defined in prob. The
    // optimization minimizes only one objective and is thus not
    // Pareto efficient.
    void operator()(problem *prob,
		    gsl_vector *vx0,       // intitial n-dimensional point
		    size_t niter,          // number of iterations
		    size_t bs,             // mini-batch size
		    double beta,           // perturbation width
		    double alpha,          // step size
		    double lambda,         // regularization factor
		    double tau,            // smoothing parameter
		    const gsl_vector *vub, // constant constraints (0-th entry unused)
		    gsl_matrix *mxval,     // niter-by-n matrix
		    gsl_matrix *mfval);    // k-by-niter matrix

    mspd();
    ~mspd();

private:
    double unif();

    static double
    kernel(const gsl_vector * x, double tau);

    // internal routines
    void
    perturb(problem *prob,
	    const gsl_vector *vx,
	    size_t iter,
	    double beta,
	    gsl_matrix *mdx,
	    gsl_matrix *mny,
	    gsl_vector *vt);

    static void
    ortho_proj(const gsl_matrix *mdx,
	       double lambda,
	       double tau,
	       gsl_matrix *mp);

// RNG context
    uint64_t _u, _v, _w;
};

#endif
