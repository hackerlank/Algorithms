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

#include <assert.h>
#include <time.h>
#include <math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_permutation.h>
#include <ulib/util_log.h>
#include <ulib/util_algo.h>
#include <ulib/math_rand_prot.h>
#include "mspd.h"

using namespace std;

mspd::mspd()
{
    uint64_t iv = time(NULL);
    RAND_NR_INIT(_u, _v, _w, iv);
}

mspd::~mspd()
{
}

double mspd::unif()
{
    return RAND_NR_DOUBLE(RAND_NR_NEXT(_u, _v, _w));
}

double mspd::kernel(const gsl_vector *vx, double tau)
{
    assert(tau > 0);

    double nrm = gsl_blas_dnrm2(vx);
    return exp(-nrm*nrm/2.0/tau/tau);
}
	
void mspd::perturb(problem *prob,
		   const gsl_vector *vx,
		   size_t iter,
		   double beta,
		   gsl_matrix *mdx,
		   gsl_matrix *mny,
		   gsl_vector *vt)
{
    assert(mdx->size1 == mny->size2 && mdx->size2 == vt->size);

    double range = beta * pow(iter, -1.0/3.0);

    for (size_t i = 0; i < mdx->size1; ++i) {
	gsl_vector_view vpx = gsl_matrix_row(mdx, i);
	gsl_vector_view vy = gsl_matrix_column(mny, i);
	for (size_t j = 0; j < mdx->size2; ++j)
	    gsl_vector_set(vt, j, (2*unif()-1) * range); // randomize
	gsl_vector_add(vt, vx);
	prob->proj(vt);  // project to a new point
	prob->eval(vt, &vy.vector);
	gsl_vector_sub(vt, vx);  // calculate the actual perturbation
	gsl_vector_memcpy(&vpx.vector, vt);
    }
}

void
mspd::ortho_proj(const gsl_matrix *mdx,
		 double lambda,
		 double tau,
		 gsl_matrix *mp)
{
    assert(mp->size1 == mdx->size2 && mp->size2 == mdx->size1);

    int signum;
    gsl_matrix *m1 = gsl_matrix_calloc(mdx->size1, mdx->size1); // kernel matrix
    gsl_matrix *m2 = gsl_matrix_alloc (mdx->size2, mdx->size2);
    gsl_matrix *m3 = gsl_matrix_alloc (mdx->size2, mdx->size2);
    gsl_matrix *m4 = gsl_matrix_alloc (mdx->size2, mdx->size1);
    gsl_permutation *perm = gsl_permutation_alloc(mdx->size2);
    gsl_vector_view vd;
    
    assert(m1 && m2 && m3 && m4 && perm);

    // compute the diagonal weight matrix
    vd = gsl_matrix_diagonal(m1);
    for (size_t i = 0; i < m1->size1; ++i) {
	gsl_vector_const_view row = gsl_matrix_const_row(mdx, i);
	gsl_vector_set(&vd.vector, i, kernel(&row.vector, tau));
    }

    vd = gsl_matrix_diagonal(m2);
    assert(!gsl_blas_dgemm(CblasTrans,   CblasNoTrans, 1.0, mdx, m1, 0.0, mp));
    assert(!gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, mp, mdx, 0.0, m2));
    assert(!gsl_vector_add_constant(&vd.vector, lambda));
    assert(!gsl_linalg_LU_decomp(m2, perm, &signum));
    assert(!gsl_linalg_LU_invert(m2, perm, m3));
    assert(!gsl_blas_dgemm(CblasNoTrans, CblasTrans,   1.0, m3, mdx, 0.0, m4));
    assert(!gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, m4, m1,  0.0, mp));

    gsl_matrix_free(m1);
    gsl_matrix_free(m2);
    gsl_matrix_free(m3);
    gsl_matrix_free(m4);
    gsl_permutation_free(perm);
}

void mspd::operator()(problem *prob,
		      gsl_vector *vx0,
		      size_t niter,
		      size_t bs,
		      double beta,
		      double alpha,
		      double lambda,
		      double tau,
		      const gsl_vector *vub,
		      gsl_matrix *mxval,
		      gsl_matrix *mfval)
{
    size_t n = mxval->size2;
    size_t k = mfval->size1;

    assert(mfval->size2 == niter);
    assert(mxval->size1 == niter);
    assert(vub->size    == k);
    assert(niter > 0);
    assert(k > 0);

    gsl_vector *vt = gsl_vector_alloc(n);
    assert(vt);

    gsl_vector_memcpy(vt, vx0);
    prob->proj(vt);
    gsl_vector_view row = gsl_matrix_row(mxval, 0);
    gsl_vector_memcpy(&row.vector, vt);
    gsl_vector_view col = gsl_matrix_column(mfval, 0);
    prob->eval(&row.vector, &col.vector);

    gsl_matrix *mny = gsl_matrix_alloc(k, bs);
    gsl_matrix *mdx = gsl_matrix_alloc(bs, n);
    gsl_matrix *mp  = gsl_matrix_alloc(n, bs);
    gsl_vector *vw  = gsl_vector_alloc(k);
    gsl_vector *vdw = gsl_vector_alloc(k);
    gsl_vector *vds = gsl_vector_alloc(bs);
    gsl_vector *vgs = gsl_vector_alloc(n);
    assert(mny && mdx && mp && vw && vdw && vds && vgs);

    // vw = (1,0,...,0)^T
    gsl_vector_set(vw, 0, 1.0);
    for (size_t i = 1; i < vw->size; ++i)
	gsl_vector_set(vw, i, 0);

    for (size_t j = 1; j < niter; ++j) {
	gsl_vector_const_view vx = gsl_matrix_const_row(mxval, j - 1);
	gsl_vector_const_view vy = gsl_matrix_const_column(mfval, j - 1);
	perturb(prob, &vx.vector, j, beta, mdx, mny, vt);
	ortho_proj(mdx, lambda, tau, mp);

	for (size_t i = 0; i < bs; ++i) {
	    double ds;
	    gsl_vector_view col = gsl_matrix_column(mny, i);
	    gsl_vector_sub(&col.vector, &vy.vector);
	    gsl_blas_ddot(vw, &col.vector, &ds);
	    gsl_vector_set(vds, i, ds);
	}

	assert(!gsl_blas_dgemv(CblasNoTrans, 1.0, mp, vds, 0.0, vgs));

	double step = alpha/(1.0+alpha*lambda*j);

	gsl_vector_memcpy(vdw, &vy.vector);
	gsl_vector_sub(vdw, vub);
	gsl_vector_scale(vdw, step);
	gsl_vector_add(vw, vdw);
	gsl_vector_set(vw, 0, 1.0);

	gsl_vector_view vxn = gsl_matrix_row(mxval, j);
	gsl_vector_view vyn = gsl_matrix_column(mfval, j);
	gsl_vector_memcpy(vt, &vx.vector);
	gsl_vector_scale(vgs, step);
	gsl_vector_sub(vt, vgs);
	prob->proj(vt);
	gsl_vector_memcpy(&vxn.vector, vt);
	prob->eval(vt, &vyn.vector);
    }

    gsl_matrix_free(mny);
    gsl_matrix_free(mdx);
    gsl_matrix_free(mp);
    gsl_vector_free(vw);
    gsl_vector_free(vdw);
    gsl_vector_free(vds);
    gsl_vector_free(vgs);
    gsl_vector_free(vt);
}
