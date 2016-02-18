/* The PAreto Local Descent (PALD) Algorithm
 * Copyright (c) 2015 Zilong Tan (eric.zltan@gmail.com)
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

#include <time.h>
#include <math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_permutation.h>
#include <glpk.h>
#include <ulib/util_log.h>
#include <ulib/util_algo.h>
#include <ulib/util_macro.h>
#include <ulib/math_rand_prot.h>
#include "pald.hpp"

using namespace std;

pald::pald()
    : _rhomax(RHOMAX), _zlimit(ZLIMIT)
{
    uint64_t iv = time(NULL);
    RAND_NR_INIT(_u, _v, _w, iv);
}

pald::~pald()
{
}

double pald::unif()
{
    return RAND_NR_DOUBLE(RAND_NR_NEXT(_u, _v, _w));
}

double pald::kernel(const gsl_vector *vx, double tau)
{
    ULIB_ASSERT(tau > 0);

    double nrm = gsl_blas_dnrm2(vx);
    return exp(-nrm*nrm/2.0/tau/tau);
}
	
void pald::positive_components(gsl_matrix *mp, const gsl_matrix *m)
{
    ULIB_ASSERT(mp->size1 == m->size1 && mp->size2 == m->size2);
    
    for (size_t i = 0; i < mp->size1; ++i) {
	for (size_t j = 0; j < mp->size2; ++j) {
	    double e = gsl_matrix_get(m, i, j);
	    gsl_matrix_set(mp, i, j, e>0? e: 0);
	}
    }
}

void pald::negative_components(gsl_matrix *mn, const gsl_matrix *m)
{
    ULIB_ASSERT(mn->size1 == m->size1 && mn->size2 == m->size2);
    
    for (size_t i = 0; i < mn->size1; ++i) {
	for (size_t j = 0; j < mn->size2; ++j) {
	    double e = gsl_matrix_get(m, i, j);
	    gsl_matrix_set(mn, i, j, e<0? e: 0);
	}
    }
}

void pald::perturb(problem *prob,
		   const gsl_vector *vx,
		   size_t iter,
		   double beta,
		   gsl_matrix *mdx,
		   gsl_matrix *mny,
		   gsl_vector *vt)
{
    ULIB_ASSERT(mdx->size1 == mny->size2 && mdx->size2 == vt->size);

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
pald::ortho_proj(const gsl_matrix *mdx,
		 double lambda,
		 double tau,
		 gsl_matrix *mp)
{
    ULIB_ASSERT(mp->size1 == mdx->size2 && mp->size2 == mdx->size1);

    int signum;
    gsl_matrix *m1 = gsl_matrix_calloc(mdx->size1, mdx->size1); // kernel matrix
    gsl_matrix *m2 = gsl_matrix_alloc (mdx->size2, mdx->size2);
    gsl_matrix *m3 = gsl_matrix_alloc (mdx->size2, mdx->size2);
    gsl_matrix *m4 = gsl_matrix_alloc (mdx->size2, mdx->size1);
    gsl_permutation *perm = gsl_permutation_alloc(mdx->size2);
    gsl_vector_view vd;
    
    ULIB_ASSERT(m1 && m2 && m3 && m4 && perm);

    // compute the diagonal weight matrix
    vd = gsl_matrix_diagonal(m1);
    for (size_t i = 0; i < m1->size1; ++i) {
	gsl_vector_const_view row = gsl_matrix_const_row(mdx, i);
	gsl_vector_set(&vd.vector, i, kernel(&row.vector, tau));
    }

    vd = gsl_matrix_diagonal(m2);
    ULIB_ASSERT(!gsl_blas_dgemm(CblasTrans,   CblasNoTrans, 1.0, mdx, m1, 0.0, mp));
    ULIB_ASSERT(!gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, mp, mdx, 0.0, m2));
    ULIB_ASSERT(!gsl_vector_add_constant(&vd.vector, lambda));
    ULIB_ASSERT(!gsl_linalg_LU_decomp(m2, perm, &signum));
    ULIB_ASSERT(!gsl_linalg_LU_invert(m2, perm, m3));
    ULIB_ASSERT(!gsl_blas_dgemm(CblasNoTrans, CblasTrans,   1.0, m3, mdx, 0.0, m4));
    ULIB_ASSERT(!gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, m4, m1,  0.0, mp));

    gsl_matrix_free(m1);
    gsl_matrix_free(m2);
    gsl_matrix_free(m3);
    gsl_matrix_free(m4);
    gsl_permutation_free(perm);
}

void
pald::jacob_trans(const gsl_matrix *mp,
		  const gsl_matrix *mny,
		  const gsl_matrix *my,
		  gsl_matrix *mjt)
{
    gsl_matrix *m = gsl_matrix_alloc(mny->size1, mny->size2);
    ULIB_ASSERT(m);

    gsl_matrix_memcpy(m, mny);
    gsl_matrix_sub(m, my);
    ULIB_ASSERT(!gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, mp, m, 0.0, mjt));

    gsl_matrix_free(m);    
}

void
pald::proxy_grad(const gsl_matrix *mp,
		 const gsl_matrix *mny,
		 const gsl_matrix *my,
		 const gsl_matrix *mub,
		 const gsl_vector *vw,
		 double rho,
		 gsl_vector *vds)
{
    ULIB_ASSERT(mp->size1 == vds->size);    
    ULIB_ASSERT(mp->size2 == mny->size2);
    ULIB_ASSERT(mny->size1 == my->size1);
    ULIB_ASSERT(mny->size2 == my->size2);
    ULIB_ASSERT(mny->size1 == mub->size1);
    ULIB_ASSERT(mny->size2 == mub->size2);
    ULIB_ASSERT(mub->size1 == vw->size);
    ULIB_ASSERT(vw->size == mny->size1);
    
    gsl_vector *vgs = gsl_vector_alloc(mny->size2);
    gsl_matrix *md  = gsl_matrix_alloc(mny->size1, mny->size2);
    gsl_matrix *m1  = gsl_matrix_alloc(mny->size1, mny->size2);
    gsl_matrix *m2  = gsl_matrix_alloc(mny->size1, mny->size2);
    
    ULIB_ASSERT(vgs && md && m1 && m2);

    gsl_matrix_memcpy(md, mny);
    gsl_matrix_sub(md, my);

    for (size_t i = 0; i < mny->size1; ++i) {
	for (size_t j = 0; j < mny->size2; ++j)
	    gsl_matrix_set(m1, i, j,
			   _max(gsl_matrix_get(mny, i, j),
				gsl_matrix_get(mub, i, j)));
    }

    for (size_t i = 0; i < mny->size1; ++i) {
	for (size_t j = 0; j < mny->size2; ++j)
	    gsl_matrix_set(m2, i, j,
			   _max(gsl_matrix_get(my,  i, j),
				gsl_matrix_get(mub, i, j)));
    }

    gsl_matrix_sub(m1, m2);
    gsl_matrix_scale(m1, rho);
    gsl_matrix_sub(md, m1);
    
    ULIB_ASSERT(!gsl_blas_dgemv(CblasTrans,   1.0, md, vw,  0.0, vgs));
    ULIB_ASSERT(!gsl_blas_dgemv(CblasNoTrans, 1.0, mp, vgs, 0.0, vds));

    gsl_vector_free(vgs);
    gsl_matrix_free(md);
    gsl_matrix_free(m1);
    gsl_matrix_free(m2);
}

void pald::comp_weight(const gsl_matrix *mjt,
		       const gsl_vector *vni,
		       gsl_vector *vw)
{
    ULIB_ASSERT(vni->size == mjt->size2);

    int rc;
    size_t nv = gsl_blas_dasum(vni);

    if (nv) {
	gsl_matrix *mjnd = gsl_matrix_alloc(mjt->size1, nv);
	gsl_matrix *m    = gsl_matrix_alloc(nv, mjt->size2);
	ULIB_ASSERT(mjnd && m);

	size_t t = 0;
	for (size_t j = 0; j < mjt->size2; ++j) {
	    if (gsl_vector_get(vni, j)) {
		gsl_vector_view dcol = gsl_matrix_column(mjnd, t++);
		gsl_vector_const_view scol = gsl_matrix_const_column(mjt, j);
		gsl_vector_memcpy(&dcol.vector, &scol.vector);
	    }
	}
	ULIB_ASSERT(t == nv);

	ULIB_ASSERT(!gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, mjnd, mjt, 0.0, m));

	glp_prob *lp = glp_create_prob();

	glp_set_obj_dir(lp, GLP_MAX);

	glp_add_rows(lp, nv);
	for (size_t i = 1; i <= nv; ++i)
	    glp_set_row_bnds(lp, i, GLP_UP, 0, 0);

	glp_add_cols(lp, mjt->size2 + 1);
	glp_set_col_bnds(lp, 1, GLP_UP, 0, _zlimit);
	glp_set_obj_coef(lp, 1, 1.0);
	for (size_t j = 2; j <= mjt->size2 + 1; ++j) {
	    glp_set_col_bnds(lp, j, GLP_LO, 0, 0);
	    glp_set_obj_coef(lp, j, 0);
	}

	size_t  ne = nv * (mjt->size2 + 1);
	int    *ia = new int    [ne + 1];
	int    *ja = new int    [ne + 1];
	double *ar = new double [ne + 1];

	for (size_t n = 1; n <= ne; ++n) {
	    ia[n] = (n + mjt->size2)/(mjt->size2 + 1);
	    ja[n] = (n - 1)%(mjt->size2 + 1) + 1;
	    if (ja[n] == 1)
		ar[n] = 1.0;
	    else
		ar[n] = -gsl_matrix_get(m, ia[n]-1, ja[n]-2);
	}

	glp_load_matrix(lp, ne, ia, ja, ar);

	//rc = glp_simplex(lp, NULL);
	rc = glp_interior(lp, NULL);
	//if (rc == 0 && glp_get_status(lp) == GLP_OPT) {
	if (rc == 0 && glp_ipt_status(lp) == GLP_OPT) {
	    for (size_t j = 2; j <= mjt->size2 + 1; ++j)
		gsl_vector_set(vw, j - 2,
			       //glp_get_col_prim(lp, j));
			       glp_ipt_col_prim(lp, j));
	    gsl_vector_scale(vw, 1.0/gsl_blas_dnrm2(vw));
	} else
	    ULIB_WARNING("will use a random weight");

	delete [] ia;
	delete [] ja;
	delete [] ar;
	glp_delete_prob(lp);

	gsl_matrix_free(mjnd);
	gsl_matrix_free(m);
    }

    if (!nv || rc) {
	for (size_t i = 0; i < mjt->size2; ++i)
	    gsl_vector_set(vw, i, unif());
	gsl_vector_scale(vw, 1.0/gsl_blas_dnrm2(vw));
    }
}

double
pald::comp_rho(const gsl_matrix *mjt,
	       const gsl_vector *vni,
	       const gsl_vector *vw)
{
    double rho;

    ULIB_ASSERT(mjt->size2 == vni->size &&
	   mjt->size2 == vw->size);

    size_t nv = gsl_blas_dasum(vni);

    gsl_vector *vri = gsl_vector_alloc(vni->size);
    ULIB_ASSERT(vri);
    for (size_t j = 0; j < vri->size; ++j) {
	gsl_vector_const_view col = gsl_matrix_const_column(mjt, j);
	gsl_vector_set(vri, j, gsl_vector_get(vni, j) &&
		       gsl_blas_dasum(&col.vector));
    }

    size_t nr = gsl_blas_dasum(vri);

    if (nr) {
	gsl_matrix *mjnd = gsl_matrix_alloc(mjt->size1, nv);
	ULIB_ASSERT(mjnd);
	for (size_t t = 0, j = 0; j < vni->size; ++j) {
	    if (gsl_vector_get(vni, j)) {
		gsl_vector_view dcol = gsl_matrix_column(mjnd, t++);
		gsl_vector_const_view scol = gsl_matrix_const_column(mjt, j);
		gsl_vector_memcpy(&dcol.vector, &scol.vector);
	    }
	}
	gsl_matrix *mjnn = gsl_matrix_alloc(mjt->size1, nr);
	ULIB_ASSERT(mjnn);
	for (size_t t = 0, j = 0; j < vri->size; ++j) {
	    if (gsl_vector_get(vri, j)) {
		gsl_vector_view dcol = gsl_matrix_column(mjnn, t++);
		gsl_vector_const_view scol = gsl_matrix_const_column(mjt, j);
		gsl_vector_memcpy(&dcol.vector, &scol.vector);
	    }
	}

	gsl_matrix *mrm  = gsl_matrix_alloc(mjnn->size2, mjt->size2);
	gsl_matrix *mrmp = gsl_matrix_alloc(mjnn->size2, mjnd->size2);
	ULIB_ASSERT(mrm && mrmp);
	ULIB_ASSERT(!gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, mjnn, mjt,  0.0,  mrm));
	ULIB_ASSERT(!gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, mjnn, mjnd, 0.0, mrmp));	

	gsl_vector *vtd = gsl_vector_alloc(mrm->size1);
	gsl_vector *vtn = gsl_vector_alloc(mrmp->size1);
	gsl_vector *vtz = gsl_vector_alloc(mrmp->size1);
	gsl_vector *vwn = gsl_vector_alloc(nv);
	ULIB_ASSERT(vtd && vtn && vtz && vwn);
	
	ULIB_ASSERT(!gsl_blas_dgemv(CblasNoTrans, 1.0, mrm, vw, 0.0, vtd));

	for (size_t t = 0, i = 0; i < vw->size; ++i) {
	    if (gsl_vector_get(vni, i))
		gsl_vector_set(vwn, t++, gsl_vector_get(vw, i));
	}

	ULIB_ASSERT(!gsl_blas_dgemv(CblasNoTrans, 1.0, mrmp, vwn, 0.0, vtn));
	
	double rho1, rho2, imp1, imp2;
	gsl_matrix *mt = gsl_matrix_alloc(mrm->size1, mrm->size2);
	ULIB_ASSERT(mt);

	// case 1: rho >= 0
	gsl_vector_memcpy(vtz, vtd);
	gsl_vector_sub(vtz, vtn);

	size_t id = gsl_vector_min_index(vtz);

	if (gsl_vector_get(vtn, id) > 0)
	    rho1 = 0;
	else {
	    positive_components(mt, mrm);
	    gsl_vector *vt = gsl_vector_alloc(vtd->size);
	    ULIB_ASSERT(vt);
	    ULIB_ASSERT(!gsl_blas_dgemv(CblasNoTrans, 1.0, mt, vw, 0.0, vt));
	    gsl_vector_memcpy(vtz, vtd);
	    gsl_vector_div(vtz, vt);
	    rho1 = gsl_vector_min(vtz);
	    gsl_vector_free(vt);
	}
	imp1 = gsl_vector_get(vtd, id) - rho1 * gsl_vector_get(vtn, id);

	// case 2: rho < 0
	gsl_vector_memcpy(vtz, vtd);
	gsl_vector_add(vtz, vtn);

	id = gsl_vector_min_index(vtz);

	if (gsl_vector_get(vtn, id) <= 0)
	    rho2 = 0;
	else {
	    negative_components(mt, mrm);
	    gsl_vector *vti = gsl_vector_alloc(mt->size1);
	    ULIB_ASSERT(vti);
	    for (size_t i = 0; i < vti->size; ++i) {
		gsl_vector_const_view row = gsl_matrix_const_row(mt, i);
		gsl_vector_set(vti, i, gsl_blas_dasum(&row.vector) != 0);
	    }
	    size_t nti = gsl_blas_dasum(vti);
	    if (nti) {
		gsl_matrix *mt1 = gsl_matrix_alloc(nti, mt->size2);
		ULIB_ASSERT(mt1);
		for (size_t t = 0, i = 0; i < vti->size; ++i) {
		    if (gsl_vector_get(vti, i)) {
			gsl_vector_view drow = gsl_matrix_row(mt1, t++);
			gsl_vector_const_view srow = gsl_matrix_const_row(mt, i);
			gsl_vector_memcpy(&drow.vector, &srow.vector);
		    }
		}
		gsl_vector *vtz1 = gsl_vector_alloc(nti);
		ULIB_ASSERT(vtz1);
		for (size_t t = 0, i = 0; i < vtd->size; ++i) {
		    if (gsl_vector_get(vti, i))
			gsl_vector_set(vtz1, t++, gsl_vector_get(vtd, i));
		}

		gsl_vector *vt = gsl_vector_alloc(vtz1->size);
		ULIB_ASSERT(vt);
		ULIB_ASSERT(!gsl_blas_dgemv(CblasNoTrans, 1.0, mt1, vw, 0.0, vt));
		gsl_vector_div(vtz1, vt);
		rho2 = gsl_vector_max(vtz1);
		gsl_vector_free(vt);
		gsl_vector_free(vtz1);
		gsl_matrix_free(mt1);
	    } else
		rho2 = 0;

	    gsl_vector_free(vti);
	}
	imp2 = gsl_vector_get(vtd, id) - rho2 * gsl_vector_get(vtn, id);

	if (imp1 > imp2)
	    rho = rho1;
	else
	    rho = rho2;

	if (rho > _rhomax)
	    rho = _rhomax;
	else if (rho < -_rhomax)
	    rho = -_rhomax;

	gsl_matrix_free(mjnd);
	gsl_matrix_free(mjnn);
	gsl_matrix_free(mrm);
	gsl_matrix_free(mrmp);
	gsl_matrix_free(mt);
	gsl_vector_free(vtd);
	gsl_vector_free(vtn);
	gsl_vector_free(vtz);
	gsl_vector_free(vwn);

    } else
	rho = 0;

    gsl_vector_free(vri);

    return rho;
}

void pald::operator()(problem *prob,
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

    ULIB_ASSERT(mfval->size2 == niter);
    ULIB_ASSERT(mxval->size1 == niter);
    ULIB_ASSERT(vub->size    == k);
    ULIB_ASSERT(niter > 0);

    gsl_vector *vt = gsl_vector_alloc(n);
    ULIB_ASSERT(vt);

    gsl_vector_memcpy(vt, vx0);
    prob->proj(vt);
    gsl_vector_view row = gsl_matrix_row(mxval, 0);
    gsl_vector_memcpy(&row.vector, vt);
    gsl_vector_view col = gsl_matrix_column(mfval, 0);
    prob->eval(&row.vector, &col.vector);

    gsl_matrix *mub = gsl_matrix_alloc(k,bs);
    gsl_matrix *mny = gsl_matrix_alloc(k, bs);
    gsl_matrix *mdx = gsl_matrix_alloc(bs, n);
    gsl_matrix *mp  = gsl_matrix_alloc(n, bs);
    gsl_matrix *my  = gsl_matrix_alloc(k, bs);
    gsl_matrix *mjt = gsl_matrix_alloc(n, k);
    gsl_vector *vni = gsl_vector_alloc(k);
    gsl_vector *vw  = gsl_vector_alloc(k);
    gsl_vector *vds = gsl_vector_alloc(n);
    ULIB_ASSERT(mub && mny && mdx && mp && my && mjt && vni && vw && vds);

    for (size_t j = 0; j < bs; ++j) {
	gsl_vector_view col = gsl_matrix_column(mub, j);
	gsl_vector_memcpy(&col.vector, vub);
    }

    glp_init_env();
    glp_term_out(GLP_OFF);
	
    for (size_t j = 1; j < niter; ++j) {
	gsl_vector_const_view vx = gsl_matrix_const_row(mxval, j - 1);
	gsl_vector_const_view vy = gsl_matrix_const_column(mfval, j - 1);
	perturb(prob, &vx.vector, j, beta, mdx, mny, vt);
	ortho_proj(mdx, lambda, tau, mp);
	for (size_t i = 0; i < bs; ++i) {
	    gsl_vector_view col = gsl_matrix_column(my, i);
	    gsl_vector_memcpy(&col.vector, &vy.vector);
	}
	jacob_trans(mp, mny, my, mjt);
	for (size_t i = 0; i < k; ++i) {
	    gsl_vector_set(vni, i,
			   gsl_vector_get(&vy.vector, i) >=
			   gsl_vector_get(vub, i));
	}
	comp_weight(mjt, vni, vw);
	double rho = comp_rho(mjt, vni, vw);
	proxy_grad(mp, mny, my, mub, vw, rho, vds);
	gsl_vector_view vxn = gsl_matrix_row(mxval, j);
	gsl_vector_view vyn = gsl_matrix_column(mfval, j);
	gsl_vector_memcpy(vt, &vx.vector);
	gsl_vector_scale(vds, alpha/(1.0+alpha*lambda*j));
	gsl_vector_sub(vt, vds);
	prob->proj(vt);
	gsl_vector_memcpy(&vxn.vector, vt);
	prob->eval(vt, &vyn.vector);
    }

    gsl_matrix_free(mub);
    gsl_matrix_free(mny);
    gsl_matrix_free(mdx);
    gsl_matrix_free(mp);
    gsl_matrix_free(my);
    gsl_matrix_free(mjt);
    gsl_vector_free(vni);
    gsl_vector_free(vw);
    gsl_vector_free(vds);
    gsl_vector_free(vt);

    glp_free_env();
}
