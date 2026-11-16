"""Interior-point alternative for small inverse-dynamics QPs with equalities."""
import numpy as np
import scipy.sparse as sp


def solve_interior(P,q,A,lower,upper):
    import clarabel
    equal=np.isfinite(lower)&np.isfinite(upper)&(np.abs(upper-lower)<1e-12)
    up=np.isfinite(upper)&~equal; low=np.isfinite(lower)&~equal
    matrix=np.concatenate([A[equal],A[up],-A[low]])
    rhs=np.r_[upper[equal],upper[up],-lower[low]]
    cones=[]
    if equal.any():cones.append(clarabel.ZeroConeT(int(equal.sum())))
    if up.any() or low.any():cones.append(clarabel.NonnegativeConeT(int(up.sum()+low.sum())))
    settings=clarabel.DefaultSettings();settings.verbose=False
    settings.tol_feas=1e-8;settings.tol_gap_abs=1e-8;settings.tol_gap_rel=1e-8
    solution=clarabel.DefaultSolver(sp.triu(sp.csc_matrix(P),format='csc'),np.asarray(q),
        sp.csc_matrix(matrix),rhs,cones,settings).solve()
    if str(solution.status) not in ('Solved','AlmostSolved'):
        raise RuntimeError(f'interior QP: {solution.status}')
    return np.asarray(solution.x)
