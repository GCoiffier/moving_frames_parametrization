import numpy as np
import math
from math import pi
import cmath
from .common import barrier_neg, barrier_neg_prime, barrier_zero, barrier_zero_prime
from numba import jit, prange

@jit(nopython=True, parallel=True, cache=True)
def constraint_edge_fixed_w_noJ1(X, PT, n, m, sepRot):
    """Edges should match up to the rotation they carry. Version where angles and the frame field are fixed"""
    val = np.zeros(2*n, dtype=np.float64)
    for e in prange(n):
        wpt = PT[e]
        w = 2*math.atan(X[sepRot +e])
        ew = cmath.rect(1, pi-wpt+w)
        ima = 4*e
        imb = 4*e+2
        ma = complex(X[ima], X[ima+1])
        mb = complex(X[imb], X[imb+1])
        en = ma + mb*ew
        val[2*e] = en.real
        val[2*e+1] = en.imag
    return val

@jit(nopython=True, parallel=True, cache=True)
def constraint_edge_fixed_w1(X, PT, n, m, sepRot):
    """Edges should match up to the rotation they carry. Version where angles and the frame field are fixed"""
    val = np.zeros(2*n, dtype=np.float64)
    valJ = np.zeros(6*n, dtype=np.float64)
    rowJ = np.zeros(6*n, dtype=np.int32)
    colJ = np.zeros(6*n, dtype=np.int32)
    for e in prange(n):
        wpt = PT[e]
        w = 2*math.atan(X[sepRot +e])
        ew = cmath.rect(1, pi-wpt+w)
        ima = 4*e
        imb = 4*e+2
        ma = complex(X[ima], X[ima+1])
        mb = complex(X[imb], X[imb+1])
        en = ma + mb*ew
        val[2*e] = en.real
        rowJ[6*e],   colJ[6*e],   valJ[6*e]   = 2*e, ima, 1
        rowJ[6*e+1], colJ[6*e+1], valJ[6*e+1] = 2*e, imb, ew.real
        rowJ[6*e+2], colJ[6*e+2], valJ[6*e+2] = 2*e, imb+1, -ew.imag

        val[2*e+1] = en.imag
        rowJ[6*e+3], colJ[6*e+3], valJ[6*e+3] = 2*e+1, ima+1, 1
        rowJ[6*e+4], colJ[6*e+4], valJ[6*e+4] = 2*e+1, imb, ew.imag
        rowJ[6*e+5], colJ[6*e+5], valJ[6*e+5] = 2*e+1, imb+1, ew.real
    return val, valJ, rowJ, colJ

######################################################################################################################################

@jit(nopython=True, parallel=True, cache=True)
def constraint_edge_fixed_w_noJ2(X, I, PT, n, m, sepRot):
    val = np.zeros(2*m, dtype=np.float64)
    for i in prange(m):
        e = I[i,0]
        isa = I[i,1]
        isb = I[i,2]
        ima = 4*e
        imb = 4*e+2
        iw = sepRot + e
        wpt = PT[e]
        w = X[iw]
        w = 2*math.atan(X[sepRot +e])
        ew = cmath.rect(1, pi-wpt+w)
        ma = complex(X[ima], X[ima+1])
        sa = complex(X[isa], X[isa+1])
        mb = complex(X[imb], X[imb+1])
        sb = complex(X[isb], X[isb+1])
        en = (sa - ma) - (sb - mb)*ew
        val[2*i] = en.real
        val[2*i+1] = en.imag
    return val

@jit(nopython=True, parallel=True, cache=True)
def constraint_edge_fixed_w2(X, I, PT, n, m, sepRot):
    val = np.zeros(2*m, dtype=np.float64)
    valJ = np.zeros(12*m, dtype=np.float64)
    rowJ = np.zeros(12*m, dtype=np.int32)
    colJ = np.zeros(12*m, dtype=np.int32)
    for i in prange(m):
        e = I[i,0]
        isa = I[i,1]
        isb = I[i,2]
        ima = 4*e
        imb = 4*e+2
        iw = sepRot + e
        wpt = PT[e]
        w = X[iw]
        w = 2*math.atan(X[sepRot +e])
        ew = cmath.rect(1, pi-wpt+w)
        ma = complex(X[ima], X[ima+1])
        sa = complex(X[isa], X[isa+1])
        mb = complex(X[imb], X[imb+1])
        sb = complex(X[isb], X[isb+1])
        en = (sa - ma) - (sb - mb)*ew

        val[2*i] = en.real
        rowJ[12*i],   colJ[12*i],   valJ[12*i]   = 2*i, isa, 1
        rowJ[12*i+1], colJ[12*i+1], valJ[12*i+1] = 2*i, ima, -1
        rowJ[12*i+2], colJ[12*i+2], valJ[12*i+2] = 2*i, isb,   -ew.real
        rowJ[12*i+3], colJ[12*i+3], valJ[12*i+3] = 2*i, isb+1,  ew.imag
        rowJ[12*i+4], colJ[12*i+4], valJ[12*i+4] = 2*i, imb,    ew.real
        rowJ[12*i+5], colJ[12*i+5], valJ[12*i+5] = 2*i, imb+1, -ew.imag

        val[2*i+1] = en.imag
        rowJ[12*i+6], colJ[12*i+6], valJ[12*i+6] = 2*i+1, isa+1, 1
        rowJ[12*i+7], colJ[12*i+7], valJ[12*i+7] = 2*i+1, ima+1, -1
        rowJ[12*i+8], colJ[12*i+8], valJ[12*i+8] = 2*i+1, isb,    -ew.imag
        rowJ[12*i+9], colJ[12*i+9], valJ[12*i+9] = 2*i+1, isb+1,  -ew.real
        rowJ[12*i+10], colJ[12*i+10], valJ[12*i+10] = 2*i+1, imb,   ew.imag
        rowJ[12*i+11], colJ[12*i+11], valJ[12*i+11] = 2*i+1, imb+1, ew.real
    return val, valJ, rowJ, colJ

######################################################################################################################################

@jit(nopython=True, parallel=True, cache=True)
def constraint_edge_noJ1(X : np.ndarray, Lengths: np.ndarray, PT : np.ndarray, n, m, sepRot):
    """Edges should match up to the rotation they carry. Version where rotations are considered as variables"""
    N = 2*(n+m)
    val = np.zeros(N, dtype=np.float64)
    for e in prange(n):
        ima = 4*e
        imb = 4*e+2
        L = Lengths[e]
        iw = sepRot + e
        wpt = PT[e]
        pt = cmath.rect(1., pi-wpt)
        w = X[iw]
        ma = complex(X[ima], X[ima+1])
        mb = complex(X[imb], X[imb+1])
        za,zb = ma/L, mb/L
        val_e = complex(1,-w) * za + complex(1, w) * pt * zb
        val[2*e] = val_e.real
        val[2*e+1] = val_e.imag
    return val

@jit(nopython=True, parallel=True, cache=True)
def constraint_edge1(X : np.ndarray, Lengths: np.ndarray, PT : np.ndarray, n, m, sepRot):
    """Edges should match up to the rotation they carry. Version where rotations are considered as variables"""
    N = 2*(n+m)
    val = np.zeros(N, dtype=np.float64)
    valJ = np.zeros(5*N, dtype=np.float64)
    rowJ = np.zeros(5*N, dtype=np.int32)
    colJ = np.zeros(5*N, dtype=np.int32)
    
    for e in prange(n):
        ima = 4*e
        imb = 4*e+2
        L = Lengths[e]
        iw = sepRot + e
        wpt = PT[e]
        pt = cmath.rect(1., pi-wpt)
        w = X[iw]

        # for ei and ej, vectors have opposite direction
        ma = complex(X[ima], X[ima+1])
        mb = complex(X[imb], X[imb+1])
        za,zb = ma/L, mb/L
        val_e = complex(1,-w) * za + complex(1, w) * pt * zb
        val[2*e] = val_e.real
        val[2*e+1] = val_e.imag
    
        dev_w = 1j * (pt * zb - za)
        rowJ[10*e], colJ[10*e], valJ[10*e] = 2*e, iw, dev_w.real
        rowJ[10*e+1], colJ[10*e+1], valJ[10*e+1] = 2*e+1, iw, dev_w.imag
        
        dev_a = complex(1, -w)/L
        rowJ[10*e+2], colJ[10*e+2], valJ[10*e+2] = 2*e, ima, dev_a.real
        rowJ[10*e+4], colJ[10*e+4], valJ[10*e+4] = 2*e+1, ima, dev_a.imag
        rowJ[10*e+3], colJ[10*e+3], valJ[10*e+3] = 2*e, ima+1, -dev_a.imag
        rowJ[10*e+5], colJ[10*e+5], valJ[10*e+5] = 2*e+1, ima+1, dev_a.real

        dev_b = complex(1, w) * pt / L
        rowJ[10*e+6], colJ[10*e+6], valJ[10*e+6] = 2*e, imb, dev_b.real
        rowJ[10*e+7], colJ[10*e+7], valJ[10*e+7] = 2*e+1, imb, dev_b.imag
        rowJ[10*e+8], colJ[10*e+8], valJ[10*e+8] = 2*e, imb+1, -dev_b.imag
        rowJ[10*e+9], colJ[10*e+9], valJ[10*e+9] = 2*e+1, imb+1, dev_b.real

    return val, valJ, rowJ, colJ

######################################################################################################################################

@jit(nopython=True, parallel=True, cache=True)
def constraint_edge_noJ2(X : np.ndarray, I : np.ndarray, Lengths: np.ndarray, PT : np.ndarray, n, m, sepRot):
    """Half edges linking an edge to a center of a face"""
    N = 2*m
    val = np.zeros(N, dtype=np.float64)
    for i in prange(m):
        e = I[i,0]
        isa = I[i,1]
        isb = I[i,2]
        ima = 4*e
        imb = 4*e+2
        L = Lengths[n+i]
        iw = sepRot + e
        wpt = PT[e]
        w = X[iw]
        pt = cmath.rect(1., pi-wpt)
        ma = complex(X[ima], X[ima+1])
        sa = complex(X[isa], X[isa+1])
        mb = complex(X[imb], X[imb+1])
        sb = complex(X[isb], X[isb+1])
        za,zb = (sa - ma)/L, (sb - mb)/L
        val_e = complex(1,-w) * za - complex(1, w) * pt * zb
        val[2*i] = val_e.real
        val[2*i+1] = val_e.imag
    return val

@jit(nopython=True, parallel=True, cache=True)
def constraint_edge2(X : np.ndarray, I : np.ndarray, Lengths: np.ndarray, PT : np.ndarray, n, m, sepRot):
    """Half edges linking an edge to a center of a face"""
    N = 2*m
    val = np.zeros(N, dtype=np.float64)
    valJ = np.zeros(9*N, dtype=np.float64)
    rowJ = np.zeros(9*N, dtype=np.int32)
    colJ = np.zeros(9*N, dtype=np.int32)

    for i in prange(m):
        e = I[i,0]
        isa = I[i,1]
        isb = I[i,2]
        ima = 4*e
        imb = 4*e+2
        L = Lengths[n+i]
        iw = sepRot + e
        wpt = PT[e]
        w = X[iw]
        pt = cmath.rect(1., pi-wpt)
        ma = complex(X[ima], X[ima+1])
        sa = complex(X[isa], X[isa+1])
        mb = complex(X[imb], X[imb+1])
        sb = complex(X[isb], X[isb+1])
        za,zb = (sa - ma)/L, (sb - mb)/L
        val_e = complex(1,-w) * za - complex(1, w) * pt * zb
        val[2*i] = val_e.real
        val[2*i+1] = val_e.imag

        dev_w = -1j * (pt * zb + za)
        rowJ[18*i], colJ[18*i],valJ[18*i] = 2*i, iw, dev_w.real
        rowJ[18*i+1], colJ[18*i+1], valJ[18*i+1] = 2*i+1, iw, dev_w.imag
        
        dev_a = complex(1,-w)/L
        rowJ[18*i+2], colJ[18*i+2], valJ[18*i+2] = 2*i, isa, dev_a.real
        rowJ[18*i+3], colJ[18*i+3], valJ[18*i+3] = 2*i, isa+1, -dev_a.imag
        rowJ[18*i+4], colJ[18*i+4], valJ[18*i+4] = 2*i+1, isa, dev_a.imag
        rowJ[18*i+5], colJ[18*i+5], valJ[18*i+5] = 2*i+1, isa+1, dev_a.real

        rowJ[18*i+6], colJ[18*i+6], valJ[18*i+6] = 2*i, ima, -dev_a.real
        rowJ[18*i+7], colJ[18*i+7], valJ[18*i+7] = 2*i, ima+1, dev_a.imag
        rowJ[18*i+8], colJ[18*i+8], valJ[18*i+8] = 2*i+1, ima, -dev_a.imag
        rowJ[18*i+9], colJ[18*i+9], valJ[18*i+9] = 2*i+1, ima+1, -dev_a.real

        dev_b = - complex(1, w) * pt / L
        rowJ[18*i+10], colJ[18*i+10], valJ[18*i+10] = 2*i, isb, dev_b.real
        rowJ[18*i+11], colJ[18*i+11], valJ[18*i+11] = 2*i, isb+1, -dev_b.imag
        rowJ[18*i+12], colJ[18*i+12], valJ[18*i+12] = 2*i+1, isb, dev_b.imag
        rowJ[18*i+13], colJ[18*i+13], valJ[18*i+13] = 2*i+1, isb+1, dev_b.real
            
        rowJ[18*i+14], colJ[18*i+14], valJ[18*i+14] = 2*i, imb, -dev_b.real
        rowJ[18*i+15], colJ[18*i+15], valJ[18*i+15] = 2*i, imb+1, dev_b.imag
        rowJ[18*i+16], colJ[18*i+16], valJ[18*i+16] = 2*i+1, imb, -dev_b.imag
        rowJ[18*i+17], colJ[18*i+17], valJ[18*i+17] = 2*i+1, imb+1, -dev_b.real
    return val, valJ, rowJ, colJ

######################################################################################################################################

@jit(nopython=True, parallel=True, cache=True)
def constraint_rotations_follow_ff_noJ(X : np.ndarray, I : np.ndarray, PT : np.ndarray, n, order):
    """Rotations along edges should match the rotation between the two frames (up to a symmetry of the square)"""
    val  = np.zeros(2*n , dtype=np.float64)
    for i in prange(n):
        iw = I[i,0]
        iza, izb = I[i,1], I[i,2]
        w = X[iw] # tangent(rotation / 2)
        zA, zB = complex(X[iza], X[iza+1]), complex(X[izb], X[izb+1]) # the ff representation
        pt = PT[i]
        wpt = cmath.rect(1, order*(pt - pi))
        eiwp = complex(1, w)**order
        eiwm = complex(1,-w)**order
        val_e = eiwp*zB - eiwm*wpt*zA
        val[2*i] = val_e.real
        val[2*i+1] = val_e.imag
    return val

@jit(nopython=True, parallel=True, cache=True)
def constraint_rotations_follow_ff(X : np.ndarray, I : np.ndarray, PT : np.ndarray, n, order):
    """Rotations along edges should match the rotation between the two frames (up to a symmetry of the square)"""
    val  = np.zeros(2*n , dtype=np.float64)
    valJ = np.zeros(10*n, dtype=np.float64)
    rowJ = np.zeros(10*n, dtype=np.int32)
    colJ = np.zeros(10*n, dtype=np.int32)
    
    for i in prange(n):
        iw = I[i,0]
        iza, izb = I[i,1], I[i,2]
        w = X[iw] # tangent(rotation / 2)
        zA, zB = complex(X[iza], X[iza+1]), complex(X[izb], X[izb+1]) # the ff representation
        pt = PT[i]
        wpt = cmath.rect(1, order*(pt - pi))

        # Cayley transform
        eiwp = complex(1, w)
        eiwp_p = eiwp**order
        eiwm = complex(1,-w)
        eiwm_p = eiwm**order
        val_e = eiwp_p*zB - eiwm_p*wpt*zA
        
        val[2*i] = val_e.real
        val[2*i+1] = val_e.imag

        dev_w = 1j * order * ( (eiwp**(order-1)) * zB   + (eiwm**(order-1)) * zA * wpt)
        rowJ[10*i], colJ[10*i], valJ[10*i] = 2*i, iw, dev_w.real
        rowJ[10*i+1], colJ[10*i+1], valJ[10*i+1] = 2*i+1, iw, dev_w.imag

        dev_a = -eiwm_p*wpt
        rowJ[10*i+2], colJ[10*i+2], valJ[10*i+2] = 2*i, iza, dev_a.real
        rowJ[10*i+4], colJ[10*i+4], valJ[10*i+4] = 2*i+1, iza, dev_a.imag
        rowJ[10*i+3], colJ[10*i+3], valJ[10*i+3] = 2*i, iza+1, -dev_a.imag
        rowJ[10*i+5], colJ[10*i+5], valJ[10*i+5] = 2*i+1, iza+1, dev_a.real

        dev_b = eiwp_p
        rowJ[10*i+6], colJ[10*i+6], valJ[10*i+6] = 2*i, izb, dev_b.real
        rowJ[10*i+8], colJ[10*i+8], valJ[10*i+8] = 2*i+1, izb, dev_b.imag
        rowJ[10*i+7], colJ[10*i+7], valJ[10*i+7] = 2*i, izb+1, -dev_b.imag
        rowJ[10*i+9], colJ[10*i+9], valJ[10*i+9] = 2*i+1, izb+1, dev_b.real

    return val, valJ, rowJ, colJ

######################################################################################################################################

@jit(nopython=True, parallel=True, cache=True)
def soft_feat_noJ(X, I, Tgt):
    n = I.shape[0]
    val = np.zeros(2*n, dtype=np.float64)
    for i in prange(n):
        e = I[i]
        a1,b1,a2,b2 = X[4*e:4*e+4]
        c1, d1 = Tgt[2*i,:]
        c2, d2 = Tgt[2*i+1, :]
        val[2*i] = a1*d1 - b1*c1
        val[2*i+1] = a2*d2 - b2*c2
    return val

@jit(nopython=True, parallel=True, cache=True)
def soft_feat(X, I, Tgt):
    n = I.shape[0]
    val = np.zeros(2*n, dtype=np.float64)
    valJ = np.zeros(4*n, dtype=np.float64)
    rowJ = np.zeros(4*n, dtype=np.int32)
    colJ = np.zeros(4*n, dtype=np.int32)
    for i in prange(n):
        e = I[i]
        a1,b1,a2,b2 = X[4*e:4*e+4]
        c1, d1 = Tgt[2*i,:]
        c2, d2 = Tgt[2*i+1, :]
        # val[2*i] = (a1*d - b1*c)/np.sqrt(a1*a1+b1*b1)
        # dv = (d*b1+c*a1) / (a1*a1+b1*b1)**1.5
        # rowJ[4*i],   colJ[4*i],   valJ[4*i]   = 2*i, 4*e, b1*dv
        # rowJ[4*i+1], colJ[4*i+1], valJ[4*i+1] = 2*i, 4*e+1, -a1*dv
        
        # val[2*i+1] = (a2*d - b2*c) / np.sqrt(a2*a2 + b2*b2)
        # dv = (d*b2+c*a2) / (a2*a2+b2*b2)**1.5
        # rowJ[4*i+2], colJ[4*i+2], valJ[4*i+2] = 2*i+1, 4*e+2, b2*dv
        # rowJ[4*i+3], colJ[4*i+3], valJ[4*i+3] = 2*i+1, 4*e+3, -a2*dv

        val[2*i] = a1*d1 - b1*c1
        rowJ[4*i],   colJ[4*i],   valJ[4*i]   = 2*i, 4*e, d1
        rowJ[4*i+1], colJ[4*i+1], valJ[4*i+1] = 2*i, 4*e+1, -c1
        
        val[2*i+1] = a2*d2 - b2*c2
        rowJ[4*i+2], colJ[4*i+2], valJ[4*i+2] = 2*i+1, 4*e+2, d2
        rowJ[4*i+3], colJ[4*i+3], valJ[4*i+3] = 2*i+1, 4*e+3, -c2

    return val, valJ, rowJ, colJ


@jit(nopython=True, parallel=True, cache=True)
def soft_feat_ff_noJ(X, I, Tgt):
    n = I.shape[0]
    val = np.zeros(2*n, dtype=np.float64)
    for i in prange(n):
        ind = I[i]
        val[2*i] = X[ind] - Tgt[i,0]
        val[2*i+1] = X[ind+1] - Tgt[i,1]
    return val

@jit(nopython=True, parallel=True, cache=True)
def soft_feat_ff(X, I, Tgt):
    n = I.shape[0]
    val = np.zeros(2*n, dtype=np.float64)
    valJ = np.zeros(2*n, dtype=np.float64)
    rowJ = np.zeros(2*n, dtype=np.int32)
    colJ = np.zeros(2*n, dtype=np.int32)
    for i in prange(n):
        ind = I[i]
        val[2*i] = X[ind] - Tgt[i,0]
        rowJ[2*i], colJ[2*i], valJ[2*i] = 2*i, ind, 1
        val[2*i+1] = X[ind+1] - Tgt[i,1]
        rowJ[2*i+1], colJ[2*i+1], valJ[2*i+1] = 2*i+1, ind+1, 1
    return val, valJ, rowJ, colJ


@jit(nopython=True, parallel=True, cache=True)
def curv_ff_noJ(X, Tgt, n, sepFF):
    val = np.zeros(2*n, dtype=np.float64)
    for i in prange(2*n):
        val[i] = X[sepFF + i] - Tgt[i]
    return val

@jit(nopython=True, parallel=True, cache=True)
def curv_ff(X, Tgt, n, sepFF):
    val = np.zeros(2*n, dtype=np.float64)
    valJ = np.zeros(2*n, dtype=np.float64)
    rowJ = np.zeros(2*n, dtype=np.int32)
    colJ = np.zeros(2*n, dtype=np.int32)
    for i in prange(2*n):
        val[i] = X[sepFF + i] - Tgt[i]
        rowJ[i], colJ[i], valJ[i] = i, sepFF+i, 1
    return val, valJ, rowJ, colJ

######################################################################################################################################

@jit(nopython=True, parallel=True, cache=True)
def barrier_det_full_noJ(X : np.ndarray, I : np.ndarray, dets : np.ndarray, orient_t, singu_t):
    n = I.shape[0]
    val  = np.zeros(3*n , dtype=np.float64)
    for i in prange(n):
        iS, iE1, iE2 = I[i,:]
        S = np.array((X[iS], X[iS+1]))
        E1 = np.array((X[iE1], X[iE1+1]))
        E2 = np.array((X[iE2], X[iE2+1]))
        d0,d1,d2 = dets[i,:]
        det0 = E2[1] * E1[0] - E2[0] * E1[1]
        det1 = E1[0] * S[1] - E1[1] * S[0]
        det2 = E2[0] * S[1] - E2[1] * S[0]
        val[3*i] = barrier_neg( det0 / d0, t = orient_t) # det(E1, E2) > 0
        val[3*i+1] = barrier_neg(det1 / d1, t = singu_t) # det(E1, S)>0
        val[3*i+2] = barrier_neg(det2 / d2, t = singu_t) # det(E2, S)>0
    return val

@jit(nopython=True, parallel=True, cache=True)
def barrier_det_full(X : np.ndarray, I : np.ndarray, dets : np.ndarray, orient_t, singu_t):
    n = I.shape[0]
    val  = np.zeros(3*n , dtype=np.float64)
    valJ = np.zeros(12*n, dtype=np.float64)
    rowJ = np.zeros(12*n, dtype=np.int32)
    colJ = np.zeros(12*n, dtype=np.int32)
    for i in prange(n):
        iS, iE1, iE2 = I[i,:]
        S = np.array((X[iS], X[iS+1]))
        E1 = np.array((X[iE1], X[iE1+1]))
        E2 = np.array((X[iE2], X[iE2+1]))
        d0,d1,d2 = dets[i,:]

        # first : orientation. Quadrilaterals should not flip
        # det(E1, E2) > 0
        det0 = E2[1] * E1[0] - E2[0] * E1[1]
        val[3*i] = barrier_neg( det0 / d0, t = orient_t)
        dev0 = barrier_neg_prime( det0 / d0, t = orient_t) / d0
        rowJ[12*i],   colJ[12*i],   valJ[12*i]   = 3*i, iE1,    E2[1] * dev0
        rowJ[12*i+1], colJ[12*i+1], valJ[12*i+1] = 3*i, iE1+1, -E2[0] * dev0
        rowJ[12*i+2], colJ[12*i+2], valJ[12*i+2] = 3*i, iE2,   -E1[1] * dev0
        rowJ[12*i+3], colJ[12*i+3], valJ[12*i+3] = 3*i, iE2+1,  E1[0] * dev0

        # second : singularity position. Center point should stay inside triangle
        # det(E1, S)>0
        det1 = E1[0] * S[1] - E1[1] * S[0]
        val[3*i+1] = barrier_neg(det1 / d1, t = singu_t)
        dev1 = barrier_neg_prime(det1 / d1, t = singu_t) / d1
        rowJ[12*i+4], colJ[12*i+4], valJ[12*i+4] = 3*i+1, iE1,    S[1] * dev1
        rowJ[12*i+5], colJ[12*i+5], valJ[12*i+5] = 3*i+1, iE1+1, -S[0] * dev1
        rowJ[12*i+6], colJ[12*i+6], valJ[12*i+6] = 3*i+1, iS,    -E1[1] * dev1
        rowJ[12*i+7], colJ[12*i+7], valJ[12*i+7] = 3*i+1, iS+1,   E1[0] * dev1

        # det(E2, S)>0
        det2 = E2[0] * S[1] - E2[1] * S[0]
        val[3*i+2] = barrier_neg(det2 / d2, t = singu_t)
        dev2 = barrier_neg_prime(det2 / d2, t = singu_t) / d2
        rowJ[12*i+8], colJ[12*i+8], valJ[12*i+8] = 3*i+2, iE2,    S[1] * dev2
        rowJ[12*i+9], colJ[12*i+9], valJ[12*i+9] = 3*i+2, iE2+1, -S[0] * dev2
        rowJ[12*i+10], colJ[12*i+10], valJ[12*i+10] = 3*i+2, iS,    -E2[1] * dev2
        rowJ[12*i+11], colJ[12*i+11], valJ[12*i+11] = 3*i+2, iS+1,   E2[0] * dev2

    return val, valJ, rowJ, colJ

@jit(nopython=True, parallel=True, cache=True)
def barrier_det_corner_noJ(X : np.ndarray, I : np.ndarray, dets : np.ndarray, orient_t, singu_t):
    n = I.shape[0]
    val  = np.zeros(n , dtype=np.float64)
    for i in prange(n):
        _, iE1, iE2 = I[i,:]
        E1 = np.array((X[iE1], X[iE1+1]))
        E2 = np.array((X[iE2], X[iE2+1]))
        d0 = dets[i,0]
        det0 = E2[1] * E1[0] - E2[0] * E1[1] # det(E1, E2) > 0
        val[i] = barrier_neg( det0 / d0, t = orient_t)
    return val

@jit(nopython=True, parallel=True, cache=True)
def barrier_det_corner(X : np.ndarray, I : np.ndarray, dets : np.ndarray, orient_t, singu_t):
    n = I.shape[0]
    val  = np.zeros(n , dtype=np.float64)
    valJ = np.zeros(4*n, dtype=np.float64)
    rowJ = np.zeros(4*n, dtype=np.int32)
    colJ = np.zeros(4*n, dtype=np.int32)
    for i in prange(n):
        _, iE1, iE2 = I[i,:]
        E1 = np.array((X[iE1], X[iE1+1]))
        E2 = np.array((X[iE2], X[iE2+1]))
        d0 = dets[i,0]
        det0 = E2[1] * E1[0] - E2[0] * E1[1] # det(E1, E2) > 0
        val[i] = barrier_neg( det0 / d0, t = orient_t)
        dev0 = barrier_neg_prime( det0 / d0, t = orient_t) / d0
        rowJ[4*i],   colJ[4*i],   valJ[4*i]   = i, iE1,    E2[1] * dev0
        rowJ[4*i+1], colJ[4*i+1], valJ[4*i+1] = i, iE1+1, -E2[0] * dev0
        rowJ[4*i+2], colJ[4*i+2], valJ[4*i+2] = i, iE2,   -E1[1] * dev0
        rowJ[4*i+3], colJ[4*i+3], valJ[4*i+3] = i, iE2+1,  E1[0] * dev0
    return val, valJ, rowJ, colJ

######################################################################################################################################
##### Distortions ##### 

@jit(nopython=True, parallel=True, cache=True)
def distortion_lscm_noJ(X : np.ndarray, I : np.ndarray, initX : np.ndarray):
    """Quads in chart should be a conformal deformation of initial quads"""
    N = I.shape[0]
    val  = np.zeros(2*N , dtype=np.float64)
    for i in prange(N):
        iS,ie1,ie2 = I[i,:]
        xs, ys, us, vs = initX[iS], initX[iS+1], X[iS], X[iS+1]
        x1, y1, u1, v1 = initX[ie1], initX[ie1+1], X[ie1], X[ie1+1]
        x2, y2, u2, v2 = initX[ie2], initX[ie2+1], X[ie2], X[ie2+1]
        area = (x1*y2 - x2*y1) /2
        val[2*i] = (y2*u1 - y1 * u2 + x2 * v1 - x1 * v2) / area      
        val[2*i+1] = (-x2*u1 + x1*u2 +y2*v1 - y1*v2) / area
    return val

@jit(nopython=True, parallel=True, cache=True)
def distortion_lscm(X : np.ndarray, I : np.ndarray, initX : np.ndarray):
    """Quads in chart should be a conformal deformation of initial quads"""
    N = I.shape[0]
    val  = np.zeros(2*N , dtype=np.float64)
    valJ = np.zeros(8*N, dtype=np.float64)
    rowJ = np.zeros(8*N, dtype=np.int32)
    colJ = np.zeros(8*N, dtype=np.int32)
    for i in prange(N):
        iS,ie1,ie2 = I[i,:]
        # xs, ys, us, vs = initX[iS], initX[iS+1], X[iS], X[iS+1]
        x1, y1, u1, v1 = initX[ie1], initX[ie1+1], X[ie1], X[ie1+1]
        x2, y2, u2, v2 = initX[ie2], initX[ie2+1], X[ie2], X[ie2+1]

        area = (x1*y2 - x2*y1) /2
        val[2*i] = (y2*u1 - y1 * u2 + x2 * v1 - x1 * v2) / area
        rowJ[8*i],   colJ[8*i],   valJ[8*i]   = 2*i, ie1, y2/area
        rowJ[8*i+1], colJ[8*i+1], valJ[8*i+1] = 2*i, ie1+1, x2/area
        rowJ[8*i+2], colJ[8*i+2], valJ[8*i+2] = 2*i, ie2, -y1/area
        rowJ[8*i+3], colJ[8*i+3], valJ[8*i+3] = 2*i, ie2+1, -x1/area
        
        val[2*i+1] = (-x2*u1 + x1*u2 +y2*v1 - y1*v2) / area
        rowJ[8*i+4], colJ[8*i+4], valJ[8*i+4] = 2*i+1, ie1, -x2/area
        rowJ[8*i+5], colJ[8*i+5], valJ[8*i+5] = 2*i+1, ie1+1, y2/area
        rowJ[8*i+6], colJ[8*i+6], valJ[8*i+6] = 2*i+1, ie2, x1/area
        rowJ[8*i+7], colJ[8*i+7], valJ[8*i+7] = 2*i+1, ie2+1, -y1/area
    return val, valJ, rowJ, colJ

@jit(nopython=True, parallel=True, cache=True)
def distortion_isometric_noJ(X : np.ndarray, I : np.ndarray, jacs : np.ndarray):
    """Forces for each quad of each chart: JtJ = Id"""
    n = I.shape[0]
    val  = np.zeros(3*n , dtype=np.float64)
    for i in prange(n):
        _,ie0,ie1 = I[i,:]
        ref_mat = jacs[i, :, :]
        u0,v0 = X[ie0], X[ie0+1]
        u1,v1 = X[ie1], X[ie1+1]
        a,b,c,d = ref_mat[0,0], ref_mat[0,1], ref_mat[1,0], ref_mat[1,1]
        acu = (a*u0+c*u1)
        acv = (a*v0+c*v1)
        bdu = (b*u0+d*u1)
        bdv = (b*v0+d*v1)
        val[3*i] = acu*acu + acv*acv - 1
        val[3*i+1] = bdu*bdu + bdv*bdv - 1
        val[3*i+2] = 2*(acu*bdu + acv*bdv)
    return val

@jit(nopython=True, parallel=True, cache=True)
def distortion_isometric(X : np.ndarray, I : np.ndarray, jacs : np.ndarray):
    """Forces for each quad of each chart: JtJ = Id"""
    n = I.shape[0]
    val  = np.zeros(3*n , dtype=np.float64)
    valJ = np.zeros(12*n, dtype=np.float64)
    rowJ = np.zeros(12*n, dtype=np.int32)
    colJ = np.zeros(12*n, dtype=np.int32)

    for i in prange(n):
        _,ie0,ie1 = I[i,:]
        ref_mat = jacs[i, :, :]
        u0,v0 = X[ie0], X[ie0+1]
        u1,v1 = X[ie1], X[ie1+1]
        a,b,c,d = ref_mat[0,0], ref_mat[0,1], ref_mat[1,0], ref_mat[1,1]

        acu = (a*u0+c*u1)
        acv = (a*v0+c*v1)
        bdu = (b*u0+d*u1)
        bdv = (b*v0+d*v1)

        val[3*i] = acu*acu + acv*acv - 1
        rowJ[12*i],   colJ[12*i],   valJ[12*i]   = 3*i, ie0,   2*a*acu
        rowJ[12*i+1], colJ[12*i+1], valJ[12*i+1] = 3*i, ie0+1, 2*a*acv
        rowJ[12*i+2], colJ[12*i+2], valJ[12*i+2] = 3*i, ie1,   2*c*acu
        rowJ[12*i+3], colJ[12*i+3], valJ[12*i+3] = 3*i, ie1+1, 2*c*acv

        val[3*i+1] = bdu*bdu + bdv*bdv - 1
        rowJ[12*i+4], colJ[12*i+4], valJ[12*i+4] = 3*i+1, ie0,   2*b*bdu
        rowJ[12*i+5], colJ[12*i+5], valJ[12*i+5] = 3*i+1, ie0+1, 2*b*bdv
        rowJ[12*i+6], colJ[12*i+6], valJ[12*i+6] = 3*i+1, ie1,   2*d*bdu
        rowJ[12*i+7], colJ[12*i+7], valJ[12*i+7] = 3*i+1, ie1+1, 2*d*bdv
        
        val[3*i+2] = 2*(acu*bdu + acv*bdv)
        rowJ[12*i+8], colJ[12*i+8], valJ[12*i+8] = 3*i+2, ie0,   2*(a*bdu + b*acu)
        rowJ[12*i+9], colJ[12*i+9], valJ[12*i+9] = 3*i+2, ie0+1, 2*(a*bdv + b*acv)
        rowJ[12*i+10], colJ[12*i+10], valJ[12*i+10] = 3*i+2, ie1,   2*(c*bdu + d*acu)
        rowJ[12*i+11], colJ[12*i+11], valJ[12*i+11] = 3*i+2, ie1+1, 2*(c*bdv + d*acv)
    return val,valJ,rowJ,colJ

@jit(nopython=True, parallel=True, cache=True)
def distortion_shear_noJ(X : np.ndarray, I : np.ndarray, jacs : np.ndarray):
    """Forces only the third term of isometric energy"""
    n = I.shape[0]
    val  = np.zeros(n , dtype=np.float64)
    for i in prange(n):
        _,ie0,ie1 = I[i,:]
        ref_mat = jacs[i, :, :]
        u0,v0 = X[ie0], X[ie0+1]
        u1,v1 = X[ie1], X[ie1+1]
        a,b,c,d = ref_mat[0,0], ref_mat[0,1], ref_mat[1,0], ref_mat[1,1]
        val[i] = ((a*u0+c*u1)*(b*u0+d*u1) + (a*v0+c*v1)*(b*v0+d*v1))
    return val


@jit(nopython=True, parallel=True, cache=True)
def distortion_shear(X : np.ndarray, I : np.ndarray, jacs : np.ndarray):
    """Forces only the third term of isometric energy"""
    n = I.shape[0]
    val  = np.zeros(n , dtype=np.float64)
    valJ = np.zeros(4*n, dtype=np.float64)
    rowJ = np.zeros(4*n, dtype=np.int32)
    colJ = np.zeros(4*n, dtype=np.int32)

    for i in prange(n):
        _,ie0,ie1 = I[i,:]
        ref_mat = jacs[i, :, :]
        u0,v0 = X[ie0], X[ie0+1]
        u1,v1 = X[ie1], X[ie1+1]
        a,b,c,d = ref_mat[0,0], ref_mat[0,1], ref_mat[1,0], ref_mat[1,1]
        acu = (a*u0+c*u1)
        acv = (a*v0+c*v1)
        bdu = (b*u0+d*u1)
        bdv = (b*v0+d*v1)
        val[i] = (acu*bdu + acv*bdv)
        rowJ[4*i],   colJ[4*i],   valJ[4*i]   = i, ie0,   (a*bdu + b*acu)
        rowJ[4*i+1], colJ[4*i+1], valJ[4*i+1] = i, ie0+1, (a*bdv + b*acv)
        rowJ[4*i+2], colJ[4*i+2], valJ[4*i+2] = i, ie1,   (c*bdu + d*acu)
        rowJ[4*i+3], colJ[4*i+3], valJ[4*i+3] = i, ie1+1, (c*bdv + d*acv)
    return val,valJ,rowJ,colJ

@jit(nopython=True, parallel=True, cache=True)
def distortion_det_noJ(X : np.ndarray, I : np.ndarray, dets : np.ndarray):
    n = I.shape[0]
    val  = np.zeros(n , dtype=np.float64)
    for i in prange(n):
        iS, iE1, iE2 = I[i,:]
        E1 = np.array((X[iE1], X[iE1+1]))
        E2 = np.array((X[iE2], X[iE2+1]))
        d0 = dets[i,0]
        det0 = E2[1] * E1[0] - E2[0] * E1[1]
        val[i] = det0 / d0 - 1 # det(E1, E2)
    return val

@jit(nopython=True, parallel=True, cache=True)
def distortion_det(X : np.ndarray, I : np.ndarray, dets : np.ndarray):
    n = I.shape[0]
    val  = np.zeros(n , dtype=np.float64)
    valJ = np.zeros(4*n, dtype=np.float64)
    rowJ = np.zeros(4*n, dtype=np.int32)
    colJ = np.zeros(4*n, dtype=np.int32)
    for i in prange(n):
        iS, iE1, iE2 = I[i,:]
        E1 = np.array((X[iE1], X[iE1+1]))
        E2 = np.array((X[iE2], X[iE2+1]))
        d0 = dets[i,0]
        # Quadrilaterals should not flip
        # det(E1, E2) > 0
        det0 = E2[1] * E1[0] - E2[0] * E1[1]
        val[i] = det0 / d0 -1
        rowJ[4*i],   colJ[4*i],   valJ[4*i]   = i, iE1,    E2[1] / d0
        rowJ[4*i+1], colJ[4*i+1], valJ[4*i+1] = i, iE1+1, -E2[0] / d0
        rowJ[4*i+2], colJ[4*i+2], valJ[4*i+2] = i, iE2,   -E1[1] / d0
        rowJ[4*i+3], colJ[4*i+3], valJ[4*i+3] = i, iE2+1,  E1[0] / d0
    return val, valJ, rowJ, colJ

@jit(nopython=True, parallel=True, cache=True)
def distortion_det3_noJ(X : np.ndarray, I : np.ndarray, dets : np.ndarray):
    n = I.shape[0]
    val  = np.zeros(3*n , dtype=np.float64)
    for i in prange(n):
        iS, iE1, iE2 = I[i,:]
        S = np.array((X[iS], X[iS+1]))
        E1 = np.array((X[iE1], X[iE1+1]))
        E2 = np.array((X[iE2], X[iE2+1]))
        d0,d1,d2 = dets[i,:]
        det0 = E2[1] * E1[0] - E2[0] * E1[1]
        det1 = E1[0] * S[1] - E1[1] * S[0]
        det2 = E2[0] * S[1] - E2[1] * S[0]
        val[3*i] = det0 / d0 - 1 # det(E1, E2)
        val[3*i+1] = det1 / d1 - 1 # det(E1, S)
        val[3*i+2] = det2 / d2 -1 # det(E2, S)
    return val

@jit(nopython=True, parallel=True, cache=True)
def distortion_det3(X : np.ndarray, I : np.ndarray, dets : np.ndarray):
    n = I.shape[0]
    val  = np.zeros(3*n , dtype=np.float64)
    valJ = np.zeros(12*n, dtype=np.float64)
    rowJ = np.zeros(12*n, dtype=np.int32)
    colJ = np.zeros(12*n, dtype=np.int32)
    for i in prange(n):
        iS, iE1, iE2 = I[i,:]
        S = np.array((X[iS], X[iS+1]))
        E1 = np.array((X[iE1], X[iE1+1]))
        E2 = np.array((X[iE2], X[iE2+1]))
        d0,d1,d2 = dets[i,:]

        # first : orientation. Quadrilaterals should not flip
        # det(E1, E2) > 0
        det0 = E2[1] * E1[0] - E2[0] * E1[1]
        val[3*i] = det0 / d0 -1
        rowJ[12*i],   colJ[12*i],   valJ[12*i]   = 3*i, iE1,    E2[1] / d0
        rowJ[12*i+1], colJ[12*i+1], valJ[12*i+1] = 3*i, iE1+1, -E2[0] / d0
        rowJ[12*i+2], colJ[12*i+2], valJ[12*i+2] = 3*i, iE2,   -E1[1] / d0
        rowJ[12*i+3], colJ[12*i+3], valJ[12*i+3] = 3*i, iE2+1,  E1[0] / d0

        # second : singularity position. Center point should stay inside triangle
        # det(E1, S)>0
        det1 = E1[0] * S[1] - E1[1] * S[0]
        val[3*i+1] = det1 / d1 - 1
        rowJ[12*i+4], colJ[12*i+4], valJ[12*i+4] = 3*i+1, iE1,    S[1] / d1
        rowJ[12*i+5], colJ[12*i+5], valJ[12*i+5] = 3*i+1, iE1+1, -S[0] / d1
        rowJ[12*i+6], colJ[12*i+6], valJ[12*i+6] = 3*i+1, iS,    -E1[1] / d1
        rowJ[12*i+7], colJ[12*i+7], valJ[12*i+7] = 3*i+1, iS+1,   E1[0] / d1

        # det(E2, S)>0
        det2 = E2[0] * S[1] - E2[1] * S[0]
        val[3*i+2] = det2 / d2 -1
        rowJ[12*i+8], colJ[12*i+8], valJ[12*i+8] = 3*i+2, iE2,    S[1] / d2
        rowJ[12*i+9], colJ[12*i+9], valJ[12*i+9] = 3*i+2, iE2+1, -S[0] / d2
        rowJ[12*i+10], colJ[12*i+10], valJ[12*i+10] = 3*i+2, iS,    -E2[1] / d2
        rowJ[12*i+11], colJ[12*i+11], valJ[12*i+11] = 3*i+2, iS+1,   E2[0] / d2

    return val, valJ, rowJ, colJ

@jit(nopython=True, parallel=True, cache=True)
def rotation_norm_noJ(X,n,sepRot):
    return X[sepRot:sepRot+n].astype(np.float64)

@jit(nopython=True, parallel=True, cache=True)
def rotation_norm(X,n,sepRot):
    val = X[sepRot:sepRot+n].astype(np.float64)
    valJ = np.ones_like(val, dtype=np.float64)
    rowJ = np.array(range(n), dtype=np.int32)
    colJ = np.array(range(sepRot, sepRot+n), dtype=np.int32)
    return val, valJ, rowJ, colJ