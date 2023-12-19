from .common import *
import numpy as np
import cmath
from math import pi, log, atan, cos, sin
from numba import jit, prange

##########

@jit(nopython=True, parallel=True, cache=True)
def constraint_edge_fixedFF_noJ(X : np.ndarray, I, E, PT, sepRot:int):
    """Edges should match up to the rotation between adjacent triangles. Version where angles and the frame field are fixed"""
    n = I.shape[0]
    val = np.zeros(2*n, dtype=np.float64)
    for i in prange(n):
        e, T1, T2 = I[i, :]
        w = 2*atan(X[sepRot+e])
        a = PT[i]
        cw,sw = cos(a - w), sin(a - w)
        e1,e2 = E[i,0,:], E[i,1,:]
        a1,b1,c1,d1 = X[4*T1: 4*(T1 + 1)] # jac of T1
        a2,b2,c2,d2 = X[4*T2: 4*(T2 + 1)] # jac of T2
        val[2*i] = e1[0] * a1 + e1[1] * c1 - cw*e2[0]*a2 - cw*e2[1]*c2 - sw*e2[0]*b2 - sw*e2[1]*d2
        val[2*i+1] = e1[0] * b1 + e1[1] * d1 + sw*e2[0]*a2 + sw*e2[1]*c2 - cw*e2[0]*b2 - cw*e2[1]*d2
    return val

@jit(nopython=True, parallel=True, cache=True)
def constraint_edge_fixedFF(X : np.ndarray, I, E, PT, sepRot:int):
    """Edges should match up to the rotation between adjacent triangles. Version where angles and the frame field are fixed"""
    n = I.shape[0]
    val = np.zeros(2*n, dtype=np.float64)
    valJ = np.zeros(12*n, dtype=np.float64)
    rowJ = np.zeros(12*n, dtype=np.int32)
    colJ = np.zeros(12*n, dtype=np.int32)

    for i in prange(n):
        e, T1, T2 = I[i, :]
        w = 2*atan(X[sepRot+e])
        a = PT[i]
        cw,sw = cos(a - w), sin(a - w)

        e1,e2 = E[i,0,:], E[i,1,:]
        a1,b1,c1,d1 = X[4*T1: 4*(T1 + 1)] # jac of T1
        a2,b2,c2,d2 = X[4*T2: 4*(T2 + 1)] # jac of T2
        # a c
        # b d

        val[2*i] = e1[0] * a1 + e1[1] * c1 - cw*e2[0]*a2 - cw*e2[1]*c2 - sw*e2[0]*b2 - sw*e2[1]*d2
        rowJ[12*i],   colJ[12*i],   valJ[12*i]   = 2*i, 4*T1,    e1[0] # a1
        rowJ[12*i+1], colJ[12*i+1], valJ[12*i+1] = 2*i, 4*T1+2,  e1[1] # c1
        rowJ[12*i+2], colJ[12*i+2], valJ[12*i+2] = 2*i, 4*T2,   -cw*e2[0]# a2
        rowJ[12*i+3], colJ[12*i+3], valJ[12*i+3] = 2*i, 4*T2+1, -sw*e2[0] # b2
        rowJ[12*i+4], colJ[12*i+4], valJ[12*i+4] = 2*i, 4*T2+2, -cw*e2[1] # c2
        rowJ[12*i+5], colJ[12*i+5], valJ[12*i+5] = 2*i, 4*T2+3, -sw*e2[1] # d2

        val[2*i+1] = e1[0] * b1 + e1[1] * d1 + sw*e2[0]*a2 + sw*e2[1]*c2 - cw*e2[0]*b2 - cw*e2[1]*d2
        rowJ[12*i+6],  colJ[12*i+6],  valJ[12*i+6]  = 2*i+1, 4*T1+1,  e1[0] # b1
        rowJ[12*i+7],  colJ[12*i+7],  valJ[12*i+7]  = 2*i+1, 4*T1+3,  e1[1] # d1
        rowJ[12*i+8],  colJ[12*i+8],  valJ[12*i+8]  = 2*i+1, 4*T2,    sw*e2[0]# a2
        rowJ[12*i+9],  colJ[12*i+9],  valJ[12*i+9]  = 2*i+1, 4*T2+1, -cw*e2[0] # b2
        rowJ[12*i+10], colJ[12*i+10], valJ[12*i+10] = 2*i+1, 4*T2+2,  sw*e2[1] # c2
        rowJ[12*i+11], colJ[12*i+11], valJ[12*i+11] = 2*i+1, 4*T2+3, -cw*e2[1] # d2
    return val, valJ, rowJ, colJ

##########

@jit(nopython=True, parallel=True, cache=True)
def constraint_edge_noJ(X, I, E, PT, sepRot):
    """Edges should match up to the rotation they carry. Version where rotations are considered as variables"""
    n = I.shape[0]
    val = np.zeros(2*n, dtype=np.float64)
    for i in prange(n):
        e, T1, T2 = I[i, :]
        iw = sepRot + e
        w = X[iw]
        a = PT[i]
        pt = np.array([[cos(a), sin(a)], [-sin(a), cos(a)]]) # parallel transport rotation matrix

        a1,b1,c1,d1 = X[4*T1 : 4*(T1 + 1)]
        a2,b2,c2,d2 = X[4*T2 : 4*(T2 + 1)]

        J1 = np.array([[a1,c1], [b1,d1]])
        J2 = np.array([[a2,c2], [b2,d2]])
        e1, e2 = E[i,0,:], E[i,1,:]

        wm = np.array([[1, w], [-w, 1]])
        wp = np.array([[1, -w], [w, 1]])

        en = wm @ J1 @ e1 - wp @ pt @ J2 @ e2
        val[2*i] = en[0]
        val[2*i+1] = en[1]
    return val

@jit(nopython=True, parallel=True, cache=True)
def constraint_edge(X, I, E, PT, sepRot):
    """Edges should match up to the rotation they carry. Version where rotations are considered as variables"""
    n = I.shape[0]
    val = np.zeros(2*n, dtype=np.float64)
    valJ = np.zeros(18*n, dtype=np.float64)
    rowJ = np.zeros(18*n, dtype=np.int32)
    colJ = np.zeros(18*n, dtype=np.int32)

    dJa = np.array([[1.,0.], [0.,0.]], dtype=np.float64)
    dJb = np.array([[0.,0.], [1.,0.]], dtype=np.float64)
    dJc = np.array([[0.,1.], [0.,0.]], dtype=np.float64)
    dJd = np.array([[0.,0.], [0.,1.]], dtype=np.float64)
    dw = np.array([[0.,1.], [-1.,0.]], dtype=np.float64)

    for i in prange(n):
        e, T1, T2 = I[i, :]
        iw = sepRot + e
        w = X[iw]
        a = PT[i]
        pt = np.array([[cos(a), sin(a)], [-sin(a), cos(a)]]) # parallel transport rotation matrix

        a1,b1,c1,d1 = X[4*T1 : 4*(T1 + 1)]
        a2,b2,c2,d2 = X[4*T2 : 4*(T2 + 1)]

        J1 = np.array([[a1,c1], [b1,d1]])
        J2 = np.array([[a2,c2], [b2,d2]])
        e1, e2 = E[i,0,:], E[i,1,:]

        wm = np.array([[1, w], [-w, 1]])
        wp = np.array([[1, -w], [w, 1]])

        en = wm @ J1 @ e1 - wp @ pt @ J2 @ e2
        val[2*i] = en[0]
        val[2*i+1] = en[1]

        dev_w = dw @ ( J1 @ e1 + pt @ J2 @ e2)
        rowJ[18*i], colJ[18*i], valJ[18*i] = 2*i, iw, dev_w[0]
        rowJ[18*i+1], colJ[18*i+1], valJ[18*i+1] = 2*i+1, iw, dev_w[1]

        dev_A1 = wm @ dJa @ e1
        rowJ[18*i+2], colJ[18*i+2], valJ[18*i+2] = 2*i, 4*T1, dev_A1[0]
        rowJ[18*i+3], colJ[18*i+3], valJ[18*i+3] = 2*i+1, 4*T1, dev_A1[1]

        dev_A2 = - wp @ pt @ dJa @ e2
        rowJ[18*i+4], colJ[18*i+4], valJ[18*i+4] = 2*i, 4*T2, dev_A2[0]
        rowJ[18*i+5], colJ[18*i+5], valJ[18*i+5] = 2*i+1, 4*T2, dev_A2[1]
        
        dev_B1 = wm @ dJb @ e1
        rowJ[18*i+6], colJ[18*i+6], valJ[18*i+6] = 2*i, 4*T1+1, dev_B1[0]
        rowJ[18*i+7], colJ[18*i+7], valJ[18*i+7] = 2*i+1, 4*T1+1, dev_B1[1]

        dev_B2 = -wp @ pt @ dJb @ e2 
        rowJ[18*i+8], colJ[18*i+8], valJ[18*i+8] = 2*i, 4*T2+1, dev_B2[0]
        rowJ[18*i+9], colJ[18*i+9], valJ[18*i+9] = 2*i+1, 4*T2+1, dev_B2[1]
        
        dev_C1 = wm @ dJc @ e1
        rowJ[18*i+10], colJ[18*i+10], valJ[18*i+10] = 2*i, 4*T1+2, dev_C1[0]
        rowJ[18*i+11], colJ[18*i+11], valJ[18*i+11] = 2*i+1, 4*T1+2, dev_C1[1]

        dev_C2 = - wp @ pt @ dJc @ e2
        rowJ[18*i+12], colJ[18*i+12], valJ[18*i+12] = 2*i, 4*T2+2, dev_C2[0]
        rowJ[18*i+13], colJ[18*i+13], valJ[18*i+13] = 2*i+1, 4*T2+2, dev_C2[1]
        
        dev_D1 = wm @ dJd @ e1
        rowJ[18*i+14], colJ[18*i+14], valJ[18*i+14] = 2*i, 4*T1+3, dev_D1[0]
        rowJ[18*i+15], colJ[18*i+15], valJ[18*i+15] = 2*i+1, 4*T1+3, dev_D1[1]

        dev_D2 = - wp @ pt @ dJd @ e2
        rowJ[18*i+16], colJ[18*i+16], valJ[18*i+16] = 2*i, 4*T2+3, dev_D2[0]
        rowJ[18*i+17], colJ[18*i+17], valJ[18*i+17] = 2*i+1, 4*T2+3, dev_D2[1]

    return val, valJ, rowJ, colJ

##########

@jit(nopython=True, parallel=True, cache=True)
def constraint_rotations_follows_ff_noJ(X,I,PT, sepRot):
    """Rotations along edges should match the rotation between the two frames (up to a symmetry of the square)"""
    n = I.shape[0]
    val = np.zeros(2*n, dtype=np.float64)
    for i in prange(n):
        e, iz1, iz2 = I[i,:]
        iw = sepRot+e
        w = X[iw]
        z1, z2 = complex(X[iz1], X[iz1+1]), complex(X[iz2], X[iz2+1]) # the ff representation
        a = PT[i]
        a = cmath.rect(1., 4*a)
        eiwp = complex(1, w) # positive
        eiwn = complex(1,-w) # negative
        val_e = (eiwp**4)*z2 - (eiwn**4)*a*z1
        val[2*i] = val_e.real
        val[2*i+1] = val_e.imag
    return val

@jit(nopython=True, parallel=True, cache=True)
def constraint_rotations_follows_ff(X,I,PT, sepRot):
    """Rotations along edges should match the rotation between the two frames (up to a symmetry of the square)"""
    n = I.shape[0]
    val = np.zeros(2*n, dtype=np.float64)
    valJ = np.zeros(10*n, dtype=np.float64)
    rowJ = np.zeros(10*n, dtype=np.int32)
    colJ = np.zeros(10*n, dtype=np.int32)

    for i in prange(n):
        e, iz1, iz2 = I[i,:]
        iw = sepRot+e
        w = X[iw]
        z1, z2 = complex(X[iz1], X[iz1+1]), complex(X[iz2], X[iz2+1]) # the ff representation
        a = PT[i]
        a = cmath.rect(1., 4*a)

        #  Cayley transform
        eiwp = complex(1, w) # positive
        eiwn = complex(1,-w) # negative
        val_e = (eiwp**4)*z2 - (eiwn**4)*a*z1
        
        val[2*i] = val_e.real
        val[2*i+1] = val_e.imag

        dev_w =  4j * ( (eiwp**3) * z2  + (eiwn**3) * a * z1)
        rowJ[10*i], colJ[10*i], valJ[10*i] = 2*i, iw, dev_w.real
        rowJ[10*i+1], colJ[10*i+1], valJ[10*i+1] = 2*i+1, iw, dev_w.imag

        dev_a = -(eiwn**4) * a
        rowJ[10*i+2], colJ[10*i+2], valJ[10*i+2] = 2*i, iz1, dev_a.real
        rowJ[10*i+3], colJ[10*i+3], valJ[10*i+3] = 2*i+1, iz1, dev_a.imag
        rowJ[10*i+4], colJ[10*i+4], valJ[10*i+4] = 2*i, iz1+1, -dev_a.imag
        rowJ[10*i+5], colJ[10*i+5], valJ[10*i+5] = 2*i+1, iz1+1, dev_a.real

        dev_b = eiwp**4
        rowJ[10*i+6], colJ[10*i+6], valJ[10*i+6] = 2*i, iz2, dev_b.real
        rowJ[10*i+7], colJ[10*i+7], valJ[10*i+7] = 2*i+1, iz2, dev_b.imag
        rowJ[10*i+8], colJ[10*i+8], valJ[10*i+8] = 2*i, iz2+1, -dev_b.imag
        rowJ[10*i+9], colJ[10*i+9], valJ[10*i+9] = 2*i+1, iz2+1, dev_b.real

    return val, valJ, rowJ, colJ

##########

@jit(nopython=True, parallel=True, cache=True)
def barrier_det_noJ(X, n, t):
    # n is number of faces
    val = np.zeros(n, dtype=np.float64)
    for i in prange(n):
        a,b,c,d = X[4*i:4*(i+1)]
        det = a*d-b*c
        v = barrier(det, t)
        val[i] = v
    return val

@jit(nopython=True, parallel=True, cache=True)
def barrier_det(X, n, t):
    """Barrier term to prevent triangles from flipping"""
    # n is number of faces
    val = np.zeros(n, dtype=np.float64)
    valJ = np.zeros(4*n, dtype=np.float64)
    rowJ = np.zeros(4*n, dtype=np.int32)
    colJ = np.zeros(4*n, dtype=np.int32)

    for i in prange(n):
        a,b,c,d = X[4*i:4*(i+1)]
        det = a*d-b*c
        v = barrier(det, t)
        dv = barrier_prime(det, t)
        val[i] = v
        rowJ[4*i],   colJ[4*i],   valJ[4*i]   = i, 4*i,    d*dv
        rowJ[4*i+1], colJ[4*i+1], valJ[4*i+1] = i, 4*i+1, -c*dv
        rowJ[4*i+2], colJ[4*i+2], valJ[4*i+2] = i, 4*i+2, -b*dv
        rowJ[4*i+3], colJ[4*i+3], valJ[4*i+3] = i, 4*i+3,  a*dv
    return val, valJ, rowJ, colJ

##### Distortions #####

@jit(nopython=True, parallel=True, cache=True)
def distortion_lscm_noJ(X, n):
    """UV and XYZ of triangles should minimize the Least-Square Conformal Map energy"""
    val = np.zeros(2*n, dtype=np.float64)
    for T in prange(n):
        a,b,c,d = X[4*T:4*(T+1)] # jacobian of T
        val[2*T] = a-d
        val[2*T+1] = b+c
    return val

@jit(nopython=True, parallel=True, cache=True)
def distortion_lscm(X, n):
    """UV and XYZ of triangles should minimize the Least-Square Conformal Map energy"""
    val = np.zeros(2*n, dtype=np.float64)
    valJ = np.zeros(4*n, dtype=np.float64)
    rowJ = np.zeros(4*n, dtype=np.int32)
    colJ = np.zeros(4*n, dtype=np.int32)
    for T in prange(n):
        ia,ib,ic,id = 4*T, 4*T+1, 4*T+2, 4*T+3
        a,b,c,d = X[4*T:4*(T+1)] # jacobian of T
        val[2*T] = a-d
        val[2*T+1] = b+c
        rowJ[4*T],   colJ[4*T],   valJ[4*T]   = 2*T, ia, 1
        rowJ[4*T+1], colJ[4*T+1], valJ[4*T+1] = 2*T, id, -1
        rowJ[4*T+2], colJ[4*T+2], valJ[4*T+2] = 2*T+1, ib, 1
        rowJ[4*T+3], colJ[4*T+3], valJ[4*T+3] = 2*T+1, ic, 1
    return val, valJ, rowJ, colJ

@jit(nopython=True, parallel=True, cache=True)
def distortion_id_noJ(X, n):
    """J = Id"""
    val = np.copy(X[:4*n])
    for T in prange(n):
        val[4*T] -= 1.
        val[4*T+3] -= 1.
    return val

@jit(nopython=True, parallel=True, cache=True)
def distortion_id(X, n):
    """J = Id"""
    val  = np.copy(X[:4*n])
    valJ = np.ones(4*n, dtype=np.float64)
    rowJ = np.arange(0,4*n).astype(np.int32)
    colJ = np.arange(0,4*n).astype(np.int32)
    for T in prange(n):
        val[4*T] -= 1.
        val[4*T+3] -= 1.
    return val, valJ, rowJ, colJ

@jit(nopython=True, parallel=True, cache=True)
def distortion_isometric_noJ(X, n):
    """JtJ = Id"""
    val = np.zeros(3*n, dtype=np.float64)
    for T in prange(n):
        a,b,c,d = X[4*T:4*(T+1)] # jacobian of T
        val[3*T] = a*a+b*b-1
        val[3*T+1] = c*c+d*d-1
        val[3*T+2] = a*c+b*d
    return val

@jit(nopython=True, parallel=True, cache=True)
def distortion_isometric(X, n):
    """JtJ = Id"""
    val = np.zeros(3*n, dtype=np.float64)
    valJ = np.zeros(8*n, dtype=np.float64)
    rowJ = np.zeros(8*n, dtype=np.int32)
    colJ = np.zeros(8*n, dtype=np.int32)
    for T in prange(n):
        ia,ib,ic,id = 4*T, 4*T+1, 4*T+2, 4*T+3
        a,b,c,d = X[4*T:4*(T+1)] # jacobian of T
        val[3*T] = a*a+b*b-1
        rowJ[8*T], colJ[8*T], valJ[8*T] = 3*T, ia, 2*a
        rowJ[8*T+1], colJ[8*T+1], valJ[8*T+1] = 3*T, ib, 2*b
        
        val[3*T+1] = c*c+d*d-1
        rowJ[8*T+2], colJ[8*T+2], valJ[8*T+2] = 3*T+1, ic, 2*c
        rowJ[8*T+3], colJ[8*T+3], valJ[8*T+3] = 3*T+1, id, 2*d
        
        val[3*T+2] = 2*(a*c+b*d)
        rowJ[8*T+4], colJ[8*T+4], valJ[8*T+4] = 3*T+2, ia, 2*c
        rowJ[8*T+5], colJ[8*T+5], valJ[8*T+5] = 3*T+2, ib, 2*d
        rowJ[8*T+6], colJ[8*T+6], valJ[8*T+6] = 3*T+2, ic, 2*a
        rowJ[8*T+7], colJ[8*T+7], valJ[8*T+7] = 3*T+2, id, 2*b
    return val, valJ, rowJ, colJ

jit(nopython=True, parallel=True, cache=True)
def distortion_det_noJ(X, n):
    # n is number of faces
    val = np.zeros(n, dtype=np.float64)
    for i in prange(n):
        a,b,c,d = X[4*i:4*(i+1)]
        val[i] = a*d-b*c - 1
    return val

@jit(nopython=True, parallel=True, cache=True)
def distortion_det(X, n):
    """Barrier term to prevent triangles from flipping"""
    # n is number of faces
    val = np.zeros(n, dtype=np.float64)
    valJ = np.zeros(4*n, dtype=np.float64)
    rowJ = np.zeros(4*n, dtype=np.int32)
    colJ = np.zeros(4*n, dtype=np.int32)

    for i in prange(n):
        a,b,c,d = X[4*i:4*(i+1)]
        det = a*d-b*c
        val[i] = det - 1
        rowJ[4*i],   colJ[4*i],   valJ[4*i]   = i, 4*i,    d
        rowJ[4*i+1], colJ[4*i+1], valJ[4*i+1] = i, 4*i+1, -c
        rowJ[4*i+2], colJ[4*i+2], valJ[4*i+2] = i, 4*i+2, -b
        rowJ[4*i+3], colJ[4*i+3], valJ[4*i+3] = i, 4*i+3,  a
    return val, valJ, rowJ, colJ