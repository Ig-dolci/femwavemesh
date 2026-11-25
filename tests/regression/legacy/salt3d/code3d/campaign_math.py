"""Reference-free controller rules and post-hoc trace metrics, units km/s/s.

These functions do not open files. Reference metrics are used only after meshes
have been generated; controller inputs are two-level estimates, never real errors.
"""
import numpy as np

def trace_errors(traces,reference,mask,norm_ratio_scope='masked'):
    u,r=np.asarray(traces),np.asarray(reference)
    b=np.asarray(mask,dtype=bool)
    if u.shape!=r.shape or b.shape!=u.shape or u.ndim!=3:
        raise ValueError('Expected matching [shot,sample,receiver] arrays')
    if not np.isfinite(u).all() or not np.isfinite(r).all():
        raise ValueError('Nonfinite pressure traces')
    denom=np.sum(np.where(b,r,0)**2,axis=(1,2))
    if np.any(denom<=0):
        raise ValueError('Empty reference window or zero reference norm')
    errors=np.sqrt(np.sum(np.where(b,u-r,0)**2,axis=(1,2))/denom)
    masked_ratios=np.sqrt(np.sum(np.where(b,u,0)**2,axis=(1,2))/denom)
    full_ratios=np.sqrt(np.sum(u*u,axis=(1,2))/np.sum(r*r,axis=(1,2)))
    if norm_ratio_scope not in ['full','masked']:
        raise ValueError('Norm ratio scope must be full or masked')
    ratios=full_ratios if norm_ratio_scope=='full' else masked_ratios
    return dict(per_source=errors.tolist(),worst=float(errors.max()),
                norm_ratios=ratios.tolist(),
                norm_ratios_full=full_ratios.tolist(),norm_ratios_masked=masked_ratios.tolist(),
                norm_ratio_scope=norm_ratio_scope,
                stop_and_ask=bool(np.any((ratios<.5)|(ratios>2))))

def measured_rate(na,ea,nb,eb):
    if min(na,ea,nb,eb)<=0 or nb<=na:
        raise ValueError('Expected positive errors and increasing DOFs')
    return float(3*np.log(ea/eb)/np.log(nb/na))

def bracket_dofs(dofs,errors,target=.05):
    n,e=np.asarray(dofs,dtype=float),np.asarray(errors,dtype=float)
    order=np.argsort(n)
    n,e=n[order],e[order]
    for i in range(len(n)-1):
        if e[i]>=target>=e[i+1] and e[i]!=e[i+1]:
            t=np.log(target/e[i])/np.log(e[i+1]/e[i])
            return dict(dofs=float(np.exp(np.log(n[i])+t*np.log(n[i+1]/n[i]))),
                        bracket_dofs=[float(n[i]),float(n[i+1])],extrapolated=False)
    return dict(dofs=None,reason='No measured bracket; no extrapolation for 5% DOF requirement')

def trajectory_error(dofs,errors,budget):
    n,e=np.asarray(dofs,dtype=float),np.asarray(errors,dtype=float)
    order=np.argsort(n)
    n,e=n[order],e[order]
    if len(n)<2 or np.any(n<=0) or np.any(e<=0) or np.any(np.diff(n)<=0):
        raise ValueError('Need distinct positive trajectory points')
    i=int(np.clip(np.searchsorted(n,budget)-1,0,len(n)-2))
    t=np.log(budget/n[i])/np.log(n[i+1]/n[i])
    return dict(error=float(np.exp(np.log(e[i])+t*np.log(e[i+1]/e[i]))),
                extrapolated=bool(budget<n[0] or budget>n[-1]))

def next_target(n,estimate,p,nmax,tol=.05,theta=.8):
    if estimate<=theta*tol:
        return dict(decision='accepted',target_dofs=None)
    if not .5<=p<=6:
        raise ValueError('Controller rate must be measured in [.5,6] or valid p0')
    growth=float(np.clip((estimate/(theta*tol))**(3/p),1.15,2))
    target=int(np.ceil(n*growth))
    return dict(decision='resource_limit' if target>nmax else 'refine',
                target_dofs=target,growth=growth,rate=p)

def size_law(alpha,h,rho,vp,q,ceq,freq=5):
    h,rho,vp=np.broadcast_arrays(h,rho,vp)
    if np.any(h<=0) or np.any(rho<0) or np.any(vp<=0):
        raise ValueError('Invalid size/density/speed')
    minimum=vp/(1.25*ceq*freq)
    if np.any(minimum>h*(1+1e-10)):
        raise ValueError('Inconsistent lower size bound exceeds current size')
    proposal=alpha*h**(q/(q+3))*np.maximum(rho,1e-300)**(-1/(q+3))
    return np.clip(proposal,minimum,h)
