"""Order-of-magnitude check that inverse Compton is negligible in the paper models.

Comoving energy densities along the jet, relative to U'_B = B'^2/8pi:
  U'_syn = L'_syn,cell / V'_cell * R/c   (same estimate as the code's Calc_Usyn; L'_syn from SEDe_syn_cuts.dat / delta^2)
  U'_RIAF = L_RIAF,bol / (4 pi z^2 c) * Gamma^2 (1-beta)^2   (RIAF as a point source behind the flow)
The IC/synchrotron cooling ratio, and the SSC/synchrotron luminosity ratio in the Thomson limit, are U'_rad/U'_B.

    python analysis/scripts/ic_estimate.py      -> analysis/runs/derived/ic_estimate.json
(run after run_models.py; reads analysis/runs/{riaf_m87,jet_riaf,jet_only})
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m87 import provenance  # noqa: E402
from m87.paths import RUNS  # noqa: E402
import numpy as np
out={}
c=2.998e10; Rg=6.674e-8*6.5e9*1.989e33/c**2
nu,L=np.loadtxt(RUNS/"riaf_m87/spectrum/lumThermal.dat",usecols=(0,8),unpack=True)
Lriaf=np.trapezoid(L/nu,nu)
print(f"L_RIAF,bol = {Lriaf:.2e} erg/s")
for run in ("jet_riaf","jet_only"):
    d=f"{RUNS}/{run}/"
    q=np.loadtxt(d+"quant_FF.dat")
    z,B,R,G,dl,V=q[:,0],q[:,1],q[:,2],q[:,4],q[:,5],q[:,7]
    E,Ls,seg=np.loadtxt(d+"SEDe_syn_cuts.dat",skiprows=1,unpack=True)
    seg=seg.astype(int); Ltot=np.zeros(len(z))
    for s in np.unique(seg):
        m=seg==s; e=E[m]; l=Ls[m]/e           # E*L -> L per energy (observer, x boost=delta^2)
        Ltot[s-1]=np.trapezoid(l,e)/dl[s-1]**2   # comoving
    UB=B**2/8/np.pi
    Usyn=Ltot/V*R/c
    # RIAF photons from behind, isotropic point source at origin: U' ~ L/(4 pi z^2 c) * Gamma^2 (1-beta)^2
    be=np.sqrt(1-1/G**2); Uriaf=Lriaf/(4*np.pi*z**2*c)*G**2*(1-be)**2
    print(f"\n{run}: z[Rg] B[G] Gamma Usyn'/UB' Uriaf'/UB'")
    for zz in (z[0]/Rg,100,300,1e3,3e3,1e4,1e5,1e6):
        i=np.argmin(abs(z/Rg-zz))
        print(f"{z[i]/Rg:9.3g} {B[i]:8.3g} {G[i]:6.2f} {Usyn[i]/UB[i]:9.2e} {Uriaf[i]/UB[i]:9.2e}")
    out[run]={"max_Usyn_over_UB":float((Usyn/UB).max()),"max_Uriaf_over_UB":float((Uriaf/UB).max())}
    print("max Usyn/UB:",(Usyn/UB).max(), " max Uriaf/UB:",(Uriaf/UB).max())
out["L_RIAF_bol"]=float(Lriaf)
(RUNS/"derived").mkdir(parents=True, exist_ok=True)
provenance.write(RUNS/"derived/ic_estimate.json", script="analysis/scripts/ic_estimate.py", summary=out)
