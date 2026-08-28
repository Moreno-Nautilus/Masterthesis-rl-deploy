import glob, os, csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
D="/tmp/deploy_plots"
name=sys.argv[1]
AXIS={0:"a0=+X lateral",1:"a1=+Y lateral",2:"a2=+Z DOWN",3:"a3=rotX",4:"a4=rotY",5:"a5 unused"}
COL={0:"red",1:"orange",2:"green",3:"purple",4:"brown",5:"gray"}
def readcsv(p):
    rows=list(csv.DictReader(open(p)))
    return {k:np.array([float(r[k]) for r in rows]) for k in rows[0]}
pol=readcsv(f"{D}/{name}_policy.csv"); frc=readcsv(f"{D}/{name}_force.csv")
drop=float(open(f"{D}/{name}_drop.txt").read().strip())
t1=drop+0.5
pm=pol["t"]<=t1; fm=(frc["t"]>=-0.3)&(frc["t"]<=t1)
fig,ax=plt.subplots(3,1,figsize=(14,10),sharex=True)
fig.suptitle(f"{name}  policy window  —  DROP at t={drop:.2f}s (red)",weight="bold",fontsize=14)
for d in range(6):
    ax[0].plot(pol["t"][pm],pol[f"a{d}"][pm],color=COL[d],lw=2.6 if d in(0,2) else 1.3,label=AXIS[d])
ax[0].axhline(1,color="k",ls=":");ax[0].axhline(-1,color="k",ls=":");ax[0].axhline(0,color="k",lw=0.4)
ax[0].set_ylim(-1.15,1.15);ax[0].set_ylabel("action");ax[0].legend(fontsize=9,ncol=3);ax[0].set_title("Actions (a0=X, a2=Z-down bold)")
ax[1].plot(frc["t"][fm],frc["fmag"][fm],"k",lw=1.6,label="|F|")
ax[1].plot(frc["t"][fm],frc["fx"][fm],"r",lw=1,alpha=.7,label="Fx")
ax[1].plot(frc["t"][fm],frc["fy"][fm],color="orange",lw=1,alpha=.7,label="Fy")
ax[1].plot(frc["t"][fm],frc["fz"][fm],"g",lw=1.2,label="Fz")
ax[1].axhline(0,color="k",lw=0.4);ax[1].set_ylabel("force N");ax[1].legend(fontsize=9);ax[1].set_title("Force: 5N rest -> 0N unload -> 10N slam@drop")
ax[2].plot(pol["t"][pm],pol["gdx"][pm],"r",lw=1.5,label="goal Δx")
ax[2].plot(pol["t"][pm],pol["gdy"][pm],color="orange",lw=1.5,label="goal Δy")
ax[2].plot(pol["t"][pm],pol["gdz"][pm],"g",lw=1.8,label="goal Δz")
dist=np.sqrt(pol["gdx"]**2+pol["gdy"]**2+pol["gdz"]**2)
ax[2].plot(pol["t"][pm],dist[pm],"b--",lw=2.2,label="|goal dist|")
ax[2].axhline(0,color="k",lw=0.4);ax[2].set_ylabel("TCP→socket mm");ax[2].set_xlabel("t (s)")
ax[2].legend(fontsize=9);ax[2].set_title("Goal delta (how far/which way policy thinks hole is)")
for a in ax: a.axvline(drop,color="red",ls="--",lw=1.8);a.grid(alpha=0.25)
fig.tight_layout(rect=[0,0,1,0.96]);fig.savefig(f"{D}/ZOOM1_{name}.png",dpi=100);print("wrote",f"{D}/ZOOM1_{name}.png")
