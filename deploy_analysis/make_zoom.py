import glob, os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D="/tmp/deploy_plots"
bags=sorted({os.path.basename(f).replace("_policy.csv","") for f in glob.glob(D+"/*_policy.csv")})
AXIS={0:"a0=+X",1:"a1=+Y",2:"a2=+Z(down)",3:"a3=rotX",4:"a4=rotY",5:"a5"}
COL={0:"red",1:"orange",2:"green",3:"purple",4:"brown",5:"gray"}
def readcsv(p):
    rows=list(csv.DictReader(open(p)))
    return {k:np.array([float(r[k]) for r in rows]) for k in rows[0]}

fig,axes=plt.subplots(4,len(bags),figsize=(20,12),sharex="col")
for c,name in enumerate(bags):
    pol=readcsv(f"{D}/{name}_policy.csv"); frc=readcsv(f"{D}/{name}_force.csv")
    drop=float(open(f"{D}/{name}_drop.txt").read().strip())
    # policy window only: from policy start (t=0) to drop+0.3s
    t1=drop+0.4
    pm=pol["t"]<=t1; fm=(frc["t"]>=-0.5)&(frc["t"]<=t1)
    # row0 actions
    a=axes[0,c]
    for d in range(6):
        a.plot(pol["t"][pm],pol[f"a{d}"][pm],color=COL[d],lw=2.4 if d in(0,2) else 1.1,label=AXIS[d])
    a.axhline(1,color="k",ls=":",lw=0.7); a.axhline(-1,color="k",ls=":",lw=0.7)
    a.set_ylim(-1.15,1.15); a.set_title(name[-6:],weight="bold")
    if c==0: a.set_ylabel("actions"); a.legend(fontsize=7,ncol=2,loc="lower left")
    # row1 force mag + fz
    a=axes[1,c]
    a.plot(frc["t"][fm],frc["fmag"][fm],"k",lw=1.4,label="|F|")
    a.plot(frc["t"][fm],frc["fz"][fm],"g",lw=1.0,label="Fz")
    if c==0: a.set_ylabel("force N"); a.legend(fontsize=7)
    # row2 goal delta xyz + dist
    a=axes[2,c]
    a.plot(pol["t"][pm],pol["gdx"][pm],"r",lw=1.3,label="Δx")
    a.plot(pol["t"][pm],pol["gdy"][pm],color="orange",lw=1.3,label="Δy")
    a.plot(pol["t"][pm],pol["gdz"][pm],"g",lw=1.6,label="Δz")
    d=np.sqrt(pol["gdx"]**2+pol["gdy"]**2+pol["gdz"]**2)
    a.plot(pol["t"][pm],d[pm],"b--",lw=2,label="|dist|")
    a.axhline(0,color="k",lw=0.5)
    if c==0: a.set_ylabel("goal Δ mm"); a.legend(fontsize=7)
    # row3 just a0 and a2 big
    a=axes[3,c]
    a.plot(pol["t"][pm],pol["a0"][pm],"r",lw=2,label="a0 X")
    a.plot(pol["t"][pm],pol["a2"][pm],"g",lw=2,label="a2 Z down")
    a.axhline(0,color="k",lw=0.5); a.set_ylim(-1.1,1.1)
    a.set_xlabel("t (s)")
    if c==0: a.set_ylabel("a0/a2"); a.legend(fontsize=7)
    for r in range(4):
        axes[r,c].axvline(drop,color="red",ls="--",lw=1.6); axes[r,c].grid(alpha=0.25)
fig.suptitle("POLICY WINDOW ZOOM (per run): actions / force / goal-delta / a0+a2  — red=drop",weight="bold",fontsize=13)
fig.tight_layout(rect=[0,0,1,0.97])
fig.savefig(f"{D}/PLOT_ZOOM_grid.png",dpi=90); print("wrote PLOT_ZOOM_grid.png")
