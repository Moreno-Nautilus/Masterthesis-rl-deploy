import glob, os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/tmp/deploy_plots"
bags = sorted({os.path.basename(f).replace("_policy.csv","") for f in glob.glob(D+"/*_policy.csv")})

AXIS = {0:"a0 = +X (lateral, toward/away socket)",1:"a1 = +Y (lateral sideways)",
        2:"a2 = +Z (DOWN into socket)",3:"a3 = rotX",4:"a4 = rotY",5:"a5 (unused)"}
COL  = {0:"red",1:"orange",2:"green",3:"purple",4:"brown",5:"gray"}

def readcsv(path):
    rows=list(csv.DictReader(open(path)))
    return {k:np.array([float(r[k]) for r in rows]) for k in rows[0]}

for name in bags:
    pol=readcsv(f"{D}/{name}_policy.csv")
    frc=readcsv(f"{D}/{name}_force.csv")
    drop=float(open(f"{D}/{name}_drop.txt").read().strip())

    fig,ax=plt.subplots(3,1,figsize=(13,10),sharex=True)
    fig.suptitle(f"{name}   —   DROP at t={drop:.2f}s (red dashed)",fontsize=13,weight="bold")

    # Panel 1: actions
    for d in range(6):
        lw = 2.6 if d==0 else 1.3
        ax[0].plot(pol["t"],pol[f"a{d}"],color=COL[d],lw=lw,label=AXIS[d])
    ax[0].axhline(1.0,color="k",ls=":",lw=0.8); ax[0].axhline(-1.0,color="k",ls=":",lw=0.8)
    ax[0].axhline(0.95,color="red",ls=":",lw=0.6,alpha=0.5)
    ax[0].set_ylabel("action (clip ±1)"); ax[0].set_ylim(-1.15,1.15)
    ax[0].legend(fontsize=8,loc="lower left",ncol=2); ax[0].set_title("Commanded actions (a0 = X bold)")

    # Panel 2: force
    ax[1].plot(frc["t"],frc["fmag"],color="black",lw=1.4,label="|F|")
    ax[1].plot(frc["t"],frc["fz"],color="green",lw=0.9,alpha=0.7,label="Fz")
    ax[1].plot(frc["t"],frc["fx"],color="red",lw=0.9,alpha=0.6,label="Fx")
    ax[1].plot(frc["t"],frc["fy"],color="orange",lw=0.9,alpha=0.6,label="Fy")
    ax[1].set_ylabel("force (N)"); ax[1].legend(fontsize=8,loc="upper left")
    ax[1].set_title("Contact force (note the 0→10N jump AT the drop, then frozen)")

    # Panel 3: goal delta (TCP->socket error)
    ax[2].plot(pol["t"],pol["gdx"],color="red",lw=1.3,label="goal Δx (mm)")
    ax[2].plot(pol["t"],pol["gdy"],color="orange",lw=1.3,label="goal Δy (mm)")
    ax[2].plot(pol["t"],pol["gdz"],color="green",lw=1.6,label="goal Δz (mm)")
    gdist=np.sqrt(pol["gdx"]**2+pol["gdy"]**2+pol["gdz"]**2)
    ax[2].plot(pol["t"],gdist,color="blue",lw=2.0,ls="--",label="|goal dist| (mm)")
    ax[2].axhline(0,color="k",lw=0.6)
    ax[2].set_ylabel("TCP→socket (mm)"); ax[2].set_xlabel("time (s)")
    ax[2].legend(fontsize=8,loc="upper left")
    ax[2].set_title("Goal delta = where the policy thinks the hole is relative to the tool")

    for a in ax:
        a.axvline(drop,color="red",ls="--",lw=1.8,alpha=0.8)
        a.grid(alpha=0.25)
    fig.tight_layout(rect=[0,0,1,0.97])
    out=f"{D}/PLOT_{name}.png"; fig.savefig(out,dpi=95); plt.close(fig)
    print("wrote",out)

# ---- summary overlay: a0 + |F| for all 4, drop aligned to t=0 ----
fig,ax=plt.subplots(2,1,figsize=(13,7),sharex=True)
fig.suptitle("ALL 4 RUNS aligned to drop (t=0): action a0(X) and |F|",weight="bold")
for name in bags:
    pol=readcsv(f"{D}/{name}_policy.csv"); frc=readcsv(f"{D}/{name}_force.csv")
    drop=float(open(f"{D}/{name}_drop.txt").read().strip())
    ax[0].plot(pol["t"]-drop,pol["a0"],lw=1.6,label=name[-6:])
    ax[1].plot(frc["t"]-drop,frc["fmag"],lw=1.3,label=name[-6:])
ax[0].axvline(0,color="red",ls="--"); ax[1].axvline(0,color="red",ls="--")
ax[0].axhline(1.0,color="k",ls=":"); ax[0].set_ylabel("a0 (X action)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)
ax[0].set_title("a0 = X-axis lateral action (1.0 = max)")
ax[1].set_ylabel("|F| (N)"); ax[1].set_xlabel("t relative to drop (s)"); ax[1].set_xlim(-4,1); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)
ax[1].set_title("|F|: the 0→10N spike lands exactly at t=0 in every run")
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(f"{D}/PLOT_SUMMARY_aligned.png",dpi=95); plt.close(fig)
print("wrote",f"{D}/PLOT_SUMMARY_aligned.png")
