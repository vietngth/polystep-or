import os, glob, json, numpy as np, torch
import exp_irp_polystep as B
CACHE=os.environ.get("CACHE","/media/anindex/Data/project-cache/ot-or-project")
REPO=os.path.join(CACHE,"InferOpt_DSIRP")
B.INSTDIR=os.path.join(REPO,"instances")
SOLDIR=os.path.join(REPO,"training/solutions/dagger/normal/penalty_200")
ID="normal-10_202212-1314-0055-0af411a5-e0e5-486d-9cbf-13fa3c8ece0a"
inst=B.load_instance(os.path.join(REPO,"instances",ID+".json"),"normal")
sols=sorted(glob.glob(os.path.join(SOLDIR,ID,"*_solutions.json")))
print("solutions files:",[os.path.basename(s) for s in sols], flush=True)
def load_valid(sols):
    for f in reversed(sols):
        p=json.load(open(f))["pctsp"]; bi=p.get("best_iteration")
        epochs=[k for k in p if k.isdigit()]
        if bi is not None and str(bi) in p and "weights" in p[str(bi)]:
            return p[str(bi)]["weights"], os.path.basename(f)
        if epochs and "weights" in p[max(epochs,key=int)]:
            return p[max(epochs,key=int)]["weights"], os.path.basename(f)
    return None,None
w,src=load_valid(sols); print("using:",src, flush=True)
wi=np.asarray(w["1"],float).flatten(); wp=np.asarray(w["2"],float).flatten()
print("w_inv[:3]",wi[:3].round(4),"| w_pen[:3]",wp[:3].round(4),"| sizes",wi.size,wp.size, flush=True)
print("init: w_inv=-1/72=%.4f  w_pen=+1/72=%.4f"%(-1/72,1/72), flush=True)
def evalp(pt,H=15):
    cs=[B.rollout(pt,inst,B.demand_seq(inst,"eval",s),horizon=H)[0] for s in range(5)]
    return float(np.mean(cs))
unt=B.PINN().to(B.DEV)
c_unt=evalp(lambda h,p: unt.theta(h,p))
wit=torch.tensor(wi,dtype=torch.float32); wpt=torch.tensor(wp,dtype=torch.float32)
c_toni=evalp(lambda h,p: B.theta_from_params(wit,wpt,h,p))
print("\nUNTRAINED held-out cost: %.1f"%c_unt, flush=True)
print("Toni-FY (1/1/1)   cost: %.1f   (%+.1f%% vs untrained)"%(c_toni,100*(c_unt-c_toni)/c_unt), flush=True)
