"""Charts of the sandbox cases. Run: python sandbox/figures.py <out_dir>"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from firesim.levelset import arrival
from firesim import mtt

out = sys.argv[1] if len(sys.argv) > 1 else '.'
os.makedirs(out, exist_ok=True)
fig,ax=plt.subplots(1,3,figsize=(15,4.6),dpi=150)
cell=5.;n=601;cy=n//2;cx=140
R=np.full((n,n),10.)
ext=[-cx*cell,(n-cx)*cell,-cy*cell,cy*cell]
th=np.linspace(0,2*np.pi,400)
for k,(lb,title) in enumerate(((1.,'No wind: should be a circle'),(3.,'Moderate wind: ellipse, length 3x width'),(6.,'Strong wind: ellipse, length 6x width'))):
    a,_=arrival(R,0.,lb,cell,[(cy,cx,0,0)],60.)
    e=np.sqrt(lb*lb-1)/lb
    am=mtt.arrival(R,0.,e,cell,[(cy,cx,0)],reach=3)
    ax[k].contourf(np.where(np.isfinite(a),a,np.nan),levels=[0,15,30,45,60],colors=['#7a1f12','#c8451f','#ee8a2c','#f9d27a'],extent=ext,origin='lower')
    for T in (15,30,45,60):
        r=10*T*(1-e)/(1-e*np.cos(th))
        ax[k].plot(r*np.cos(th),r*np.sin(th),'k--',lw=1,label='known answer' if T==60 else None)
    ax[k].contour(am,levels=[60],colors=['#2a6fdb'],linewidths=1.6,extent=ext,origin='lower')
    ax[k].plot([],[],color='#2a6fdb',lw=1.6,label='rejected method, 60 min')
    ax[k].plot(0,0,'w*',ms=10,mec='k')
    ax[k].set_title(title,fontsize=11); ax[k].set_aspect('equal'); ax[k].set_xlabel('metres downwind')
    if lb==1: ax[k].set_xlim(-650,650); ax[k].set_ylim(-650,650)
    else: ax[k].set_xlim(-80,680); ax[k].set_ylim(-330 if lb==3 else -200,330 if lb==3 else 200)
ax[0].set_ylabel('metres'); ax[2].legend(loc='upper left',fontsize=8,framealpha=.9)
fig.suptitle('FireSim spread solver on closed cases: colour bands are the fire every 15 minutes, dashed lines are the known answer',fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(out,'01_solver_closed_cases.png'))
