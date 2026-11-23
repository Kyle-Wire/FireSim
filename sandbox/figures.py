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


def synth_fig():
    """Synthetic landscapes: one example fire and the cell-size test (reads sandbox/results_synth.json)."""
    import json
    from firesim import synth
    from firesim.landscape import fields
    res = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_synth.json')))
    seed, cell = 11, 10.0
    ls = synth.landscape(seed); g = synth.on_grid(ls, cell)
    rng = np.random.default_rng(1000 + seed); wto = rng.uniform(0, 2 * np.pi)
    F = fields(g['fuel'], (.06, .07, .08, .60, .90), 6.0, wto, g['tan'], g['upslope'], g['cc'], g['ch'], g['cbh'], g['cbd'])
    b = ls['block']; nb = ls['fuel'].shape[0]; by, bx = np.mgrid[:nb, :nb]
    tx = ls['size'] / 2 - 1200 * np.cos(wto); ty = ls['size'] / 2 - 1200 * np.sin(wto)
    d = np.hypot((bx + .5) * b - tx, (by + .5) * b - ty); d[ls['fuel'] == 91] = 1e9
    iy, ix = np.unravel_index(d.argmin(), d.shape)
    r, c = int((iy + .5) * b / cell), int((ix + .5) * b / cell)
    a, _ = arrival(F['rate'], F['heading'], F['lb'], cell, [(r, c, 0, 15.0)], 60.0)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=150, gridspec_kw=dict(width_ratios=[1, 1.15]))
    names = {102: 'grass', 104: 'tall grass', 122: 'grass and shrub', 145: 'heavy shrub', 165: 'timber with understory', 183: 'light timber litter', 188: 'heavy timber litter', 91: 'no fuel (pavement, water)'}
    cols = {102: '#e9e3a6', 104: '#d6cc6d', 122: '#c2b280', 145: '#8f9e58', 165: '#5d8a5a', 183: '#a9c7a4', 188: '#3f6b45', 91: '#bdbdbd'}
    img = np.zeros(g['fuel'].shape + (3,))
    for k, v in cols.items():
        img[g['fuel'] == k] = matplotlib.colors.to_rgb(v)
    ext = [0, 4, 0, 4]
    ax[0].imshow(img, origin='lower', extent=ext)
    ax[0].contour(g['z'], levels=8, colors='k', linewidths=.3, alpha=.4, extent=ext)
    ax[0].contourf(np.where(a <= 60, a, np.nan), levels=[0, 15, 30, 45, 60], colors=['#7a1f12', '#c8451f', '#ee8a2c', '#f9d27a'], extent=ext, origin='lower', alpha=.9)
    ax[0].plot((c + .5) * cell / 1000, (r + .5) * cell / 1000, 'w*', ms=12, mec='k')
    ax[0].annotate('', xy=(0.5 + 0.35 * np.cos(wto), 3.5 + 0.35 * np.sin(wto)), xytext=(0.5, 3.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax[0].text(0.5, 3.72, 'wind', ha='center', fontsize=9)
    ax[0].legend(handles=[matplotlib.patches.Patch(color=cols[k], label=names[k]) for k in cols], fontsize=7, loc='lower right', framealpha=.9)
    ax[0].set_title('One made-up landscape: fire after 15, 30, 45 and 60 minutes', fontsize=11); ax[0].set_xlabel('km'); ax[0].set_ylabel('km')
    rows = res['Y3']['rows']
    for j, cs in enumerate((10, 15, 30)):
        v = [100 * x[f'area_{cs}'] for x in rows]
        ax[1].scatter(np.full(len(v), j) + np.linspace(-.18, .18, len(v)), v, s=45, color='#c8451f', zorder=3)
        ax[1].plot([j - .28, j + .28], [np.median(v)] * 2, 'k', lw=2)
    ax[1].axhline(0, color='k', lw=.8); ax[1].axhspan(-5, 5, color='#d9ead3', label='within 5 %')
    ax[1].set_xticks([0, 1, 2]); ax[1].set_xticklabels(['10 m cells', '15 m cells', '30 m cells\n(the size fuel maps come in)'])
    ax[1].set_ylabel('burned area compared with a 5 m run, %'); ax[1].legend(fontsize=8, loc='lower left')
    ax[1].set_title('12 made-up landscapes: how much the answer moves with solver cell size\n(each dot is one landscape, bar is the median)', fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out, '03_synthetic_landscapes.png'))


import matplotlib.patches, matplotlib.colors
if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_synth.json')):
    synth_fig()
