"""Freeze thresholds from the unmodified references and pre-training Stage1 rollout.
Run from repository root after calibration/stage1_track.json is complete.
"""
import json
from pathlib import Path
import numpy as np
import torch
from holosoma.config_values.wbt.g1.loaded_4d import DATA_ROOT
from holosoma.utils.rotations import quat_apply, quat_conjugate

base = Path('/home/han/pyx/HMI_clean_runs/loaded_4d/stage2_vxh')
report = json.loads((base/'calibration/stage1_track.json').read_text())
all_rows = [r for t in report['trials'] for r in t['trace'] if r['time_s'] > .5]
# Recompute tilt from saved actual quaternion: original baseline trace stored directed z angle.
for t in report['trials']:
    for r in t['trace']:
        q=np.asarray(r['box_quaternion'])
        r['box_tilt']=float(np.arccos(np.clip(abs(1-2*(q[0]**2+q[1]**2)),0,1)))
squat_rows = [r for t in report['trials'] if 'squat' in t['motion'] for r in t['trace'] if r['time_s'] > .5]
local = [np.array([r['hand_box_local'] for r in all_rows])]
references = {}
for p in sorted((DATA_ROOT/'joint').glob('*.npz')):
    d=np.load(p); ids=[list(d['body_names']).index(s+'_wrist_yaw_link') for s in ('left','right')]
    q=torch.tensor(d['object_quat_w'])[:,[1,2,3,0]]
    rel=torch.tensor(d['body_pos_w'][:,ids]-d['object_pos_w'][:,None])
    vals=quat_apply(quat_conjugate(q,w_last=True)[:,None,:].expand(-1,2,-1),rel,w_last=True).numpy()
    local.append(vals)
    references[p.name]={'min':vals.min(0).tolist(),'max':vals.max(0).tolist()}
local=np.concatenate(local)
def up(x, step=.01): return float(np.ceil(x/step)*step)
def stat(rows,key):
    a=np.asarray([r[key] for r in rows]); return {'p01':np.quantile(a,.01,axis=0).tolist(),'p50':np.quantile(a,.50,axis=0).tolist(),'p99':np.quantile(a,.99,axis=0).tolist()}
stats={k:stat(squat_rows if k.startswith('foot') else all_rows,k) for k in ('foot_force_z','foot_height','foot_slip','hand_box_speed','box_tilt')}
criteria={
    'version':1, 'min_steady_s':2., 'vy_mae_max':.05,'yaw_mae_max':.15,
    'foot_force_min':20.,
    'foot_height_max':min(.05, max(.025, up(np.quantile([r['foot_height'] for r in squat_rows],.99)+.01))),
    'foot_slip_max':min(.35, max(.15, up(np.quantile([r['foot_slip'] for r in squat_rows],.99)+.05))),
    'hand_local_min':(np.floor(np.quantile(local,.001,axis=0)*100)/100-.01).round(3).tolist(),
    'hand_local_max':(np.ceil(np.quantile(local,.999,axis=0)*100)/100+.01).round(3).tolist(),
    'hand_margin':.05, 'hand_speed_max':max(.35,up(np.quantile([r['hand_box_speed'] for r in all_rows],.99)+.1)),
    'box_tilt_max':max(.30,up(np.quantile([r['box_tilt'] for r in all_rows],.99)+.1)),
    'support_fraction_min':.9,'carry_fraction_min':.95,
}
summary={'source_checkpoint':report['checkpoint'],'source_file':str(base/'calibration/stage1_track.json'),
         'source_completion_fraction':report['completion_fraction'],'source_trials':len(report['trials']),
         'sensor_definition':'Ankle-roll net world vertical contact force; four sole probes at z=-.035; box-local wrist geometry (not filtered hand-box contact); box vertical-axis tilt acos(abs(local_z dot gravity_up)), because reference local z points down.',
         'derivation':'Before Stage2 training: squat p99 + .01m sole / .05m/s slip, capped .05m/.35m/s; carry union reference and Stage1 q.001-.999 rounded out +.01m; .05m acceptance margin; Stage1 p99 speed/tilt +.1 with floors .35m/s/.30rad. 20N excludes light touch, no equal-load condition.',
         'statistics':stats,'reference_hand_local':references,'criteria':criteria,
         'track_results':[{'motion':m,'completed':sum(t['outcome']=='clip_complete' for t in report['trials'] if t['motion']==m),'trials':sum(t['motion']==m for t in report['trials']),
             'body_object_mae':np.mean([t['tracking_errors_body_object']['mae'] for t in report['trials'] if t['motion']==m],axis=0).tolist()} for m in sorted(set(t['motion'] for t in report['trials']))]}
Path('docs/loaded_4d_vxh/criteria.json').write_text(json.dumps(criteria,indent=2)+'\n')
Path('docs/loaded_4d_vxh/calibration_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
