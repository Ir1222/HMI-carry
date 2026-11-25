"""Risk checks for the local experiment guard, without invoking training."""
import importlib.util
import json
from pathlib import Path


def load_module():
    path = Path(__file__).parents[1] / 'summarize_loaded_4d_vxh.py'
    spec = importlib.util.spec_from_file_location('vxh_summary', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_falling_motion_with_short_steady_window_is_not_control_response(tmp_path):
    goals = [[v, 0., 0., .73] for v in (.1, .4, .8)] + [[0., 0., 0., h] for h in (.6, .69, .8)]
    trials = []
    for goal in goals:
        trials.append({'segments': [{'target': goal, 'mae': [.01]*4, 'p95': [.02]*4,
                       'steady_seconds': .06, 'measured_mean': goal, 'failure_reasons': ['insufficient_steady_time']}],
                       'termination_terms': ['physical_failure'], 'task_success': False, 'outcome': 'terminated',
                       'seconds': 1., 'physical_resets_after_initial': 1})
    gen = {'sequences': [[g] for g in goals], 'criteria': {'min_steady_s': 2}, 'trials': trials,
           'checkpoint': 'early.pt', 'checkpoint_iteration': 6100, 'criteria_sha256': 'frozen',
           'task_success_fraction': 0., 'completion_fraction': 0.}
    track = {'completion_fraction': 1., 'trials': [
        {'motion': name, 'outcome': 'clip_complete', 'tracking_errors_body_object': {'mae': [.03, .04]}}
        for name in ('loaded_locomotion', 'continuous_squat', 'repeated_squat')]}
    (tmp_path/'targets.json').write_text(json.dumps(gen))
    (tmp_path/'track.json').write_text(json.dumps(track))
    result = load_module().summarize(tmp_path, 103)
    assert result['no_response']
    assert not result['severe_track_loss']
    assert all(t['measured_mean'] is None for t in result['targets'])
    track['completion_fraction'] = .3
    track['trials'][0]['outcome'] = 'terminated'
    (tmp_path/'track.json').write_text(json.dumps(track))
    assert load_module().summarize(tmp_path, 103)['severe_track_loss']


def test_update_budget_counts_repeated_iteration_labels(tmp_path):
    import torch
    baseline=tmp_path/'stage1.pt'
    torch.save({'iter':6000,'experiment_config':{'command':{'setup_terms':{'motion_command':{'params':{'loaded_4d':{}}}}}}},baseline)
    previous=baseline
    for name,first,last,requested in [('smoke',6000,6002,3),('restore',6002,6003,2),('short',6003,6997,995)]:
        folder=tmp_path/name;folder.mkdir()
        (folder/'segment.json').write_text(json.dumps({'checkpoint':str(previous),'first_display_iteration':first,'requested_updates':requested}))
        checkpoint=folder/'model.pt'
        torch.save({'iter':last,'experiment_config':{'logger':{'base_dir':str(folder)},'command':{'setup_terms':{'motion_command':{'params':{'loaded_4d':{'profile':'vxh'}}}}}}},checkpoint)
        previous=checkpoint
    assert load_module().checkpoint_updates(previous)==1000
