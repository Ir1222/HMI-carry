"""Exercise shell failure propagation without launching or mocking a simulator."""
import json
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize('kind,state_key,completed', [('segment','status',0), ('driver','state',2500)])
def test_child_crash_marks_failed_and_preserves_update_accounting(tmp_path, kind, state_key, completed):
    helper=Path(__file__).resolve().parents[1]/'loaded_4d_status.sh'
    path=tmp_path/'status.json'
    path.write_text(json.dumps({state_key:'running','completed_updates':completed,'checkpoint':'saved.pt'}))
    result=subprocess.run(['bash','-c', '''set -euo pipefail
source "$1"
trap 'code=$?; if (( code != 0 )); then loaded_4d_record_failure "$2" "$code" "$3"; fi' EXIT
bash -c 'exit 139'
''', 'test',str(helper),str(path),kind],capture_output=True,text=True)
    assert result.returncode==139, result.stderr
    data=json.loads(path.read_text())
    assert data[state_key]=='failed' and data['exit_code']==139
    assert data['completed_updates']==completed and data['checkpoint']=='saved.pt'
    assert 'SIGSEGV' in data['failure_reason']


def test_failure_reporting_does_not_rewrite_completed_segment(tmp_path):
    helper=Path(__file__).resolve().parents[1]/'loaded_4d_status.sh'
    path=tmp_path/'segment.json'; original='{"status":"complete","completed_updates":1500}\n'
    path.write_text(original)
    subprocess.run(['bash','-c','source "$1"; loaded_4d_record_failure "$2" 1 segment','test',str(helper),str(path)],check=True)
    assert path.read_text()==original
