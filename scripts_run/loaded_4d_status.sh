# Shared exit handling for the two sequential VXH shell entry points.
# No torch import: failure reporting must work even if simulator imports crash.
loaded_4d_record_failure() {
  local status_path=$1 exit_code=$2 kind=$3
  python - "$status_path" "$exit_code" "$kind" <<'PY'
import json, sys, time
from pathlib import Path
path, code, kind = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
if not path.exists():
    raise SystemExit(0)
data = json.loads(path.read_text())
state_key = 'status' if kind == 'segment' else 'state'
if data.get(state_key) == 'complete':
    raise SystemExit(0)
data['phase_before_failure'] = data.get(state_key)
data.update({state_key: 'failed', 'exit_code': code, 'failed_unix': time.time(),
             'failure_reason': 'SIGSEGV (native process crash)' if code == 139 else f'process exited with code {code}'})
temporary = path.with_suffix('.tmp')
temporary.write_text(json.dumps(data, indent=2) + '\n')
temporary.replace(path)
PY
}
