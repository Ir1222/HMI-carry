"""Exercise viewer import ordering without launching a simulator or a server."""

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("enabled", [True, False])
def test_viewer_dependencies_resolve_before_simulator_path_shadowing(tmp_path, enabled):
    # Model Kit's bundled old websockets: it has no asyncio subpackage.
    package = tmp_path / "websockets"
    package.mkdir()
    (package / "__init__.py").write_text('__version__ = "12.0"\n')
    code = textwrap.dedent(
        """
        import sys
        from holosoma import eval_agent
        from holosoma.config_types.eval_callback import (
            EvalCallbacksConfig, ViserCallbackConfig, ViserConfig,
        )

        enabled = sys.argv[2] == "True"

        class StopBeforePhysics(Exception):
            pass

        def fake_setup(config):
            sys.path.insert(0, sys.argv[1])
            if enabled:
                # This raised the reported ModuleNotFoundError before the fix.
                import viser
                import websockets.asyncio.server
            else:
                # Non-viewer evaluation must not acquire an optional dependency.
                assert "viser" not in sys.modules
                import websockets
                assert websockets.__version__ == "12.0"
            raise StopBeforePhysics

        eval_agent.setup_simulation_environment = fake_setup
        callbacks = EvalCallbacksConfig(
            viser=ViserCallbackConfig(config=ViserConfig(enabled=enabled)),
        )
        try:
            eval_agent.run_eval_with_tyro(None, None, None, None, callbacks)
        except StopBeforePhysics:
            pass
        else:
            raise AssertionError("Simulation setup was not reached")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), str(enabled)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_missing_viewer_dependency_fails_before_simulation(monkeypatch):
    from holosoma import eval_agent
    from holosoma.config_types.eval_callback import EvalCallbacksConfig, ViserCallbackConfig, ViserConfig

    def missing_import(name):
        raise ModuleNotFoundError("No module named 'websockets.asyncio'")

    def unexpected_setup(config):
        pytest.fail("Simulation must not start after a viewer import failure")

    monkeypatch.setattr(eval_agent, "import_module", missing_import)
    monkeypatch.setattr(eval_agent, "setup_simulation_environment", unexpected_setup)
    callbacks = EvalCallbacksConfig(viser=ViserCallbackConfig(config=ViserConfig(enabled=True)))
    with pytest.raises(RuntimeError, match="Cannot import Viser before simulator startup") as exc:
        eval_agent.run_eval_with_tyro(None, None, None, None, callbacks)
    assert isinstance(exc.value.__cause__, ModuleNotFoundError)
