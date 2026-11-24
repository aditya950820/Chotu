import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from watchdog import decide  # noqa: E402


def ev(time, state, **kw):
    return dict(time=f"2026-11-24T{time}+00:00", stage="pretrain", state=state, **kw)


def test_spot_interruption_mid_training_resumes():
    events = [ev("06:20:00", "start"), ev("08:00:00", "progress", step=2000)]
    assert decide(events, None, False)[0] == "resume"


def test_signal_stop_resumes_but_kill_rule_and_failures_do_not():
    signal = [ev("08:00:00", "progress"), ev("08:00:05", "ALERT", message="signal 15; interrupted stage can resume"),
              ev("08:00:06", "final_backup")]
    assert decide(signal, None, False)[0] == "resume"
    nan = [ev("08:00:00", "progress"), ev("08:00:05", "ALERT", message="pretrain child exited 20; see log"),
           ev("08:00:06", "final_backup"), ev("08:00:30", "pausing")]
    action, reason = decide(nan, None, False)
    assert action == "never" and "exited 20" in reason


def test_setup_no_go_is_not_resumed_but_newer_chain_activity_wins():
    setup = dict(time="2026-11-24T06:16:00+00:00", code=1, reason="during: speed sweep")
    assert decide([], setup, False)[0] == "never"
    older_chain = [ev("06:10:00", "progress")]
    assert decide(older_chain, setup, False)[0] == "never"
    newer_chain = [ev("07:00:00", "progress")]       # setup failed once, later the chain ran fine
    assert decide(newer_chain, setup, False)[0] == "resume"


def test_finished_run_stops_the_watchdog():
    events = [ev("20:00:00", "done"), ev("20:00:01", "final_backup"), ev("20:00:02", "pausing")]
    assert decide(events, None, True)[0] == "finished"
