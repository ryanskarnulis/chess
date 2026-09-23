"""The offline latency report (#317), on synthetic records.

What it must never do is the point of these tests: print a percentile its
sample cannot support, drop a failure or a censored wait from the count, mix
a cold call into a warm row, or add overlapping spans into a serial total.
"""

import json

from latency_report import load, main, percentile, render, summarize


def call(phase="planner", status="ok", ms=500, server_ms=480, **extra):
    return {
        "seq": 0,
        "phase": phase,
        "status": status,
        "ms": ms,
        "server_ms": server_ms,
        "prompt_tokens": 1,
        "completion_tokens": 1,
        **extra,
    }


_hour = iter(range(10_000))


def turn(calls=(), *, ts=None, route="brain", **fields):
    """A schema-2 turn record. Each default timestamp is an hour after the
    last, so fixtures never overlap unless a test places them to."""
    if ts is None:
        ts = f"2026-01-{1 + (h := next(_hour)) // 24:02d}T{h % 24:02d}:00:00+00:00"
    return {
        "schema": 2,
        "kind": "turn",
        "ts": ts,
        "route": route,
        "interaction_id": fields.pop("interaction_id", ""),
        "serving": fields.pop(
            "serving", {"manifest_id": "m1", "session": "s1", "experiment": ""}
        ),
        "calls": list(calls),
        "model_ms": sum(c["ms"] for c in calls),
        "spans_ms": fields.pop("spans_ms", {"total": 1000}),
        "planning": fields.pop("planning", None),
        **fields,
    }


def voice(interaction_id="i1", *, censored=False, outcome="ended", **marks):
    return {
        "schema": 2,
        "kind": "voice",
        "ts": "2026-09-23T12:00:05+00:00",
        "interaction_id": interaction_id,
        "origin": "voice",
        "clock": "client_monotonic_ms",
        "start": "speech_end",
        "marks": marks,
        "outcome": outcome,
        "censored": censored,
    }


def test_nearest_rank_percentiles_need_a_sample_that_supports_them():
    values = list(range(1, 101))
    assert percentile(values, 50) == 50
    assert percentile(values, 95) == 95
    assert percentile(values, 99) == 99
    assert percentile(values[:19], 95) is None
    assert percentile(values[:99], 99) is None
    assert percentile([], 50) is None
    assert percentile([7], 50) == 7


def test_failed_and_late_calls_are_counted_but_not_ranked():
    report = summarize(
        [
            turn([call(ms=400), call(ms=600)]),
            turn([call(status="failed", ms=30_000, server_ms=None)]),
            turn([call(phase="closer", status="late", ms=10_000, server_ms=None)]),
        ]
    )
    warm = report.calls[("m1", "warm", "planner")].row()
    assert (warm["n"], warm["ok"], warm["p50"]) == (2, 2, 400)
    failed = report.calls[("m1", "unknown", "planner")].row()
    assert (failed["n"], failed["failed"], failed["p50"]) == (1, 1, None)
    late = report.calls[("m1", "unknown", "closer")].row()
    assert (late["censored"], late["p50"]) == (1, None)


def test_a_call_that_waited_outside_the_server_is_cold():
    report = summarize(
        [turn([call(ms=104_000, server_ms=900)]), turn([call(ms=950, server_ms=900)])],
        cold_gap_ms=5_000,
    )
    assert report.calls[("m1", "cold", "planner")].values == [104_000]
    assert report.calls[("m1", "warm", "planner")].values == [950]
    # The turn's spans follow the same split.
    assert ("brain", "cold", "model") in report.spans
    assert ("brain", "warm", "model") in report.spans


def test_a_turn_that_queued_or_overlapped_another_is_contended():
    report = summarize(
        [
            turn([call()], spans_ms={"queue": 300, "total": 1000}),
            turn([call()], ts="2026-09-23T13:00:00+00:00", spans_ms={"total": 5000}),
            turn([call()], ts="2026-09-23T13:00:02+00:00", spans_ms={"total": 3000}),
            turn([call()], ts="2026-09-23T14:00:00+00:00", spans_ms={"total": 1000}),
        ]
    )
    assert report.calls[("m1", "warm+contended", "planner")].n == 3
    assert report.calls[("m1", "warm", "planner")].n == 1


def test_overlapping_spans_are_listed_side_by_side_never_summed():
    report = summarize(
        [turn([call(ms=700)], spans_ms={"engine": 900, "tool": 40, "total": 1100})]
    )
    spans = {key[2]: s.values for key, s in report.spans.items()}
    assert spans == {"engine": [900], "tool": [40], "total": [1100], "model": [700]}


def test_planning_overruns_are_counted():
    report = summarize(
        [
            turn(
                planning={
                    "elapsed_ms": 65_000,
                    "deadline_ms": 60_000,
                    "overrun_ms": 5_000,
                }
            ),
            turn(
                planning={"elapsed_ms": 1_000, "deadline_ms": 60_000, "overrun_ms": 0}
            ),
        ]
    )
    assert report.planning["phases"] == 2
    assert report.planning["overruns"] == 1
    assert report.planning["overrun_max_ms"] == 5_000


def test_client_segments_stay_in_the_browsers_clock():
    report = summarize(
        [
            turn(interaction_id="i1"),
            voice(
                stt_done=2_000,
                command_sent=2_010,
                first_board_update=2_600,
                command_done=4_000,
                tts_requested=4_010,
                tts_ready=5_000,
                playback_started=5_100,
                playback_ended=8_000,
            ),
        ]
    )
    segment = {key[1]: s.values for key, s in report.client.items()}
    assert segment["speech_end→transcript"] == [2_000]
    assert segment["command→first_board"] == [590]
    assert segment["start→first_audio"] == [5_100]
    assert segment["playback"] == [2_900]
    # A milestone the page never saw is absent, not zero.
    assert "command→engine_reply" not in segment
    assert report.outcomes == {"ended": 1}


def test_a_censored_interaction_is_counted_not_ranked():
    report = summarize([voice(censored=True, outcome="timeout", stt_done=1_500)])
    row = report.client[("voice", "speech_end→transcript")].row()
    assert (row["censored"], row["ok"], row["p50"]) == (1, 0, None)


def test_speech_failures_are_counted():
    report = summarize(
        [
            {"schema": 2, "kind": "speech", "op": "tts", "ms": 800, "status": "ok"},
            {
                "schema": 2,
                "kind": "speech",
                "op": "tts",
                "ms": 60_000,
                "status": "failed",
            },
        ]
    )
    row = report.speech["tts"].row()
    assert (row["n"], row["ok"], row["failed"], row["p50"]) == (2, 1, 1, 800)


def test_old_records_are_counted_and_skipped():
    report = summarize(
        [{"utterance": "e4", "model_latencies_ms": [300]}, turn([call()])]
    )
    assert report.skipped_old == 1
    assert report.calls[("m1", "warm", "planner")].n == 1


def test_an_experiment_filter_keeps_its_turns_and_the_records_joined_to_them():
    report = summarize(
        [
            turn(
                [call(ms=100)],
                interaction_id="i1",
                serving={"manifest_id": "m1", "session": "s", "experiment": "a"},
            ),
            turn(
                [call(ms=900)],
                interaction_id="i2",
                serving={"manifest_id": "m1", "session": "s", "experiment": "b"},
            ),
            {
                "schema": 2,
                "kind": "speech",
                "op": "stt",
                "ms": 50,
                "status": "ok",
                "interaction_id": "i1",
            },
            {
                "schema": 2,
                "kind": "speech",
                "op": "stt",
                "ms": 70,
                "status": "ok",
                "interaction_id": "i2",
            },
            voice("i1", stt_done=50),
            voice("i2", stt_done=70),
        ],
        experiment="a",
    )
    assert report.calls[("m1", "warm", "planner")].values == [100]
    assert report.speech["stt"].values == [50]
    assert report.client[("voice", "speech_end→transcript")].values == [50]


def test_the_manifests_seen_are_listed():
    report = summarize(
        [
            {
                "schema": 2,
                "kind": "serving",
                "manifest_id": "m1",
                "session": {"id": "s1", "experiment": ""},
                "app": {"revision": "abc", "version": "0.1.0"},
                "server": {
                    "source": "props",
                    "model_path": "/x/Q4.gguf",
                    "build_info": "b1",
                    "n_ctx": 8192,
                },
            }
        ]
    )
    assert report.manifests["m1"]["model_path"] == "/x/Q4.gguf"


def test_the_cli_prints_a_table_and_json(tmp_path, capsys):
    path = tmp_path / "turns.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in [turn([call()]), voice(stt_done=10)]) + "\n"
    )
    assert load([path])[0]["kind"] == "turn"

    assert main([str(path)]) == 0
    text = capsys.readouterr().out
    assert "model calls" in text and "planner" in text
    assert "–" in text  # a p95 of one reading is not printed as a number

    assert main([str(path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["model_calls"][0]["key"] == ["m1", "warm", "planner"]
    assert data["model_calls"][0]["p95"] is None


def test_render_survives_an_empty_trace():
    assert "(none)" in render(summarize([]))
