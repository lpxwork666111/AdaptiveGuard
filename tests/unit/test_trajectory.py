from adaptiveguard.core.trajectory import StepRecord, TrajectoryBuffer


def test_cold_start_and_successful_trajectory_criticality() -> None:
    buffer = TrajectoryBuffer()
    assert buffer.criticality("open") == 0.5
    buffer.add_trajectory([StepRecord("open", "x", 0), StepRecord("finish", "y", 1)], success=True)
    assert buffer.criticality("open") == 1.0


def test_sps_is_clipped() -> None:
    buffer = TrajectoryBuffer()
    assert buffer.observe_score(500, 100) == 1.0


def test_score_window_uses_recent_values_and_respects_gamma() -> None:
    buffer = TrajectoryBuffer(sps_window=2, gamma=0.0)

    buffer.observe_score(1, 1)
    assert buffer.observe_score(-1, 1) == -1.0


def test_action_statistics_normalize_templates_consistently() -> None:
    buffer = TrajectoryBuffer()
    buffer.add_trajectory([StepRecord(" Open   Door ", "x", 0)], success=True)

    assert buffer.action_frequency("open door") == 1.0
    assert buffer.path_relevance("OPEN DOOR") == 1.0
