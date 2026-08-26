from adaptiveguard.core.trajectory import StepRecord, TrajectoryBuffer


def test_cold_start_and_successful_trajectory_criticality() -> None:
    buffer = TrajectoryBuffer()
    assert buffer.criticality("open") == 0.5
    buffer.add_trajectory([StepRecord("open", "x", 0), StepRecord("finish", "y", 1)], success=True)
    assert buffer.criticality("open") == 1.0


def test_sps_is_clipped() -> None:
    buffer = TrajectoryBuffer()
    assert buffer.observe_score(500, 100) == 1.0
