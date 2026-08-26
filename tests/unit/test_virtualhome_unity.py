from adaptiveguard.environments.virtualhome_unity import VirtualHomeUnityAdapter


class FakeCommunication:
    def __init__(self) -> None:
        self.graph = {"nodes": [{"id": 1, "states": []}], "edges": []}
        self.closed = False

    def reset(self, environment_id):
        return True

    def add_character(self):
        return True

    def environment_graph(self):
        return True, self.graph

    def render_script(self, script, **kwargs):
        self.graph["nodes"][0]["states"] = ["ON"]
        return True, "ok"

    def close(self):
        self.closed = True


def test_unity_adapter_with_injected_communication() -> None:
    communication = FakeCommunication()
    adapter = VirtualHomeUnityAdapter(
        communication=communication,
        goal_predicate=lambda graph: "ON" in graph["nodes"][0]["states"],
    )
    adapter.reset()
    transition = adapter.step("[SWITCHON] <lamp> (1)")
    assert transition.done is True
    adapter.close()
    assert communication.closed is True
