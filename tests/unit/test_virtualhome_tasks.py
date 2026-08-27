from adaptiveguard.environments.virtualhome_tasks import make_goal_predicate


def test_goal_predicate_requires_matching_node_state() -> None:
    predicate = make_goal_predicate(
        {"node_states": [{"id": 1, "class_name": "lamp", "state": "on"}]}
    )

    assert predicate({"nodes": [{"id": 1, "class_name": "lamp", "states": ["ON"]}]})
    assert not predicate({"nodes": [{"id": 1, "class_name": "lamp", "states": ["OFF"]}]})


def test_goal_predicate_requires_all_node_and_edge_conditions() -> None:
    predicate = make_goal_predicate(
        {
            "node_states": [{"class_name": "lamp", "state": "on"}],
            "edges": [{"from_id": 1, "to_id": 2, "relation": "INSIDE"}],
        }
    )
    complete = {
        "nodes": [{"class_name": "lamp", "states": ["ON"]}],
        "edges": [{"from_id": 1, "to_id": 2, "relation": "INSIDE"}],
    }

    assert predicate(complete)
    assert not predicate({**complete, "edges": []})


def test_empty_goal_spec_is_not_satisfied() -> None:
    predicate = make_goal_predicate(None)

    assert not predicate({"nodes": [], "edges": []})
