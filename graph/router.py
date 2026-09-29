from state import State


def route_query(state: State) -> str:
    """
    Route the classified query to the correct branch.

    Possible routes:
        academic
        fee
        both
        general
    """

    query_type = state["query_type"]

    routes = {
        "academic": "academic",
        "fee": "fee",
        "both": "both",
        "general": "general",
    }

    if query_type not in routes:
        raise ValueError(
            f"Unknown query type: {query_type}"
        )

    return routes[query_type]