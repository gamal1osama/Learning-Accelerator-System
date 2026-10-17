def get_langfuse_config(session_id: str) -> dict:
    """
    Build the graph run config with session ID as the checkpoint thread ID.

    The config is passed to graph.invoke() on every call: both the initial
    invocation and any subsequent resume calls. LangGraph uses the thread_id
    to find and load the right checkpoint.
    """
    config = {
        "configurable": {
            "thread_id": session_id,
        }
    }
    # If Langfuse is configured, callbacks are added here (Chapter 6)
    handler = get_langfuse_handler(session_id)
    if handler:
        config["callbacks"] = [handler]
    return config

