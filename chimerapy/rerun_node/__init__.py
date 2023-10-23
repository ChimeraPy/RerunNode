def register_nodes_metadata():
    return {
        "description": "Rerun ChimeraPy Node",
        "nodes": [
            "chimerapy.rerun_node.node:RerunNode",
            "chimerapy.rerun_node.example_logging_nodes:Detector",
        ],
    }
