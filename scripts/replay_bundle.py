from belief_graph_orchestrator.cli import main

if __name__ == '__main__':
    import sys; sys.argv = ['belief-graph-orchestrator','replay',*sys.argv[1:]]; main()
