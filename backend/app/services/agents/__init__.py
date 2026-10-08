"""
Agents package. Keep this __init__.py free of eager submodule imports so that
cross-package imports (e.g. query_parser -> taxonomy) do not create cycles
through investigation_tools -> hybrid_search_engine -> query_parser.
"""
