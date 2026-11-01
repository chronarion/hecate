"""hecate: a process-wide switch that makes every module CUDA graph (BC_* Graph objects) take its eager launch
sequence while an outer capture records a whole forward (runtime/core/exl3_vgraph.py; a graph launch cannot nest inside
a stream capture). Graph::disabled becomes a flag whose bool conversion ORs in the switch, so the call sites stay as they
are.   python3 vg_eager.py EXT_DIR"""
import sys
ext = sys.argv[1]


def patch(name, reps):
    p = f"{ext}/{name}"; s = open(p).read()
    for old, new in reps:
        assert s.count(old) == 1, (name, old[:80], s.count(old))
        s = s.replace(old, new)
    open(p, "w").write(s); print("patched", name)


patch("graph.cuh", [
    ("class Graph\n{\npublic:\n",
     "// hecate: set while an outer capture records a whole forward: every module graph runs its eager sequence\n"
     "extern bool g_graphs_force_eager;\n"
     "void graphs_force_eager(bool v);\n"
     "struct GraphDisabledFlag\n"
     "{\n"
     "    bool v = false;\n"
     "    operator bool() const { return v || g_graphs_force_eager; }\n"
     "    GraphDisabledFlag& operator=(bool b) { v = b; return *this; }\n"
     "};\n\n"
     "class Graph\n{\npublic:\n"),
    ("    bool disabled;\n", "    GraphDisabledFlag disabled;\n"),
])
patch("graph.cu", [
    ("//#define GRAPHDEBUG 1\n",
     "//#define GRAPHDEBUG 1\n\nbool g_graphs_force_eager = false;\nvoid graphs_force_eager(bool v) { g_graphs_force_eager = v; }\n"),
])
patch("bindings.cpp", [
    ("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)\n",
     "void graphs_force_eager(bool v);\n\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m)\n"),
    ('    m.def("stloader_read", &stloader_read, "stloader_read");\n',
     '    m.def("stloader_read", &stloader_read, "stloader_read");\n'
     '    m.def("graphs_force_eager", &graphs_force_eager, "graphs_force_eager");\n'),
])
