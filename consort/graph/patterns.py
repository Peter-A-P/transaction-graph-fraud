"""The pattern labeller: what shape a set of transfers makes, from its structure alone.

The IBM generator names each laundering attempt's pattern. Before recall by pattern is
trusted, every attempt's transfers are classified here from their structure, without the name,
and the agreement between the generator's name and the shape is published in docs/patterns.md.
The same classifier labels the hand-built toy graphs the tests use.

Shapes, on the attempt's deduplicated directed graph with n accounts:

- CYCLE: connected, every account sends once and receives once.
- FAN-IN: one account receives from all the others, who only send, once each.
- FAN-OUT: one account sends to all the others, who only receive, once each.
- SCATTER-GATHER: one source sends to k >= 2 intermediaries, each of which sends only to one
  sink, which receives only from them.
- GATHER-SCATTER: k >= 2 sources send only to one hub, which sends to m >= 2 sinks that
  receive only from it.
- STACK: a layered DAG of three or more layers, edges only from one layer to the next, at
  least one layer of two or more accounts.
- BIPARTITE: every account only sends or only receives, two or more on each side.
- RANDOM: in time order, each transfer leaves the account the previous one reached (a walk),
  and the shape is none of the above.
- OTHER: none of these.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

SHAPES = (
    "CYCLE",
    "FAN-IN",
    "FAN-OUT",
    "SCATTER-GATHER",
    "GATHER-SCATTER",
    "STACK",
    "BIPARTITE",
    "RANDOM",
    "OTHER",
)


def classify(src: Sequence[str | int], dst: Sequence[str | int]) -> str:
    """Shape of the transfers src[i] -> dst[i], given in time order."""
    if len(src) != len(dst) or not src:
        raise ValueError("classify needs one or more transfers")
    edges = sorted({(a, b) for a, b in zip(src, dst, strict=True) if a != b}, key=str)
    nodes = sorted({x for e in edges for x in e}, key=str)
    if not edges:
        return "OTHER"
    outs: dict[str | int, set[str | int]] = defaultdict(set)
    ins: dict[str | int, set[str | int]] = defaultdict(set)
    for a, b in edges:
        outs[a].add(b)
        ins[b].add(a)
    n = len(nodes)
    od = {v: len(outs[v]) for v in nodes}
    idg = {v: len(ins[v]) for v in nodes}

    if n >= 2 and all(od[v] == 1 and idg[v] == 1 for v in nodes) and _connected(nodes, edges):
        return "CYCLE"
    if n >= 3:
        hubs_in = [v for v in nodes if idg[v] == n - 1 and od[v] == 0]
        if len(hubs_in) == 1 and all(od[v] == 1 and idg[v] == 0 for v in nodes if v != hubs_in[0]):
            return "FAN-IN"
        hubs_out = [v for v in nodes if od[v] == n - 1 and idg[v] == 0]
        if len(hubs_out) == 1 and all(
            idg[v] == 1 and od[v] == 0 for v in nodes if v != hubs_out[0]
        ):
            return "FAN-OUT"
    sources = [v for v in nodes if idg[v] == 0]
    sinks = [v for v in nodes if od[v] == 0]
    middle = [v for v in nodes if idg[v] > 0 and od[v] > 0]
    if len(sources) == 1 and len(sinks) == 1 and len(middle) >= 2:
        s, t = sources[0], sinks[0]
        if (
            outs[s] == set(middle)
            and ins[t] == set(middle)
            and all(ins[m] == {s} and outs[m] == {t} for m in middle)
        ):
            return "SCATTER-GATHER"
    if len(middle) == 1 and len(sources) >= 2 and len(sinks) >= 2:
        h = middle[0]
        if all(outs[s] == {h} for s in sources) and all(ins[t] == {h} for t in sinks):
            return "GATHER-SCATTER"
    layers = _layers(nodes, outs, ins)
    if layers is not None and max(layers.values()) >= 2:
        sizes: dict[int, int] = defaultdict(int)
        for v in nodes:
            sizes[layers[v]] += 1
        if max(sizes.values()) >= 2 and all(layers[b] == layers[a] + 1 for a, b in edges):
            return "STACK"
    if not middle and len(sources) >= 2 and len(sinks) >= 2:
        return "BIPARTITE"
    if all(dst[i] == src[i + 1] for i in range(len(src) - 1)):
        return "RANDOM"
    return "OTHER"


def _connected(nodes: list[str | int], edges: list[tuple[str | int, str | int]]) -> bool:
    adj: dict[str | int, set[str | int]] = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen = {nodes[0]}
    stack = [nodes[0]]
    while stack:
        for w in adj[stack.pop()]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return len(seen) == len(nodes)


def _layers(
    nodes: list[str | int],
    outs: dict[str | int, set[str | int]],
    ins: dict[str | int, set[str | int]],
) -> dict[str | int, int] | None:
    """Longest-path depth from the sources, or None if the graph has a cycle."""
    indeg = {v: len(ins[v]) for v in nodes}
    depth = {v: 0 for v in nodes}
    ready = [v for v in nodes if indeg[v] == 0]
    done = 0
    while ready:
        v = ready.pop()
        done += 1
        for w in outs[v]:
            depth[w] = max(depth[w], depth[v] + 1)
            indeg[w] -= 1
            if indeg[w] == 0:
                ready.append(w)
    return depth if done == len(nodes) else None
