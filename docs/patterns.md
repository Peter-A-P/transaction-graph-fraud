# Laundering patterns

The IBM synthetic set (Altman et al., 2023) plants laundering attempts in eight shapes, the
list it takes from AMLSim. **These are the generator's patterns.** Recall by pattern measures
how well each arm finds what this generator plants; it does not say how often a bank sees
each shape, or that real laundering looks like this. The real-data headline is Elliptic's.

## The eight shapes

| Pattern | What the money does | Why a row model struggles |
|---|---|---|
| Fan-out | One account sends to many | Each transfer is an ordinary payment; only the count of recipients is unusual |
| Fan-in | Many accounts send to one | The same, seen from the receiver |
| Cycle | Money leaves an account and returns to it through others | No transfer in a cycle is unusual; the closure is the signal, and it is two or more hops away |
| Scatter-gather | One account scatters to intermediaries who all pay one account | The source and sink are linked only through the middle layer |
| Gather-scatter | Many pay one hub, which pays many | A fan-in and a fan-out sharing an account |
| Bipartite | A set of senders pays a set of receivers | Each account looks like a modest fan |
| Stack | Layers of bipartite transfers, one after another | Depth: the shape is only visible over several hops |
| Random | A walk from account to account | No shape at all beyond the chain in time |

Most laundering transfers in the set belong to no named attempt; they are reported as "no
named pattern" in the recall table, not dropped.

## The labeller

`consort.graph.patterns.classify` names the shape of a set of transfers from their structure
alone, without the generator's label. The rules are in its module docstring; the tests build
each shape by hand (a 3-cycle, a fan-in of five, a scatter-gather, a stack of three layers,
and shapes that must not pass: two disjoint cycles, a cycle with a chord). Before any recall by
pattern is reported, every attempt in the pattern file is classified and the agreement with
its generator name is published below, so a reader knows how literally to take each label.

## Agreement between the generator's name and the shape

Filled by `consort data build ibm` on the first fetch; not yet run.

| Pattern | Attempts | Classified as named | Most common other shape |
|---|---|---|---|
