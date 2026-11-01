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

From `consort patterns --tier HI-Small` and `--tier LI-Small`, run 2026-11-01 on the first
fetch. An attempt counts as agreeing when the labeller, given only its transfers, names the
shape the generator gave it.

**HI-Small** (370 attempts, 290 agree, 78%)

| Pattern | Attempts | Classified as named | Most common other shape |
|---|---|---|---|
| Bipartite | 49 | 31 (63%) | RANDOM (18) |
| Cycle | 54 | 54 (100%) | none |
| Fan-in | 40 | 36 (90%) | RANDOM (4) |
| Fan-out | 48 | 40 (83%) | RANDOM (8) |
| Gather-scatter | 51 | 33 (65%) | OTHER (12) |
| Random | 41 | 40 (98%) | OTHER (1) |
| Scatter-gather | 44 | 31 (70%) | CYCLE (11) |
| Stack | 43 | 25 (58%) | CYCLE (13) |

**LI-Small** (117 attempts, 93 agree, 79%)

| Pattern | Attempts | Classified as named | Most common other shape |
|---|---|---|---|
| Bipartite | 16 | 14 (88%) | RANDOM (2) |
| Cycle | 12 | 12 (100%) | none |
| Fan-in | 12 | 10 (83%) | RANDOM (2) |
| Fan-out | 19 | 17 (89%) | RANDOM (1) |
| Gather-scatter | 12 | 8 (67%) | OTHER (3) |
| Random | 15 | 13 (87%) | OTHER (2) |
| Scatter-gather | 13 | 8 (62%) | CYCLE (5) |
| Stack | 18 | 11 (61%) | CYCLE (5) |

**Why they disagree.** Of the 80 disagreeing attempts in HI-Small, 61 involve two accounts or
fewer: the file holds the laundering transfers the generator emitted, and many attempts are
cut short (a "fan-out" of one transfer, a "stack" that is one payment and its return), so
there is no shape left to see. That is what RANDOM and CYCLE mean in the right-hand column:
`A>B` alone is a one-step walk, and `A>B B>A` is a 2-cycle. Seven more contain a transfer from
an account to itself, which the labeller reads as OTHER. The remaining 12 are shapes whose
ends meet: a scatter-gather whose sink is its own source (`A>B A>C B>A C>A`), which is
structurally a set of cycles. LI-Small splits 19, 0 and 5 the same way.

**What this means for the recall table.** Recall by pattern is reported under the generator's
names, since those are what it plants; but about a fifth of the attempts behind each name,
and over a third of bipartite and stack attempts in HI-Small, are too small to carry the
shape the name promises. A shape-only arm cannot be expected to find a "stack" that is one
transfer, and its row in the table is read with that in mind.
