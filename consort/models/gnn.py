"""Arm N: a graph neural network with neighbour sampling, trained inductively.

GraphSAGE is the arm; GAT the second architecture. On Elliptic each transaction is a node to
classify; on IBM each transfer is an edge to classify, from its two accounts' embeddings and
its own row. The hop count is the number of message-passing layers, and the hub cap is the
recency cap of consort.graph.sampler: the same neighbourhood arm G aggregates over and the
serving path fetches, rather than PyG's random sampling (which also needs a native extension
with no wheels for this torch on Windows). At hop 0 there is no message passing and the
model is a multilayer perceptron on the row, the GNN's own zero point on the hop curve.

The directed-multigraph adaptations from Egressy et al. (Multi-GNN, 2024) are flags for the
ablation on the synthetic set: reverse message passing (separate weights for messages along
and against an edge's direction) and ego identifiers (a marker on the nodes being classified,
so a layer can tell the seed from its neighbours, which is what lets a GNN see a cycle close).
Port numbering is not implemented; docs/models.md says so.

Training sees only the training partition's graph and labels: on Elliptic, steps are
disconnected so the training nodes' neighbourhoods hold no other step; on IBM the message-
passing graph is the training window's transfers. Unknown labels pass messages and never
enter the loss.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn
from torch_geometric.nn import GATConv, SAGEConv

from consort.data.leakage import check_training_rows
from consort.data.splits import PARTITIONS, Partition
from consort.data.tasks import (
    IBM_NODE_COLUMNS,
    EllipticPPTask,
    EllipticTask,
    IBMTask,
    Task,
    ibm_node_table,
)
from consort.eval import outcomes as oc
from consort.eval.metrics import Ranked, average_precision
from consort.graph.build import Graph
from consort.graph.sampler import CappedSampler, local_ids

Arch = Literal["sage", "gat"]


@dataclass(frozen=True)
class GNNConfig:
    arch: Arch = "sage"
    hops: int = 2
    cap: int = 25
    hidden: int = 64
    dropout: float = 0.2
    lr: float = 3e-3
    weight_decay: float = 1e-5
    epochs: int = 40
    patience: int = 6
    batch_size: int = 2048
    reverse_mp: bool = False
    ego_ids: bool = False
    seed: int = 0

    @property
    def tag(self) -> str:
        flags = ("+rmp" if self.reverse_mp else "") + ("+ego" if self.ego_ids else "")
        return f"N-{self.arch}{flags}"


class Encoder(nn.Module):
    def __init__(self, in_dim: int, cfg: GNNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        dim = in_dim + (1 if cfg.ego_ids else 0)
        self.inp = nn.Linear(dim, cfg.hidden)
        self.fwd = nn.ModuleList()
        self.rev = nn.ModuleList()
        for _ in range(cfg.hops):
            self.fwd.append(self._conv(cfg))
            if cfg.reverse_mp:
                self.rev.append(self._conv(cfg))
        self.norms = nn.ModuleList(nn.LayerNorm(cfg.hidden) for _ in range(cfg.hops))
        self.drop = nn.Dropout(cfg.dropout)

    @staticmethod
    def _conv(cfg: GNNConfig) -> nn.Module:
        if cfg.arch == "sage":
            return SAGEConv(cfg.hidden, cfg.hidden)
        return GATConv(cfg.hidden, cfg.hidden // 4, heads=4, add_self_loops=False)

    def forward(self, x: Tensor, edge_index: Tensor, ego: Tensor | None) -> Tensor:
        if self.cfg.ego_ids:
            assert ego is not None
            x = torch.cat([x, ego.unsqueeze(1).float()], dim=1)
        h = torch.relu(self.inp(x))
        both = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        for i in range(self.cfg.hops):
            if self.cfg.reverse_mp:
                m = self.fwd[i](h, edge_index) + self.rev[i](h, edge_index.flip(0))
            else:
                m = self.fwd[i](h, both)
            h = self.norms[i](h + self.drop(torch.relu(m)))
        return h


class NodeModel(nn.Module):
    def __init__(self, in_dim: int, cfg: GNNConfig) -> None:
        super().__init__()
        self.enc = Encoder(in_dim, cfg)
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.hidden), nn.ReLU(), nn.Linear(cfg.hidden, 1)
        )

    def forward(self, x: Tensor, edge_index: Tensor, ego: Tensor | None) -> Tensor:
        out: Tensor = self.head(self.enc(x, edge_index, ego)).squeeze(-1)
        return out


class EdgeModel(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, cfg: GNNConfig) -> None:
        super().__init__()
        self.enc = Encoder(node_dim, cfg)
        self.head = nn.Sequential(
            nn.Linear(2 * cfg.hidden + edge_dim, cfg.hidden), nn.ReLU(), nn.Linear(cfg.hidden, 1)
        )

    def forward(
        self, x: Tensor, edge_index: Tensor, ego: Tensor | None, pair: Tensor, edge_x: Tensor
    ) -> Tensor:
        h = self.enc(x, edge_index, ego)
        out: Tensor = self.head(torch.cat([h[pair[0]], h[pair[1]], edge_x], dim=1)).squeeze(-1)
        return out


def _standardise(x: npt.NDArray[np.float32], train: npt.NDArray[np.bool_]) -> Tensor:
    mu = x[train].mean(axis=0)
    sd = x[train].std(axis=0) + 1e-6
    return torch.from_numpy(((x - mu) / sd).astype(np.float32))


def _auprc(y: npt.NDArray[np.int8], s: npt.NDArray[np.float64]) -> float:
    return average_precision(Ranked.of(y, s, np.zeros(len(y))), np.ones(len(y)))


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _Trainer:
    """Early stopping on validation AUPRC, the same rule arms R and G use."""

    def __init__(self, model: nn.Module, cfg: GNNConfig, pos_weight: float) -> None:
        self.model = model.to(_device())
        self.cfg = cfg
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=_device()))

    def fit(self, train_epoch: Any, val_scores: Any, y_val: npt.NDArray[np.int8]) -> dict[str, Any]:
        best, best_state, stale, history = -1.0, None, 0, []
        for epoch in range(self.cfg.epochs):
            self.model.train()
            loss = train_epoch(self.model, self.opt, self.loss)
            self.model.eval()
            with torch.no_grad():
                ap = _auprc(y_val, val_scores(self.model))
            history.append({"epoch": epoch, "loss": loss, "val_auprc": ap})
            if ap > best:
                best, best_state, stale = ap, copy.deepcopy(self.model.state_dict()), 0
            else:
                stale += 1
                if stale >= self.cfg.patience:
                    break
        if best_state is None:
            raise ValueError(
                f"validation AUPRC was never a number ({history[-1]['val_auprc']}): the "
                "validation partition has no labelled positives to rank"
            )
        self.model.load_state_dict(best_state)
        return {"best_val_auprc": best, "epochs_run": len(history)}


def _batches(
    n: int, size: int, shuffle: bool, rng: np.random.Generator
) -> list[npt.NDArray[np.int64]]:
    order = rng.permutation(n) if shuffle else np.arange(n)
    return [order[i : i + size] for i in range(0, n, size)]


@dataclass
class _View:
    """One partition's graph and node inputs, sampled with the recency cap."""

    x: Tensor
    sampler: CappedSampler
    hops: int

    def tensors(
        self, seeds: npt.NDArray[np.int64], dev: torch.device
    ) -> tuple[Tensor, Tensor, Tensor, Any]:
        b = self.sampler.batch(seeds, self.hops)
        x = self.x[torch.from_numpy(b.nodes)].to(dev)
        ei = torch.from_numpy(b.edge_index).to(dev)
        ego = torch.zeros(len(b.nodes), dtype=torch.bool, device=dev)
        ego[: b.n_seeds] = True
        return x, ei, ego, b


def _seed_ego(cfg: GNNConfig, ego: Tensor) -> Tensor | None:
    return ego if cfg.ego_ids else None


def _score_nodes(
    model: nn.Module, view: _View, idx: npt.NDArray[np.int64], cfg: GNNConfig
) -> npt.NDArray[np.float64]:
    dev = _device()
    model.eval()
    out = []
    for chunk in _batches(len(idx), cfg.batch_size, False, np.random.default_rng(0)):
        seeds = idx[chunk]
        x, ei, ego, b = view.tensors(seeds, dev)
        with torch.no_grad():
            logit = model(x, ei, _seed_ego(cfg, ego))
        out.append(torch.sigmoid(logit[torch.from_numpy(local_ids(b, seeds)).to(dev)]).cpu())
    return np.asarray(torch.cat(out).numpy(), dtype=np.float64)


def run_node(task: EllipticTask, cfg: GNNConfig, out_dir: Path) -> Path:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    assert task.data is not None
    part = task.partition()
    y = task.items["label"].to_numpy(np.int8)
    lab = y >= 0
    idx = {p: np.flatnonzero((part == p) & lab).astype(np.int64) for p in PARTITIONS}
    check_training_rows(task.items["split_group"].to_numpy()[idx["train"]], task.split, task.name)
    x = _standardise(task.items[task.row_columns].to_numpy(np.float32), part == "train")
    # Steps are disconnected, so one graph serves every partition without crossing steps.
    view = _View(x, CappedSampler(task.data.graph, cfg.cap), cfg.hops)
    y_all = torch.from_numpy(y.astype(np.float32))
    y_train = y[idx["train"]]
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    trainer = _Trainer(NodeModel(x.shape[1], cfg), cfg, pos_weight)
    dev = _device()

    def train_epoch(m: nn.Module, opt: torch.optim.Optimizer, loss_fn: nn.Module) -> float:
        total = 0.0
        for chunk in _batches(len(idx["train"]), cfg.batch_size, True, rng):
            seeds = idx["train"][chunk]
            xb, ei, ego, b = view.tensors(seeds, dev)
            opt.zero_grad()
            logit = m(xb, ei, _seed_ego(cfg, ego))[torch.from_numpy(local_ids(b, seeds)).to(dev)]
            loss = loss_fn(logit, y_all[torch.from_numpy(seeds)].to(dev))
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(seeds)
        return total / len(idx["train"])

    info = trainer.fit(train_epoch, lambda m: _score_nodes(m, view, idx["val"], cfg), y[idx["val"]])
    scores = _score_nodes(trainer.model, view, idx["test"], cfg)
    return _write(task, idx["test"], scores, cfg, info, out_dir)


def run_node_steps(task: EllipticPPTask, cfg: GNNConfig, out_dir: Path) -> Path:
    """Elliptic++: each address is scored on the graph and the node features as of the step it
    first appears in (consort.data.ellipticpp). Batches never mix first steps; step views are
    built on demand and the last three kept."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    data = task.data
    assert data is not None
    part = task.partition()
    y = task.items["label"].to_numpy(np.int8)
    lab = y >= 0
    idx = {p: np.flatnonzero((part == p) & lab).astype(np.int64) for p in PARTITIONS}
    check_training_rows(task.items["split_group"].to_numpy()[idx["train"]], task.split, task.name)
    rows = task.items[task.row_columns].to_numpy(np.float32)
    mu = rows[idx["train"]].mean(axis=0)
    sd = rows[idx["train"]].std(axis=0) + 1e-6
    first = data.first_step
    views: dict[int, _View] = {}

    def view(step: int) -> _View:
        if step not in views:
            if len(views) >= 3:
                views.pop(next(iter(views)))
            nodes = data.node_table(data.inc_step <= step).to_numpy(np.float32)
            x = torch.from_numpy(((nodes - mu) / sd).astype(np.float32))
            views[step] = _View(x, CappedSampler(data.graph_upto(step), cfg.cap), cfg.hops)
        return views[step]

    def batches(
        items: npt.NDArray[np.int64], shuffle: bool
    ) -> list[tuple[int, npt.NDArray[np.int64]]]:
        out = []
        for st in np.unique(first[items]):
            of = items[first[items] == st]
            out += [(int(st), of[c]) for c in _batches(len(of), cfg.batch_size, shuffle, rng)]
        return [out[i] for i in rng.permutation(len(out))] if shuffle else out

    y_all = torch.from_numpy(y.astype(np.float32))
    y_train = y[idx["train"]]
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    trainer = _Trainer(NodeModel(rows.shape[1], cfg), cfg, pos_weight)
    dev = _device()

    def logits(m: nn.Module, step: int, seeds: npt.NDArray[np.int64]) -> Tensor:
        xb, ei, ego, b = view(step).tensors(seeds, dev)
        out: Tensor = m(xb, ei, _seed_ego(cfg, ego))[torch.from_numpy(local_ids(b, seeds)).to(dev)]
        return out

    def train_epoch(m: nn.Module, opt: torch.optim.Optimizer, loss_fn: nn.Module) -> float:
        total = 0.0
        for step, seeds in batches(idx["train"], True):
            opt.zero_grad()
            loss = loss_fn(logits(m, step, seeds), y_all[torch.from_numpy(seeds)].to(dev))
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(seeds)
        return total / len(idx["train"])

    def score(m: nn.Module, p: Partition) -> npt.NDArray[np.float64]:
        m.eval()
        out = np.full(len(y), np.nan)
        for step, seeds in batches(idx[p], False):
            with torch.no_grad():
                out[seeds] = torch.sigmoid(logits(m, step, seeds)).cpu().numpy()
        return out[idx[p]]

    info = trainer.fit(train_epoch, lambda m: score(m, "val"), y[idx["val"]])
    return _write(task, idx["test"], score(trainer.model, "test"), cfg, info, out_dir)


def _graph_view(g: Graph, cfg: GNNConfig) -> _View:
    # Node inputs are already log-scaled volume statistics of the same graph (ibm_node_table).
    x = torch.from_numpy(ibm_node_table(g).to_numpy(np.float32))
    return _View(x, CappedSampler(g, cfg.cap), cfg.hops)


class _EdgeViews:
    """The graph a transfer's neighbourhood is drawn from, as arm G's features see it: every
    transfer from the days before its own (the task's prior-days view), or, for the gap, the
    whole window of its partition. Day graphs are built on demand and the last few kept."""

    def __init__(self, task: IBMTask, cfg: GNNConfig, keep: int = 3) -> None:
        self.task, self.cfg, self.keep = task, cfg, keep
        self.day = task.items["split_group"].to_numpy()
        self._views: dict[tuple[str, int], _View] = {}

    def __call__(self, partition: Partition, day: int) -> _View:
        window = self.task.view == "window"
        key: tuple[str, int] = (partition, -1) if window else ("day", day)
        if key not in self._views:
            if not window and len(self._views) >= self.keep:
                self._views.pop(next(iter(self._views)))
            g = (
                self.task.window_graph(partition)
                if window
                else self.task._graph_upto(self.day < day)
            )
            self._views[key] = _graph_view(g, self.cfg)
        return self._views[key]

    def batches(
        self, items: npt.NDArray[np.int64], size: int, shuffle: bool, rng: np.random.Generator
    ) -> list[tuple[int, npt.NDArray[np.int64]]]:
        """Batches that never mix days, so each is scored on one graph; shuffled across days
        when training, in day order otherwise."""
        out = []
        for d in np.unique(self.day[items]):
            of_day = items[self.day[items] == d]
            out += [(int(d), of_day[c]) for c in _batches(len(of_day), size, shuffle, rng)]
        if shuffle:
            out = [out[i] for i in rng.permutation(len(out))]
        return out


def _edge_logits(
    m: nn.Module,
    view: _View,
    items: npt.NDArray[np.int64],
    src: npt.NDArray[np.int64],
    dst: npt.NDArray[np.int64],
    edge_x: Tensor,
    cfg: GNNConfig,
    dev: torch.device,
) -> Tensor:
    s, d = src[items], dst[items]
    x, ei, ego, b = view.tensors(np.concatenate([s, d]), dev)
    pair = torch.from_numpy(np.stack([local_ids(b, s), local_ids(b, d)])).to(dev)
    out: Tensor = m(x, ei, _seed_ego(cfg, ego), pair, edge_x[torch.from_numpy(items)].to(dev))
    return out


def run_edge(task: IBMTask, cfg: GNNConfig, out_dir: Path) -> Path:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    assert task.data is not None
    part = task.partition()
    y = task.items["label"].to_numpy(np.int8)
    t = task.data.transfers
    src, dst = t["src"].to_numpy(np.int64), t["dst"].to_numpy(np.int64)
    items = {p: np.flatnonzero(part == p).astype(np.int64) for p in PARTITIONS}
    check_training_rows(task.items["split_group"].to_numpy()[items["train"]], task.split, task.name)
    views = _EdgeViews(task, cfg)
    edge_x = _standardise(task.items[task.row_columns].to_numpy(np.float32), part == "train")
    y_all = torch.from_numpy(y.astype(np.float32))
    y_train = y[items["train"]]
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    model = EdgeModel(len(IBM_NODE_COLUMNS), edge_x.shape[1], cfg)
    trainer = _Trainer(model, cfg, pos_weight)
    dev = _device()

    def train_epoch(m: nn.Module, opt: torch.optim.Optimizer, loss_fn: nn.Module) -> float:
        total = 0.0
        for day, batch in views.batches(items["train"], cfg.batch_size, True, rng):
            opt.zero_grad()
            logit = _edge_logits(m, views("train", day), batch, src, dst, edge_x, cfg, dev)
            loss = loss_fn(logit, y_all[torch.from_numpy(batch)].to(dev))
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(batch)
        return total / len(items["train"])

    def score(m: nn.Module, p: Partition) -> npt.NDArray[np.float64]:
        m.eval()
        out = np.full(len(y), np.nan)
        for day, batch in views.batches(items[p], cfg.batch_size, False, rng):
            with torch.no_grad():
                logit = _edge_logits(m, views(p, day), batch, src, dst, edge_x, cfg, dev)
            out[batch] = torch.sigmoid(logit).cpu().numpy()
        # Batches run day by day; the scores go back in the partition's item order.
        return out[items[p]]

    info = trainer.fit(train_epoch, lambda m: score(m, "val"), y[items["val"]])
    return _write(task, items["test"], score(trainer.model, "test"), cfg, info, out_dir)


def _write(
    task: Task,
    test: npt.NDArray[np.int64],
    scores: npt.NDArray[np.float64],
    cfg: GNNConfig,
    info: dict[str, Any],
    out_dir: Path,
) -> Path:
    items = task.items
    outcomes = oc.Outcomes.build(
        items["item_id"].to_numpy()[test],
        items["boot_group"].to_numpy()[test],
        items["label"].to_numpy()[test],
        scores,
    )
    run = f"{task.name}.temporal.{cfg.tag}.h{cfg.hops}.cap{cfg.cap}.s{cfg.seed}"
    meta = oc.RunMeta(
        run=run,
        dataset=task.name,
        split=task.split.name,
        arm=cfg.tag,
        model=f"pyg-{cfg.arch}",
        hops=cfg.hops,
        seed=cfg.seed,
        val_auprc=float(info["best_val_auprc"]),
        notes="; ".join(f"{k}={v}" for k, v in {**asdict(cfg), **info}.items()),
    )
    return oc.write(outcomes, meta, out_dir)


def run(task: Task, cfg: GNNConfig, out_dir: Path) -> Path:
    if isinstance(task, EllipticTask):
        return run_node(task, cfg, out_dir)
    if isinstance(task, IBMTask):
        return run_edge(task, cfg, out_dir)
    if isinstance(task, EllipticPPTask):
        return run_node_steps(task, cfg, out_dir)
    raise TypeError(f"no GNN task for {type(task).__name__}")
