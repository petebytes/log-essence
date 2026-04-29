"""Minimal Drain log-template miner.

Reimplementation of the Drain algorithm (He et al., ICWS 2017) tailored to
log-essence's needs. Replaces the upstream `drain3` package, whose PyPI
release pins `cachetools==4.2.1` and conflicts with `fastmcp>=3.2.0`
(which needs `cachetools>=5`).

API surface intentionally narrow — only what server.py uses:
    miner = TemplateMiner(config)
    result = miner.add_log_message(line)        # -> {"cluster_id": int, ...}
    for cluster in miner.drain.clusters: ...    # cluster_id, size, get_template()

Algorithm reference: https://github.com/logpai/Drain3 (MIT-licensed).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

PARAM = "<*>"


@dataclass
class TemplateMinerConfig:
    """Tunables matching drain3 field names so server.py needs no changes."""

    drain_sim_th: float = 0.4
    drain_depth: int = 4
    drain_max_children: int = 100
    drain_max_clusters: int | None = None
    drain_extra_delimiters: tuple[str, ...] = ()
    parametrize_numeric_tokens: bool = True


class _LogCluster:
    __slots__ = ("cluster_id", "log_template_tokens", "size")

    def __init__(self, tokens: list[str], cluster_id: int) -> None:
        self.log_template_tokens: tuple[str, ...] = tuple(tokens)
        self.cluster_id = cluster_id
        self.size = 1

    def get_template(self) -> str:
        return " ".join(self.log_template_tokens)


@dataclass
class _Node:
    key_to_child: dict[str, _Node] = field(default_factory=dict)
    cluster_ids: list[int] = field(default_factory=list)


class _LRU(OrderedDict):
    """Tiny LRU mapping. Bypass-eviction read via plain dict access."""

    def __init__(self, maxsize: int) -> None:
        super().__init__()
        self.maxsize = maxsize

    def __getitem__(self, key):  # type: ignore[override]
        value = OrderedDict.__getitem__(self, key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value) -> None:  # type: ignore[override]
        if key in self:
            self.move_to_end(key)
        OrderedDict.__setitem__(self, key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]

    def peek(self, key):
        """Read without LRU touch (matches drain3's bypass semantics)."""
        return OrderedDict.__getitem__(self, key) if OrderedDict.__contains__(self, key) else None


class Drain:
    def __init__(
        self,
        depth: int = 4,
        sim_th: float = 0.4,
        max_children: int = 100,
        max_clusters: int | None = None,
        extra_delimiters: tuple[str, ...] = (),
        parametrize_numeric_tokens: bool = True,
    ) -> None:
        if depth < 3:
            raise ValueError("depth must be >= 3")
        self.log_cluster_depth = depth
        self.max_node_depth = depth - 2
        self.sim_th = sim_th
        self.max_children = max_children
        self.extra_delimiters = extra_delimiters
        self.parametrize_numeric_tokens = parametrize_numeric_tokens
        self.root_node = _Node()
        self.id_to_cluster: dict[int, _LogCluster] | _LRU = (
            _LRU(maxsize=max_clusters) if max_clusters else {}
        )
        self.clusters_counter = 0

    @property
    def clusters(self):
        return self.id_to_cluster.values()

    @staticmethod
    def _has_numbers(s: str) -> bool:
        return any(c.isdigit() for c in s)

    def _get_cluster(self, cluster_id: int) -> _LogCluster | None:
        if isinstance(self.id_to_cluster, _LRU):
            return self.id_to_cluster.peek(cluster_id)
        return self.id_to_cluster.get(cluster_id)

    def _tokenize(self, content: str) -> list[str]:
        content = content.strip()
        for d in self.extra_delimiters:
            content = content.replace(d, " ")
        return content.split()

    def _tree_search(self, tokens: list[str]) -> _LogCluster | None:
        token_count = len(tokens)
        node = self.root_node.key_to_child.get(str(token_count))
        if node is None:
            return None
        if token_count == 0:
            return self._get_cluster(node.cluster_ids[0]) if node.cluster_ids else None

        depth = 1
        for token in tokens:
            if depth >= self.max_node_depth or depth == token_count:
                break
            children = node.key_to_child
            nxt = children.get(token) or children.get(PARAM)
            if nxt is None:
                return None
            node = nxt
            depth += 1

        return self._fast_match(node.cluster_ids, tokens)

    def _seq_distance(
        self, template: tuple[str, ...], tokens: list[str]
    ) -> tuple[float, int]:
        if not template:
            return 1.0, 0
        sim = 0
        params = 0
        for t1, t2 in zip(template, tokens, strict=False):
            if t1 == PARAM:
                params += 1
            elif t1 == t2:
                sim += 1
        return sim / len(template), params

    def _fast_match(self, cluster_ids: list[int], tokens: list[str]) -> _LogCluster | None:
        best = None
        best_sim = -1.0
        best_params = -1
        for cid in cluster_ids:
            cluster = self._get_cluster(cid)
            if cluster is None or len(cluster.log_template_tokens) != len(tokens):
                continue
            sim, params = self._seq_distance(cluster.log_template_tokens, tokens)
            if sim > best_sim or (sim == best_sim and params > best_params):
                best_sim = sim
                best_params = params
                best = cluster
        return best if best_sim >= self.sim_th else None

    def _add_to_tree(self, cluster: _LogCluster) -> None:
        token_count = len(cluster.log_template_tokens)
        key = str(token_count)
        first = self.root_node.key_to_child.get(key)
        if first is None:
            first = _Node()
            self.root_node.key_to_child[key] = first

        node = first
        if token_count == 0:
            node.cluster_ids = [cluster.cluster_id]
            return

        depth = 1
        for token in cluster.log_template_tokens:
            if depth >= self.max_node_depth or depth >= token_count:
                live_ids = [
                    cid for cid in node.cluster_ids if self._get_cluster(cid) is not None
                ]
                live_ids.append(cluster.cluster_id)
                node.cluster_ids = live_ids
                break

            if token in node.key_to_child:
                node = node.key_to_child[token]
            else:
                use_param = (
                    self.parametrize_numeric_tokens and self._has_numbers(token)
                )
                if use_param:
                    nxt = node.key_to_child.get(PARAM)
                    if nxt is None:
                        nxt = _Node()
                        node.key_to_child[PARAM] = nxt
                    node = nxt
                else:
                    has_param_child = PARAM in node.key_to_child
                    children_count = len(node.key_to_child)
                    if has_param_child:
                        if children_count < self.max_children:
                            new = _Node()
                            node.key_to_child[token] = new
                            node = new
                        else:
                            node = node.key_to_child[PARAM]
                    else:
                        if children_count + 1 < self.max_children:
                            new = _Node()
                            node.key_to_child[token] = new
                            node = new
                        elif children_count + 1 == self.max_children:
                            new = _Node()
                            node.key_to_child[PARAM] = new
                            node = new
                        else:
                            node = node.key_to_child[PARAM]

            depth += 1

    def _merge_template(
        self, tokens: list[str], template: tuple[str, ...]
    ) -> list[str]:
        out = list(tokens)
        for i, (t1, t2) in enumerate(zip(tokens, template, strict=False)):
            if t1 != t2:
                out[i] = PARAM
        return out

    def add_log_message(self, content: str) -> tuple[_LogCluster, str]:
        tokens = self._tokenize(content)
        match = self._tree_search(tokens)

        if match is None:
            self.clusters_counter += 1
            cid = self.clusters_counter
            cluster = _LogCluster(tokens, cid)
            self.id_to_cluster[cid] = cluster
            self._add_to_tree(cluster)
            return cluster, "cluster_created"

        new_template = self._merge_template(tokens, match.log_template_tokens)
        change_type = "none"
        if tuple(new_template) != match.log_template_tokens:
            match.log_template_tokens = tuple(new_template)
            change_type = "cluster_template_changed"
        match.size += 1
        if isinstance(self.id_to_cluster, _LRU):
            self.id_to_cluster[match.cluster_id] = match  # touch
        return match, change_type


class TemplateMiner:
    """Drop-in replacement for `drain3.TemplateMiner` (subset).

    log-essence does not need masking, persistence, or parameter extraction —
    those are handled elsewhere or unused. Surface limited to what server.py
    invokes.
    """

    def __init__(self, config: TemplateMinerConfig | None = None) -> None:
        config = config or TemplateMinerConfig()
        self.config = config
        self.drain = Drain(
            depth=config.drain_depth,
            sim_th=config.drain_sim_th,
            max_children=config.drain_max_children,
            max_clusters=config.drain_max_clusters,
            extra_delimiters=config.drain_extra_delimiters,
            parametrize_numeric_tokens=config.parametrize_numeric_tokens,
        )

    def add_log_message(self, log_message: str) -> dict:
        cluster, change_type = self.drain.add_log_message(log_message)
        return {
            "change_type": change_type,
            "cluster_id": cluster.cluster_id,
            "cluster_size": cluster.size,
            "template_mined": cluster.get_template(),
            "cluster_count": len(list(self.drain.clusters)),
        }
