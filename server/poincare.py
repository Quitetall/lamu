"""
Poincaré ball embedding for knowledge graphs.

Takes graphify's graph.json and embeds it into hyperbolic space:
- God nodes (high degree) → near the origin (center of the disk)
- Leaf nodes → near the boundary
- Hierarchical depth = distance from center
- Clusters visible as angular sectors

Produces an interactive Plotly visualization (HTML) of the Poincaré disk.

Usage:
    python -m server.poincare graphify-out/graph.json
    python -m server.poincare graphify-out/graph.json --output poincare.html
    python -m server.poincare graphify-out/graph.json --dim 3  # 3D ball
"""

import argparse
import json
import math
from pathlib import Path

import geoopt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import torch
import torch.optim as optim


# ── Poincaré Ball Embedding ─────────────────────────────────────────────

class PoincareBallEmbedding:
    """Embed a graph into the Poincaré ball model of hyperbolic space."""

    def __init__(self, n_nodes: int, dim: int = 2, curvature: float = 1.0):
        self.dim = dim
        self.ball = geoopt.PoincareBall(c=curvature)
        # Initialize embeddings near origin with small random noise
        init = torch.randn(n_nodes, dim) * 0.01
        self.embeddings = geoopt.ManifoldParameter(
            self.ball.expmap0(init), manifold=self.ball
        )

    def distance(self, i: int, j: int) -> torch.Tensor:
        """Hyperbolic distance between nodes i and j."""
        return self.ball.dist(self.embeddings[i], self.embeddings[j])

    def train(
        self,
        edges: list[tuple[int, int]],
        weights: list[float] = None,
        epochs: int = 300,
        lr: float = 0.01,
        neg_samples: int = 10,
        n_nodes: int = 0,
    ):
        """Train embeddings using hyperbolic distance loss."""
        if weights is None:
            weights = [1.0] * len(edges)

        optimizer = geoopt.optim.RiemannianAdam([self.embeddings], lr=lr)

        edges_t = torch.tensor(edges, dtype=torch.long)
        weights_t = torch.tensor(weights, dtype=torch.float32)

        print(f"  Training Poincaré embeddings ({epochs} epochs, {len(edges)} edges)...")

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Positive loss: connected nodes should be close
            pos_dists = torch.stack([
                self.ball.dist(self.embeddings[e[0]], self.embeddings[e[1]])
                for e in edges_t
            ])
            pos_loss = (weights_t * pos_dists).mean()

            # Negative sampling: random pairs should be far apart
            neg_i = torch.randint(0, n_nodes, (neg_samples * len(edges),))
            neg_j = torch.randint(0, n_nodes, (neg_samples * len(edges),))
            neg_dists = torch.stack([
                self.ball.dist(self.embeddings[ni], self.embeddings[nj])
                for ni, nj in zip(neg_i, neg_j)
            ])
            neg_loss = torch.clamp(2.0 - neg_dists, min=0).mean()

            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}/{epochs}: loss={loss.item():.4f}")

        return self.embeddings.detach().numpy()


# ── Graph Loading ────────────────────────────────────────────────────────

def load_graphify_json(path: str) -> nx.Graph:
    """Load graphify's graph.json into a NetworkX graph."""
    data = json.loads(Path(path).read_text())

    G = nx.Graph()

    # Nodes
    for node in data.get("nodes", []):
        node_id = node.get("id", node.get("label", ""))
        G.add_node(node_id, **{
            "label": node.get("label", node_id),
            "type": node.get("type", "unknown"),
            "community": node.get("community", 0),
            "file": node.get("file", ""),
        })

    # Edges
    for edge in data.get("edges", []):
        source = edge.get("source", "")
        target = edge.get("target", "")
        if source and target and source in G and target in G:
            G.add_edge(source, target, **{
                "relation": edge.get("relation", ""),
                "type": edge.get("type", "EXTRACTED"),
                "confidence": edge.get("confidence_score", 1.0),
            })

    return G


STDLIB_MODULES = {
    "os", "sys", "json", "re", "math", "time", "datetime", "pathlib",
    "typing", "collections", "functools", "itertools", "io", "abc",
    "dataclasses", "enum", "copy", "hashlib", "uuid", "struct",
    "subprocess", "shutil", "tempfile", "argparse", "logging",
    "contextlib", "inspect", "importlib", "pkgutil", "warnings",
    "asyncio", "threading", "multiprocessing", "concurrent",
    "urllib", "http", "socket", "ssl", "email",
    "unittest", "pytest", "textwrap", "string", "operator",
    "ctypes", "signal", "atexit", "traceback", "pprint",
}

THIRD_PARTY_NOISE = {
    "torch", "numpy", "np", "pandas", "scipy", "sklearn",
    "transformers", "pydantic", "fastapi", "uvicorn", "starlette",
    "rich", "click", "tqdm", "requests", "httpx", "aiohttp",
    "PIL", "cv2", "matplotlib", "plotly", "networkx",
    "dotenv", "yaml", "toml",
}


def load_networkx_from_codebase(path: str, filter_noise: bool = True) -> nx.Graph:
    """Build a graph from Python AST. Filters stdlib/third-party noise by default."""
    import ast

    G = nx.Graph()
    repo = Path(path).resolve()

    # Assign community by top-level directory
    community_map = {}
    community_counter = 0

    py_files = list(repo.rglob("*.py"))
    for f in py_files:
        if any(skip in str(f) for skip in [".venv", "__pycache__", "lucebox-hub", "node_modules"]):
            continue

        rel = str(f.relative_to(repo))
        top_dir = rel.split("/")[0] if "/" in rel else "root"
        if top_dir not in community_map:
            community_map[top_dir] = community_counter
            community_counter += 1
        comm = community_map[top_dir]

        G.add_node(rel, label=f.stem, type="file", community=comm, file=rel)

        try:
            tree = ast.parse(f.read_text())
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("_") and node.name != "__init__":
                        continue  # skip private helpers
                    func_id = f"{rel}:{node.name}"
                    G.add_node(func_id, label=node.name, type="function", community=comm, file=rel)
                    G.add_edge(rel, func_id, relation="defines", type="EXTRACTED", confidence=1.0)
                elif isinstance(node, ast.ClassDef):
                    cls_id = f"{rel}:{node.name}"
                    G.add_node(cls_id, label=node.name, type="class", community=comm, file=rel)
                    G.add_edge(rel, cls_id, relation="defines", type="EXTRACTED", confidence=1.0)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        mod = alias.name.split(".")[0]
                        if filter_noise and mod in STDLIB_MODULES | THIRD_PARTY_NOISE:
                            continue
                        G.add_node(alias.name, label=alias.name, type="module", community=comm, file="")
                        G.add_edge(rel, alias.name, relation="imports", type="EXTRACTED", confidence=1.0)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    mod = node.module.split(".")[0]
                    if filter_noise and mod in STDLIB_MODULES | THIRD_PARTY_NOISE:
                        continue
                    G.add_node(node.module, label=node.module, type="module", community=comm, file="")
                    G.add_edge(rel, node.module, relation="imports", type="EXTRACTED", confidence=1.0)
        except (SyntaxError, UnicodeDecodeError):
            continue

    return G


# ── Visualization ────────────────────────────────────────────────────────

# Color palette for communities
COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]


def create_poincare_plot(
    G: nx.Graph,
    coords: np.ndarray,
    node_list: list[str],
    title: str = "Poincaré Ball — Knowledge Graph",
) -> go.Figure:
    """Create an interactive Plotly figure of the Poincaré disk."""

    # Node properties
    degrees = dict(G.degree())
    communities = nx.get_node_attributes(G, "community")
    labels = nx.get_node_attributes(G, "label")
    types = nx.get_node_attributes(G, "type")

    # Scale node sizes by degree
    max_deg = max(degrees.values()) if degrees else 1
    sizes = [5 + 25 * (degrees.get(n, 1) / max_deg) for n in node_list]

    # Color by community
    node_colors = [COLORS[communities.get(n, 0) % len(COLORS)] for n in node_list]

    # Hover text
    hover = [
        f"<b>{labels.get(n, n)}</b><br>"
        f"Type: {types.get(n, '?')}<br>"
        f"Degree: {degrees.get(n, 0)}<br>"
        f"Community: {communities.get(n, 0)}"
        for n in node_list
    ]

    fig = go.Figure()

    # Draw the unit disk boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta),
        mode="lines",
        line=dict(color="rgba(100,100,100,0.3)", width=1),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Draw edges
    for u, v in G.edges():
        if u in node_list and v in node_list:
            i, j = node_list.index(u), node_list.index(v)
            fig.add_trace(go.Scatter(
                x=[coords[i, 0], coords[j, 0]],
                y=[coords[i, 1], coords[j, 1]],
                mode="lines",
                line=dict(color="rgba(150,150,150,0.15)", width=0.5),
                hoverinfo="skip",
                showlegend=False,
            ))

    # Draw nodes
    fig.add_trace(go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=node_colors,
            line=dict(width=0.5, color="white"),
        ),
        text=[labels.get(n, n)[:20] for n in node_list],
        textposition="top center",
        textfont=dict(size=7, color="rgba(50,50,50,0.8)"),
        hovertext=hover,
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis=dict(
            range=[-1.1, 1.1], showgrid=False, zeroline=False,
            showticklabels=False, scaleanchor="y",
        ),
        yaxis=dict(
            range=[-1.1, 1.1], showgrid=False, zeroline=False,
            showticklabels=False,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="white",
        width=900, height=900,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Poincaré ball embedding for knowledge graphs")
    parser.add_argument("input", nargs="?", default="graphify-out/graph.json",
                        help="Path to graph.json or a code directory")
    parser.add_argument("--output", "-o", default="poincare.html", help="Output HTML file")
    parser.add_argument("--dim", type=int, default=2, help="Embedding dimension (2 or 3)")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--max-nodes", type=int, default=500, help="Max nodes to embed")
    args = parser.parse_args()

    # Load graph
    input_path = Path(args.input)
    if input_path.suffix == ".json" and input_path.exists():
        print(f"Loading graphify graph: {input_path}")
        G = load_graphify_json(str(input_path))
    elif input_path.is_dir():
        print(f"Building graph from codebase: {input_path}")
        G = load_networkx_from_codebase(str(input_path))
    else:
        print(f"Error: {input_path} not found. Run /graphify first or point to a directory.")
        return

    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    # Prune to top nodes by degree if too large
    if G.number_of_nodes() > args.max_nodes:
        top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:args.max_nodes]
        G = G.subgraph([n for n, _ in top_nodes]).copy()
        print(f"  Pruned to top {args.max_nodes} nodes by degree")

    if G.number_of_nodes() == 0:
        print("Error: graph is empty")
        return

    # Create node index
    node_list = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(node_list)}

    # Edge list with weights
    edges = [(node_idx[u], node_idx[v]) for u, v in G.edges() if u in node_idx and v in node_idx]
    weights = [G[u][v].get("confidence", 1.0) for u, v in G.edges() if u in node_idx and v in node_idx]

    if not edges:
        print("Error: no edges in graph")
        return

    # Train Poincaré embedding
    embedder = PoincareBallEmbedding(len(node_list), dim=args.dim)
    coords = embedder.train(
        edges, weights,
        epochs=args.epochs, lr=args.lr,
        neg_samples=5, n_nodes=len(node_list),
    )

    # Visualize
    if args.dim == 2:
        fig = create_poincare_plot(G, coords, node_list)
        fig.write_html(args.output)
        print(f"\n  Poincaré disk saved: {args.output}")
        print(f"  Open in browser to explore interactively.")
    else:
        # 3D ball
        fig = go.Figure(data=[go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers",
            marker=dict(size=3),
            text=[G.nodes[n].get("label", n) for n in node_list],
        )])
        fig.write_html(args.output)
        print(f"\n  Poincaré ball (3D) saved: {args.output}")


if __name__ == "__main__":
    main()
