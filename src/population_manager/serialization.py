import numpy as np
import sys

                                                                      
try:
    from src.core.evolution.genome import (
        PrimitiveNode,
        OpNode,
        PrimitiveGene,
        TransformParams,
        CompositionGenome,
        OperatorKind,
    )
except ImportError:
    try:
        from evolution.genome import (                        
            PrimitiveNode,
            OpNode,
            PrimitiveGene,
            TransformParams,
            CompositionGenome,
            OperatorKind,
        )
    except ImportError as e:
        print(f"Critical error: Could not import 'evolution.genome'. {e}")
        sys.exit(1)

                                        

def _gene_to_dict(g) -> dict:
    """Serialize a PrimitiveGene instance into a dictionary."""
    d = {
        "kind": g.kind,
        "transform": {
            "sx": float(g.transform.sx), "sy": float(g.transform.sy),
            "theta": float(g.transform.theta), "dx": float(g.transform.dx), "dy": float(g.transform.dy),
        },
        "color_rgb": [float(x) for x in (g.color_rgb.tolist() if hasattr(g.color_rgb, "tolist") else g.color_rgb)],
        "polygon_vertices": None,
    }
    if g.polygon_vertices is not None:
        d["polygon_vertices"] = [[float(a), float(b)] for a, b in g.polygon_vertices.tolist()]
    return d

def _node_to_dict(node) -> dict:
    """Recursively serialize a composition tree node into a dictionary."""
    if isinstance(node, PrimitiveNode):
        return {"primitive": _gene_to_dict(node.gene)}
    if isinstance(node, OpNode):
        return {"op": node.kind, "children": [_node_to_dict(ch) for ch in node.children]}
    return {"unknown": True}

def _population_to_dict(population) -> dict:
    """Serialize a list of genomes (population) into a dictionary."""
    return {
        "population": [
            {"index": i, "composition": _node_to_dict(genome.root)}
            for i, genome in enumerate(population)
        ]
    }

                                             
                                                                                   

def _dict_to_node(d) -> PrimitiveNode | OpNode:
    """Deserialize a dictionary into either a PrimitiveNode or an OpNode."""
    if "primitive" in d:
        g = d["primitive"]
        t = g["transform"]
        gene = PrimitiveGene(
            kind=g["kind"],
            transform=TransformParams(
                sx=float(t["sx"]),
                sy=float(t["sy"]),
                theta=float(t["theta"]),
                dx=float(t["dx"]),
                dy=float(t["dy"]),
            ),
            color_rgb=np.array(g.get("color_rgb", [0.6, 0.6, 0.6]), dtype=float),
            polygon_vertices=None if g.get("polygon_vertices") is None else np.array(g["polygon_vertices"], dtype=float),
        )
        return PrimitiveNode(gene=gene)
    if "op" in d and "children" in d:
        op_value = str(d["op"])
        if op_value == "union":
            op_kind: "OperatorKind" = "union"
        elif op_value == "intersection":
            op_kind = "intersection"
        elif op_value == "difference":
            op_kind = "difference"
        else:
            raise ValueError(f"Invalid operator in JSON: {op_value}")
        return OpNode(kind=op_kind, children=[_dict_to_node(c) for c in d["children"]])
    raise ValueError("Invalid composition node in JSON file")

def _dict_to_genome(comp_dict) -> CompositionGenome:
    """Deserialize a composition dictionary into a full genome."""
    node = _dict_to_node(comp_dict)
    return CompositionGenome(root=node)