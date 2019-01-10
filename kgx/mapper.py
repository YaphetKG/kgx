import networkx as nx
import logging, click

from collections import defaultdict

def map_graph(G, mapping, preserve=True):
    if preserve:
        for nid in G.nodes_iter():
            if nid in mapping:
                # add_node will append attributes
                G.add_node(nid, source_curie=nid)
        for oid,sid in G.edges_iter():
            if oid in mapping:
                for ex in G[oid][sid]:
                    G[oid][sid][ex].update(source_object=oid)
            if sid in mapping:
                for ex in G[oid][sid]:
                    G[oid][sid][ex].update(source_subject=oid)
    nx.relabel_nodes(G, mapping, copy=False)

def relabel_nodes(graph:nx.Graph, mapping:dict) -> nx.Graph:
    """
    Performs the relabelling of nodes, and ensures that list properties are
    copied over.

    Example:
        graph = nx.Graph()

        graph.add_edge('a', 'b')
        graph.add_edge('c', 'd')

        graph.node['a']['name'] = ['A']
        graph.node['b']['name'] = ['B']
        graph.node['c']['name'] = ['C']
        graph.node['d']['name'] = ['D']

        graph = relabel_nodes(graph, {'c' : 'b'})

        for n in graph.nodes():
            print(n, graph.node[n])
    Output:
        a {'name': ['A']}
        b {'name': ['B', 'C']}
        d {'name': ['D']}
    """
    g = nx.relabel_nodes(graph, mapping, copy=True)

    for n in g.nodes():
        d = g.node[n]
        attr = graph.node[n]

        for key, value in attr.items():
            if key in d:
                if isinstance(d[key], (list, set, tuple)) and isinstance(attr[key], (list, set, tuple)):
                    s = set(d[key])
                    s.update(attr[key])
                    d[key] = list(s)
            else:
                d[key] = value
    return g

def clique_merge(graph:nx.Graph) -> nx.Graph:
    """
    Builds up cliques using the `same_as` attribute of each node. Uses those
    cliques to build up a mapping for relabelling nodes. Chooses labels so as
    to preserve the original nodes, rather than taking xrefs that don't appear
    as nodes in the graph.
    """
    cliqueGraph = nx.Graph()

    with click.progressbar(graph.nodes(), label='building cliques') as bar:
        for n in bar:
            attr_dict = graph.node[n]
            if 'same_as' in attr_dict:
                for m in attr_dict['same_as']:
                    cliqueGraph.add_edge(n, m)

    mapping = {}

    with click.progressbar(list(nx.connected_components(cliqueGraph)), label='building mapping') as bar:
        for component in bar:
            nodes = list(c for c in component if c in graph)
            nodes.sort()
            for n in nodes:
                if n != nodes[0]:
                    mapping[n] = nodes[0]

    return relabel_nodes(graph, mapping)
