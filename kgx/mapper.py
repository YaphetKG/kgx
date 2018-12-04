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

class Clique(set):
    def add_all(e):
        if isinstance(e, (list, set, tuple)):
            self.update(e)
        else:
            self.add(e)

    def get_leader():
        l = list(self)
        l.sort()
        return l[0]

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
    cliques = []

    with click.progressbar(graph.nodes(), label='building cliques') as bar:
        for n in bar:
            clique = Clique()
            clique.add_all(n)

            attr_dict = graph.node[n]
            if 'same_as' in attr_dict:
                clique.add_all(attr_dict['same_as'])

    with click.progressbar(list(range(len(cliques))), label='reducing cliques') as bar:
        for _ in bar:
            popped_clique = cliques.pop(0)

            for clique in cliques:
                if any(e in clique for e in popped_clique):
                    clique.add_all(popped_clique)
                    break
            else:
                cliques.append(popped_clique)

    mapping = {}

    with click.progressbar(graph.nodes(), label='building mapping') as bar:
        for n in bar:
            for clique in cliques:
                if n in clique:
                    mapping[n] = clique.get_leader()
                    break
            else:
                logging.warn('No clique found for {}'.format(n))

    return relabel_nodes(graph, mapping)
