new_attr_dictimport networkx as nx
import logging, click, bmt

from collections import defaultdict
from typing import Union, List

bmt.load('https://biolink.github.io/biolink-model/biolink-model.yaml')

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
    Performs the relabelling of nodes, and ensures that list attributes are
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
    print('relabelling nodes...')
    g = nx.relabel_nodes(graph, mapping, copy=True)

    with click.progressbar(graph.nodes(), label='concatenating list attributes') as bar:
        for n in bar:
            if n not in mapping or n == mapping[n]:
                continue

            new_attr_dict = g.node[mapping[n]]
            old_attr_dict = graph.node[n]

            for key, value in old_attr_dict.items():
                if key in new_attr_dict:
                    is_list = \
                        isinstance(new_attr_dict[key], (list, set, tuple)) \
                        and isinstance(old_attr_dict[key], (list, set, tuple))
                    if is_list:
                        s = set(new_attr_dict[key])
                        s.update(old_attr_dict[key])
                        new_attr_dict[key] = list(s)
                else:
                    new_attr_dict[key] = value
    return g

def listify(o:object) -> Union[list, set, tuple]:
    if isinstance(o, (list, set, tuple)):
        return o
    else:
        return [o]

def get_prefix(curie:str) -> str:
    if ':' in curie:
        prefix, _ = curie.rsplit(':', 1)
        return prefix
    else:
        return None

def sort_key(n, list_of_prefixes:List[List[str]]):
    """
    For a list of lists of prefixes, gets the lowest
    index of a matching prefix.
    """
    k = len(list_of_prefixes) + 1
    p = get_prefix(n).upper()
    for prefixes in list_of_prefixes:
        for i, prefix in enumerate(prefixes):
            if p == prefix.upper():
                if i < k:
                    k = i
    return k

def clique_merge(graph:nx.Graph) -> nx.Graph:
    """
    Builds up cliques using the `same_as` attribute of each node. Uses those
    cliques to build up a mapping for relabelling nodes. Chooses labels so as
    to preserve the original nodes, rather than taking xrefs that don't appear
    as nodes in the graph.

    This method will also expand the `same_as` attribute of the clique leader.
    """

    original_size = len(graph)
    print('original graph has {} nodes'.format(original_size))

    cliqueGraph = nx.Graph()

    with click.progressbar(graph.nodes(), label='building cliques') as bar:
        for n in bar:
            attr_dict = graph.node[n]
            if 'same_as' in attr_dict:
                for m in attr_dict['same_as']:
                    cliqueGraph.add_edge(n, m)

    mapping = {}

    connected_components = list(nx.connected_components(cliqueGraph)

    print('Discovered {} cliques'.format(len(connected_components))

    with click.progressbar(connected_components, label='building mapping') as bar:
        for nodes in bar:
            categories = set()
            for n in nodes:
                attr_dict = graph.node[n]

                if 'category' in attr_dict:
                    categories.addAll(listify(attr_dict['category']))

                if 'categories' in attr_dict:
                    categories.addAll(listify(attr_dict['categories']))

            list_of_prefixes = []
            for category in categories:
                try:
                    list_of_prefixes.append(bmt.get_element(category).id_prefixes)
                except:
                    pass

            nodes.sort(key=sort_key(n, list_of_prefixes))

            for n in nodes:
                if n != nodes[0]:
                    mapping[n] = nodes[0]

            graph.node[nodes[0]]['same_as'] = nodes

    g = relabel_nodes(graph, mapping)

    final_size = len(g)
    print('Resulting graph has {} nodes'.format(final_size))
    print('Eliminated {} nodes'.format(original_size - final_size))

    return g
