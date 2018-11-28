import click, rdflib, logging

from rdflib import Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL

from .transformer import Transformer
from .utils.rdf_utils import find_category, category_mapping, property_mapping, make_curie

from collections import defaultdict

class RdfTransformer(Transformer):
    def __init__(self, t:Transformer=None):
        super().__init__(t)
        self.ontologies = []

    def parse(self, filename:str=None):
        rdfgraph = rdflib.Graph()
        rdfgraph.parse(filename, format=rdflib.util.guess_format(filename))
        logging.info("Parsed : {}".format(filename))
        self.load_edges(rdfgraph)
        self.load_nodes(rdfgraph)

    def add_ontology(self, owlfile:str):
        ont = rdflib.Graph()
        ont.parse(owlfile, format=rdflib.util.guess_format(owlfile))
        self.ontologies.append(ont)
        logging.info("Parsed : {}".format(owlfile))

    def load_edges(self, rdfgraph:rdflib.Graph):
        pass

    def load_nodes(self, rdfgraph:rdflib.Graph):
        with click.progressbar(self.graph.nodes(), label='loading nodes') as bar:
            for node_id in bar:
                node_attr = defaultdict(set)

                if 'iri' in self.graph.node[node_id]:
                    iri = self.graph.node[node_id]['iri']
                else:
                    continue

                for s, p, o in rdfgraph.triples((URIRef(iri), None, None)):
                    if p in property_mapping or isinstance(o, rdflib.term.Literal):
                        p = property_mapping.get(p, make_curie(p))
                        o = make_curie(o)

                        if p in node_attr:
                            if isinstance(node_attr[p], set):
                                node_attr[p].add(o)
                            else:
                                node_attr[p] = {node_attr[p], o}
                        else:
                            node_attr[p] = o

                c = find_category(rdfgraph, iri)
                if c is not None:
                    node_attr['category'] = [c]
                else:
                    for ont in self.ontologies:
                        c = find_category(ont, iri)
                        if c is not None:
                            node_attr['category'] = [c]
                            break

                for key, value in node_attr.items():
                    self.graph.node[node_id][key] = value

class ObanRdfTransformer(RdfTransformer):
    OBAN = Namespace('http://purl.org/oban/')

    def load_edges(self, rdfgraph:rdflib.Graph):
        associations = list(rdfgraph.subjects(RDF.type, self.OBAN.association))
        with click.progressbar(associations, label='loading edges') as bar:
            for association in bar:
                edge_attr = defaultdict(list)

                for s, p, o in rdfgraph.triples((association, None, None)):
                    if p in property_mapping:
                        p = property_mapping[p]
                        edge_attr[p].append(str(o))
                    elif isinstance(o, rdflib.term.Literal):
                        edge_attr[make_curie(p)].append(str(o))

                subjects = edge_attr['subject']
                objects = edge_attr['object']

                for key, value in edge_attr.items():
                    if isinstance(value, (list, str, tuple)):
                        edge_attr[key] = {make_curie(v) for v in value}
                    else:
                        edge_attr[key] = make_curie(value)

                for subject_iri in subjects:
                    for object_iri in objects:
                        sid = make_curie(subject_iri)
                        oid = make_curie(object_iri)

                        self.graph.add_edge(sid, oid, **edge_attr)

                        self.graph.node[sid]['iri'] = subject_iri
                        self.graph.node[sid]['id'] = sid

                        self.graph.node[oid]['iri'] = object_iri
                        self.graph.node[oid]['id'] = oid
