import click, rdflib, logging, os

from rdflib import Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL

from .transformer import Transformer
from .utils.rdf_utils import find_category, category_mapping, property_mapping, make_curie, predicate_mapping, process_iri

from collections import defaultdict

class RdfTransformer(Transformer):
    def __init__(self, t:Transformer=None):
        super().__init__(t)
        self.ontologies = []

    def parse(self, filename:str=None, provided_by:str=None):
        rdfgraph = rdflib.Graph()
        rdfgraph.parse(filename, format=rdflib.util.guess_format(filename))

        logging.info("Parsed : {}".format(filename))

        if provided_by is None:
            provided_by = os.path.basename(filename)

        self.load_edges(rdfgraph, provided_by=provided_by)
        self.load_nodes(rdfgraph, provided_by=provided_by)

    def add_ontology(self, owlfile:str):
        ont = rdflib.Graph()
        ont.parse(owlfile, format=rdflib.util.guess_format(owlfile))
        self.ontologies.append(ont)
        logging.info("Parsed : {}".format(owlfile))

    def load_edges(self, rdfgraph:rdflib.Graph, provided_by:str=None):
        pass

    def load_nodes(self, rdfgraph:rdflib.Graph, provided_by:str=None):
        """
        This method loads the properties of nodes in the NetworkX graph. As there
        can be many values for a single key, all properties are lists by default.

        This method assumes that load_edges has been called, and that all nodes
        have had their IRI saved as an attribute.
        """
        with click.progressbar(self.graph.nodes(), label='loading nodes') as bar:
            for node_id in bar:
                if 'iri' in self.graph.node[node_id]:
                    iri = self.graph.node[node_id]['iri']
                else:
                    logging.warning("Expected IRI for {} provided by {}".format(node_id, provided_by))
                    continue

                node_attr = defaultdict(list)

                for s, p, o in rdfgraph.triples((URIRef(iri), None, None)):
                    if p in property_mapping or isinstance(o, rdflib.term.Literal):
                        p = property_mapping.get(p, process_iri(p))
                        o = process_iri(o)
                        node_attr[p].append(o)

                category = find_category(iri, [rdfgraph] + self.ontologies)

                if category is not None:
                    node_attr['category'].append(category)

                if provided_by is not None:
                    node_attr['provided_by'].append(provided_by)

                for k, values in node_attr.items():
                    if isinstance(values, list):
                        node_attr[k] = [make_curie(v) for v in values]

                for key, value in node_attr.items():
                    self.graph.node[node_id][key] = value

class ObanRdfTransformer(RdfTransformer):
    OBAN = Namespace('http://purl.org/oban/')

    def load_edges(self, rdfgraph:rdflib.Graph, provided_by:str=None):
        associations = list(rdfgraph.subjects(RDF.type, self.OBAN.association))
        with click.progressbar(associations, label='loading edges') as bar:
            for association in bar:
                edge_attr = defaultdict(list)

                edge_attr['iri'] = str(association)
                edge_attr['id'] = make_curie(association)

                for s, p, o in rdfgraph.triples((association, None, None)):
                    if p in property_mapping or isinstance(o, rdflib.term.Literal):
                        p = property_mapping.get(p, process_iri(p))
                        o = process_iri(o)
                        edge_attr[p].append(o)

                if 'predicate' not in edge_attr:
                    edge_attr['predicate'].append('related to')

                if provided_by is not None:
                    edge_attr['provided_by'].append(provided_by)

                subjects = edge_attr['subject']
                objects = edge_attr['object']

                for k, values in edge_attr.items():
                    if isinstance(values, list):
                        edge_attr[k] = [make_curie(v) for v in values]

                for subject_iri in subjects:
                    for object_iri in objects:
                        sid = make_curie(subject_iri)
                        oid = make_curie(object_iri)

                        self.graph.add_edge(sid, oid, **edge_attr)

                        self.graph.node[sid]['iri'] = subject_iri
                        self.graph.node[sid]['id'] = sid

                        self.graph.node[oid]['iri'] = object_iri
                        self.graph.node[oid]['id'] = oid
