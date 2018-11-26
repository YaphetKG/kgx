from .transformer import Transformer

import networkx as nx
import rdflib
import logging
import uuid
import click
from rdflib import Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL
from typing import NewType
from collections import defaultdict, Counter

from prefixcommons.curie_util import contract_uri, expand_uri, default_curie_maps

UriString = NewType("UriString", str)

OBAN = Namespace('http://purl.org/oban/')

# TODO: use JSON-LD context
mapping = {
    'subject': OBAN.association_has_subject,
    'object': OBAN.association_has_object,
    'predicate': OBAN.association_has_predicate,
    'type' : RDF.type,
    'comment': RDFS.comment,
    'name': RDFS.label,
    'description' : URIRef('http://purl.org/dc/elements/1.1/description'),
    'has_evidence' : URIRef('http://purl.obolibrary.org/obo/RO_0002558'),
    # Is exact_match same as xrefs?
    'exact_match' : URIRef('http://www.w3.org/2004/02/skos/core#exactMatch'),
    'xrefs' : URIRef('http://www.geneontology.org/formats/oboInOwl#hasDbXref'),
    'category' : URIRef('http://www.w3.org/2000/01/rdf-schema#subClassOf'),
    'synonyms' : URIRef('http://www.geneontology.org/formats/oboInOwl#hasExactSynonym'),
}
reverse_mapping = {y: x for x, y in mapping.items()}

iri_to_categories_map = {
    # These are from the original yaml file
    "http://purl.obolibrary.org/obo/CL_0000000" : "cell",
    "http://purl.obolibrary.org/obo/UBERON_0001062" : "anatomical entity",
    "http://purl.obolibrary.org/obo/ZFA_0009000" : "cell",
    "http://purl.obolibrary.org/obo/UBERON_0004529" : "anatomical projection",
    "http://purl.obolibrary.org/obo/UBERON_0000468" : "multi-cellular organism",
    "http://purl.obolibrary.org/obo/UBERON_0000955" : "brain",
    "http://purl.obolibrary.org/obo/PATO_0000001" : "quality",
    "http://purl.obolibrary.org/obo/GO_0005623" : "cell",
    "http://purl.obolibrary.org/obo/WBbt_0007833" : "organism",
    "http://purl.obolibrary.org/obo/WBbt_0004017" : "cell",
    "http://purl.obolibrary.org/obo/MONDO_0000001" : "disease",
    "http://purl.obolibrary.org/obo/PATO_0000003" : "assay",
    "http://purl.obolibrary.org/obo/PATO_0000006" : "process",
    "http://purl.obolibrary.org/obo/PATO_0000011" : "age",
    "http://purl.obolibrary.org/obo/ZFA_0000008" : "brain",
    "http://purl.obolibrary.org/obo/ZFA_0001637" : "bony projection",
    "http://purl.obolibrary.org/obo/WBPhenotype_0000061" : "extended life span",
    "http://purl.obolibrary.org/obo/WBPhenotype_0000039" : "life span variant",
    "http://purl.obolibrary.org/obo/WBPhenotype_0001171" : "shortened life span",
    "http://purl.obolibrary.org/obo/CHEBI_23367" : "molecular entity",
    "http://purl.obolibrary.org/obo/CHEBI_23888" : "drug",
    "http://purl.obolibrary.org/obo/CHEBI_51086" : "chemical role",
    "http://purl.obolibrary.org/obo/UPHENO_0001001" : "Phenotype",
    "http://purl.obolibrary.org/obo/GO_0008150" : "biological_process",
    "http://purl.obolibrary.org/obo/GO_0005575" : "cellular component",
    "http://purl.obolibrary.org/obo/SO_0000704" : "gene",
    "http://purl.obolibrary.org/obo/SO_0000110" : "sequence feature",
    "http://purl.obolibrary.org/obo/GENO_0000536" : "genotype",
}

def walk(rdfgraph, node_iri, next_node_generator):
    """
    next_node_generator is a function that takes an iri and returns a generator for iris.
    next_node_generator might return Tuple[iri, int], in which case int is taken to be
    the score of the edge. If no score is returned, then the score will be
    taken to be zero.
    """
    if not isinstance(node_iri, URIRef):
        node_iri = URIRef(node_iri)
    to_visit = {node_iri : 0}
    visited = {}
    while to_visit != {}:
        iri, score = to_visit.popitem()
        visited[iri] = score
        for t in next_node_generator(iri):
            if isinstance(t, tuple) and len(t) > 1:
                n, s = t
            else:
                n, s = t, 0
            if n not in visited:
                to_visit[n] = score + s
                yield n, to_visit[n]

def find_category(rdfgraph, iri):
    if not isinstance(iri, URIRef):
        iri = URIRef(iri)

    def super_class_generator(iri:URIRef) -> URIRef:
        """
        Generates nodes and scores for walking a path from the given iri to its
        superclasses. equivalence edges are weighted zero, since they don't count
        as moving further up the ontological hierarchy.

        Note: Not every node generated is gaurenteed to be a superclass
        """
        for equivalent_iri in rdfgraph.subjects(predicate=OWL['equivalentClass'], object=iri):
            yield equivalent_iri, 0
        for equivalent_iri in rdfgraph.objects(subject=iri, predicate=OWL['equivalentClass']):
            yield equivalent_iri, 0
        for superclass_iri in rdfgraph.objects(subject=iri, predicate=RDFS.subClassOf):
            yield superclass_iri, 1

    for node, score in walk(rdfgraph, iri, super_class_generator):
        if str(node) in iri_to_categories_map:
            return iri_to_categories_map[str(node)]

    return None

class RdfTransformer(Transformer):
    """
    Transforms to and from RDF

    We support different RDF metamodels, including:

     - OBAN reification (as used in Monarch)
     - RDF reification

    TODO: we will have some of the same logic if we go from a triplestore. How to share this?
    """

    def parse(self, filename:str=None, input_format:str=None, provided_by:str=None):
        """
        Parse a file into an graph, using rdflib
        """
        rdfgraph = rdflib.Graph()

        guessed_format = rdflib.util.guess_format(filename)
        if guessed_format is not None:
            input_format = guessed_format

        if input_format is None:
            if filename.endswith(".ttl"):
                input_format = 'turtle'
            elif filename.endswith(".rdf"):
                input_format = 'xml'
            elif filename.endswith(".owl"):
                input_format = 'xml'
            else:
                raise Exception('Unrecognized RDF format {}'.format(input_format))

        rdfgraph.parse(filename, format=input_format)
        logging.info("Parsed : {}".format(filename))

        # TODO: use source from RDF
        if provided_by is not None and isinstance(filename, str):
            provided_by = filename
        self.graph_metadata['provided_by'] = provided_by
        logging.info('Loading edges')
        self.load_edges(rdfgraph)
        logging.info('Loading nodes')
        self.load_nodes(rdfgraph)

    def curie(self, uri: UriString) -> str:
        """
        Translate a URI into a CURIE (prefixed identifier)
        """
        pm = self.prefix_manager
        return pm.contract(str(uri))
        #curies = contract_uri(str(uri))
        #if len(curies)>0:
        #    return curies[0]
        #return str(uri)

    def load_nodes(self, rdfgraph: rdflib.Graph):
        G = self.graph
        with click.progressbar(G.nodes(), label='loading nodes') as bar:
            for nid in bar:
                n = G.node[nid]
                if 'iri' not in n:
                    logging.warning("Expected IRI for {}".format(n))
                    continue
                iri = URIRef(n['iri'])
                npmap = defaultdict(list)
                for s,p,o in rdfgraph.triples((iri, None, None)):
                    if p in reverse_mapping:
                        p = reverse_mapping[p]
                        npmap[p].append(str(o))

                    # if isinstance(o, rdflib.term.Literal):
                    #     if p in reverse_mapping:
                    #         p = reverse_mapping[p]
                    #     npmap[p] = str(o)
                    # if p == rdflib.RDFS.subClassOf:
                    #     if 'category' not in npmap:
                    #         npmap['category'] = []
                    #
                    #     category_curie = self.curie(str(o))
                    #
                    #     if category_curie in category_map:
                    #         npmap['category'] += category_map[category_curie]
                    #     else:
                    #         npmap['category'] += [category_curie]

                G.add_node(nid, **npmap)

    def load_edges(self, rg: rdflib.Graph):
        pass

    def add_edge(self, o:UriString, s:UriString, attr_dict={}):
        sid = self.curie(s)
        oid = self.curie(o)
        if 'category' not in self.graph.node[sid]:
            return
        if 'category' not in self.graph.node[oid]:
            return
        self.graph.add_node(sid, iri=str(s))
        self.graph.add_node(oid, iri=str(o))
        self.graph.add_edge(oid, sid, **attr_dict)

class ObanRdfTransformer(RdfTransformer):
    """
    Transforms to and from RDF, assuming OBAN-style modeling
    """
    rprop_set = set(('subject', 'predicate', 'object', 'provided_by', 'id', str(RDF.type)))
    inv_cmap = {}
    cmap = {}

    def __init__(self, graph=None):
        super().__init__(graph)
        # Generate the map and the inverse map from default curie maps, which will be used later.
        for cmap in default_curie_maps:
            for k, v in cmap.items():
                self.inv_cmap[v] = k
                self.cmap[k] = v

    def load_node(self, rdfgraph, iri, networkx_id):
        if not self.graph.has_node(networkx_id):
            node_attr = defaultdict(set)

            for s, p, o in rdfgraph.triples((iri, None, None)):
                if p in reverse_mapping:
                    p = reverse_mapping[p]
                    node_attr[p].add(str(o))
                elif isinstance(o, rdflib.term.Literal):
                    node_attr[p].add(str(o))

            for key, value in node_attr.items():
                node_attr[key] = list(value)

            node_attr['iri'] = iri
            node_attr['id'] = networkx_id

            c = find_category(rdfgraph, iri)
            if c is not None:
                node_attr['category'] = [c]

            self.graph.add_node(networkx_id, **node_attr)

    def load_edges(self, rdfgraph: rdflib.Graph):
        with click.progressbar(rdfgraph.subjects(RDF.type, OBAN.association), label='loading edges') as bar:
            for association in bar:
                edge_attr = defaultdict(list)
                # Keep the id of this entity (e.g., <https://monarchinitiative.org/MONARCH_08830...>) as the value of 'id'.
                # edge_attr['id'] = pm.contract(str(association))
                edge_attr['iri'] = str(association)
                edge_attr['id'] = self.curie(association)
                edge_attr['provided_by'] = self.graph_metadata['provided_by']

                for s, p, o in rdfgraph.triples((association, None, None)):
                    if p in reverse_mapping:
                        p = reverse_mapping[p]
                        edge_attr[p].append(str(o))
                    elif isinstance(o, rdflib.term.Literal):
                        edge_attr[p].append(str(o))

                subjects = edge_attr['subject']
                objects = edge_attr['object']

                id_map = {}
                for iri in set(subjects + objects):
                    node_id = id_map[iri] if iri in id_map else self.curie(iri)
                    id_map[iri] = str(node_id)
                    if not self.graph.has_node(node_id):
                        self.load_node(rdfgraph, iri, node_id)

                for subject_iri in subjects:
                    for object_iri in objects:
                        sid = id_map[subject_iri]
                        oid = id_map[object_iri]

                        self.graph.add_edge(sid, oid, **edge_attr)

    def load_nodes(self, rdfgraph: rdflib.Graph):
        pass

    def curie(self, uri: UriString) -> str:
        """
        We sort the curies to ensure that we take the same item every time
        """
        curies = contract_uri(str(uri))
        curies.sort()
        if len(curies) > 0:
            return curies[0]
        return str(uri)

    # move to superclass?
    def uriref(self, id) -> URIRef:
        if id in mapping:
            uri = mapping[id]
        else:
            uri = self.prefix_manager.expand(id)
        return URIRef(uri)

    def save(self, filename: str = None, output_format: str = None, **kwargs):
        """
        Transform the internal graph into the RDF graphs that follow OBAN-style modeling and dump into the file.
        """
        # Make a new rdflib.Graph() instance to generate RDF triples
        rdfgraph = rdflib.Graph()
        # Register OBAN's url prefix (http://purl.org/oban/) as `OBAN` in the namespace.
        rdfgraph.bind('OBAN', str(OBAN))

        # <http://purl.obolibrary.org/obo/RO_0002558> is currently stored as OBO:RO_0002558 rather than RO:0002558
        # because of the bug in rdflib. See https://github.com/RDFLib/rdflib/issues/632
        rdfgraph.bind('OBO', 'http://purl.obolibrary.org/obo/')

        # Using an iterator of (node, adjacency dict) tuples for all nodes,
        # we iterate every edge (only outgoing adjacencies)
        for n, nbrs in self.graph.adjacency_iter():
            a_object = n
            for nbr, eattr in nbrs.items():
                a_subject = nbr
                for entry, adjitem in eattr.items():
                    pred = "relatedTo"
                    if 'predicate' in adjitem:
                        pred = adjitem['predicate'][0]
                    # assoc_id here is used as subject for each entry, e.g.,
                    # <https://monarchinitiative.org/MONARCH_08830...>
                    if 'id' in adjitem and adjitem['id'] is not None:
                        assoc_id = URIRef(adjitem['id'])
                    else:
                        assoc_id = URIRef('urn:uuid:{}'.format(uuid.uuid4()))
                    self.unpack_adjitem(rdfgraph, assoc_id, adjitem)

                    # The remaining ones are then OBAN's properties and corresponding objects. Store them as triples.
                    rdfgraph.add((assoc_id, mapping['subject'], self.uriref(a_subject)))
                    rdfgraph.add((assoc_id, mapping['predicate'], self.uriref(pred)))
                    rdfgraph.add((assoc_id, mapping['object'], self.uriref(a_object)))

        # For now, assume that the default format is turtle if it is not specified.
        if output_format is None:
            output_format = "turtle"

        # Serialize the graph into the file.
        rdfgraph.serialize(destination=filename, format=output_format)

    def unpack_adjitem(self, rdfgraph, assoc_id, adjitem):
        # Iterate adjacency dict, which contains pairs of properties and objects sharing the same subject.
        for prop_id, prop_values in adjitem.items():
            # See whether the current pair's prop/obj is the OBAN's one.
            if prop_id in self.rprop_set:
                continue

            # If not, see whether its props and objs can be registered as curies in namespaces.
            # Once they are registered, URI/IRIs can be shorten using the curies registered in namespaces.
            # e.g., register http://purl.obolibrary.org/obo/ECO_ with ECO.
            if not isinstance(prop_values,list):
                prop_values = [prop_values]
            for prop_value in prop_values:
                obj_uri = self.uriref(prop_value)

                # Store the pair as a triple.
                rdfgraph.add((assoc_id, self.uriref(prop_id), obj_uri))

    def split_uri(self, prop_uri):
        """
        Utility function that splits into URI/IRI into prefix and value, e.g.,
        http://purl.obolibrary.org/obo/RO_0002558 as http://purl.obolibrary.org/obo/RO_ and 0002558
        """
        prop_splits = prop_uri.split('_')
        if len(prop_splits) > 1:
            return prop_splits[0] + "_", prop_splits[1]
        else:
            prop_splits = prop_uri.split('#')
            if len(prop_splits) > 1:
                return prop_splits[0] + "#", prop_splits[1]
            else:
                slash_index = prop_uri.rfind("/")
                return prop_uri[0:slash_index + 1], prop_uri[slash_index + 1:]
            self.add_edge(o, s, attr_dict=obj)

class RdfOwlTransformer(RdfTransformer):
    """
    Transforms from an OWL ontology in RDF, retaining class-class
    relationships
    """

    def load_edges(self, rg: rdflib.Graph):
        """
        """
        for s,p,o in rg.triples( (None,RDFS.subClassOf,None) ):
            if isinstance(s, rdflib.term.BNode):
                continue
            pred = None
            parent = None
            obj = {}
            if isinstance(o, rdflib.term.BNode):
                # C SubClassOf R some D
                prop = None
                parent = None
                for x in rg.objects( o, OWL.onProperty ):
                    pred = x
                for x in rg.objects( o, OWL.someValuesFrom ):
                    parent = x
                if pred is None or parent is None:
                    logging.warning("Do not know how to handle: {}".format(o))
            else:
                # C SubClassOf D (C and D are named classes)
                pred = 'owl:subClassOf'
                parent = o
            obj['predicate'] = pred
            obj['provided_by'] = self.graph_metadata['provided_by']
            self.add_edge(parent, s, attr_dict=obj)
