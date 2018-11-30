from typing import List

import rdflib
from rdflib import Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL

from prefixcommons.curie_util import contract_uri, expand_uri, default_curie_maps

OBAN = Namespace('http://purl.org/oban/')

def reverse_mapping(d:dict):
    """
    Returns a dictionary where the keys are the values of the given dictionary,
    and the values are sets of keys from the given dictionary.
    """
    return {value : set(k for k, v in d.items() if v == value) for value in d.items()}

category_mapping = {
# Added myself!
    "http://purl.obolibrary.org/obo/SO_0001217" : "protein coding gene",
    "http://purl.obolibrary.org/obo/GENO_0000002" : "variant allele",
    # sequence feature for clinvar?
    "http://purl.obolibrary.org/obo/SO_0000110" : "sequence feature",
# Taken from the yaml
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

reverse_category_mapping = reverse_mapping(category_mapping)

property_mapping = {
    OBAN.association_has_subject : 'subject',
    OBAN.association_has_object : 'object',
    OBAN.association_has_predicate : 'predicate',
    RDF.type : 'type',
    RDFS.comment : 'comment',
    RDFS.label : 'name',
    URIRef('http://purl.org/dc/elements/1.1/description') : 'description',
    URIRef('http://purl.obolibrary.org/obo/RO_0002558') : 'has_evidence',
    URIRef('http://www.geneontology.org/formats/oboInOwl#hasExactSynonym') : 'synonyms',
    URIRef('http://www.w3.org/2004/02/skos/core#exactMatch') : 'exact_match',
    URIRef('http://www.geneontology.org/formats/oboInOwl#hasDbXref') : 'exact_match',
    OWL.equivalentClass : 'exact_match'
}

reverse_property_mapping = reverse_mapping(property_mapping)

equals_predicates = [
    OWL.equivalentClass,
    URIRef('http://www.geneontology.org/formats/oboInOwl#hasDbXref'),
    URIRef('http://www.w3.org/2004/02/skos/core#exactMatch'),
]

isa_predicates = [RDFS.subClassOf, RDF.type]

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

def find_category(rdfgraph:rdflib.Graph, iri:URIRef) -> str:
    """
    Finds a category for the given iri, by walking up edges with isa predicates
    and across edges with identity predicates.

    Tries to get a category in category_mapping. If none are found then takes
    the highest superclass it can find.
    """
    if not isinstance(iri, URIRef):
        iri = URIRef(iri)

    def super_class_generator(iri:URIRef) -> URIRef:
        """
        Generates nodes and scores for walking a path from the given iri to its
        superclasses. Equivalence edges are weighted zero, since they don't count
        as moving further up the ontological hierarchy.

        Note: Not every node generated is gaurenteed to be a superclass
        """
        for predicate in equals_predicates:
            if not isinstance(predicate, URIRef):
                predicate = URIRef(predicate)

            for equivalent_iri in rdfgraph.subjects(predicate=predicate, object=iri):
                yield equivalent_iri, 0
            for equivalent_iri in rdfgraph.objects(subject=iri, predicate=predicate):
                yield equivalent_iri, 0

        for predicate in isa_predicates:
            if not isinstance(predicate, URIRef):
                predicate = URIRef(predicate)

            for superclass_iri in rdfgraph.objects(subject=iri, predicate=predicate):
                yield superclass_iri, 1

    best_iri, best_score = None, 0
    for uri_ref, score in walk(rdfgraph, iri, super_class_generator):
        if str(uri_ref) in category_mapping and score > 0:
            return category_mapping[str(uri_ref)]
        elif score > best_score:
            best_iri, best_score = str(uriref), score

    return best_iri

def make_curie(uri:URIRef) -> str:
    """
    We sort the curies to ensure that we take the same item every time
    """
    curies = contract_uri(str(uri))
    curies.sort()
    if len(curies) > 0:
        return curies[0]
    return str(uri)
