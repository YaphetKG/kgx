import pandas as pd
import networkx as nx
import logging, yaml
from .transformer import Transformer

from neo4j.v1 import GraphDatabase

class NeoTransformer(Transformer):
    """
    TODO: use bolt

    We expect a Translator canonical style http://bit.ly/tr-kg-standard
    E.g. predicates are names with underscores, not IDs.

    TODO: also support mapping from Monarch neo4j
    """

    def __init__(self, t=None):
        super(NeoTransformer, self).__init__(t)
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile)

        uri = "bolt://{}:{}".format(cfg['neo4j']['host'], cfg['neo4j']['port'])
        self.driver = GraphDatabase.driver(uri, auth=(cfg['neo4j']['username'], cfg['neo4j']['password']))

    def load(self):
        """
        Read a neo4j database and create a nx graph
        """

        with self.driver.session() as session:
            self.load_nodes(session.read_transaction(self.get_nodes))
            self.load_edges(session.read_transaction(self.get_edges))

    def load_nodes(self, node_records):
        """
        Load nodes from neo4j records
        """

        for node in node_records:
            self.load_node(node)

    def load_edges(self, edge_records):
        """
        Load edges from neo4j records
        """

        for edge in edge_records:
            self.load_edge(edge)

    def load_node(self, node_record):
        """
        Load node from a neo4j record
        """

        node=node_record[0]
        attributes = {}
        for i in node.items():
            attributes[i[0]] = i[1]

        self.graph.add_node(node.get('id'), attr_dict=attributes)

    def load_edge(self, edge_record):
        """
        Load an edge from a neo4j record
        """

        s = edge_record[0]
        p = edge_record[1]
        o = edge_record[2]
        attributes = {}
        for i in p.items():
            attributes[i[0]] = i[1]

        self.graph.add_edge(s['id'], o['id'], attr_dict=attributes)

    def get_nodes(self, tx):
        """
        Get all nodes from neo4j database
        """

        return tx.run("MATCH (n) RETURN n")

    def get_edges(self, tx):
        """
        Get all edges from neo4j database
        """

        return tx.run("MATCH (s)-[p]->(o) RETURN s,p,o")

    def save_node(self, tx, obj):
        """
        Load a node into neo4j
        """

        if 'id' not in obj:
            raise KeyError("node does not have 'id' property")
        if 'name' not in obj:
            logging.warning("node does not have 'name' property")

        if 'category' not in obj:
            logging.warning("node does not have 'category' property. Using 'named_thing' as default")
            label = 'named_thing'
        else:
            label = obj['category']
            del obj['category']

        query = "CREATE (n:{label} {{ {properties} }})".format(label = label, properties = self.parse_properties(obj))
        tx.run(query)

    def save_edge(self, tx, obj):
        """
        Load an edge into neo4j
        """

        queryString = "MATCH (s {{ id: '{subject_id}' }}) MATCH (o {{ id: '{object_id}' }}) MERGE (s)-[r:{relationship} {{ {relationship_properties} }}]->(o)"
        queryStringWithLabel = "MATCH (s:{subject_label} {{ id: '{subject_id}' }}) MATCH (o:{object_label} {{ id: '{object_id}' }}) MERGE (s)-[r:{relationship} {{ {relationship_properties} }}]->(o)"

        subject_label = obj['subject_label'] if 'subject_label' in obj else 'named_thing'
        object_label = obj['object_label'] if 'object_label' in obj else 'named_thing'

        query_params = {
            'subject_label': subject_label, 'subject_id': obj['subject'],
            'object_label': object_label, 'object_id': obj['object'],
            'relationship': obj['predicate'], 'relationship_properties': self.parse_properties(obj)
        }

        if query_params['subject_label'] and query_params['object_label']:
            query = queryStringWithLabel.format(**query_params)
        else:
            query = queryString.format(**query_params)

        logging.debug(query)
        tx.run(query)

    def save_from_csv(self, nodes_filename, edges_filename):
        """
        Load from a CSV to neo4j
        """
        nodes_df = pd.read_csv(nodes_filename)
        edges_df = pd.read_csv(edges_filename)

        with self.driver.session() as session:
            for index, row in nodes_df.iterrows():
                session.write_transaction(self.save_node, row.to_dict())
            for index, row in edges_df.iterrows():
                session.write_transaction(self.save_edge, row.to_dict())
        self.neo4j_report()

    def save(self):
        """
        Load from a nx graph to neo4j
        """

        with self.driver.session() as session:
            for n in self.graph.nodes():
                node_attributes = self.graph.node[n]
                session.write_transaction(self.save_node, node_attributes)
            for n, nbrs in self.graph.adjacency_iter():
                for nbr, eattr in nbrs.items():
                    for entry, adjitem in eattr.items():
                        session.write_transaction(self.save_edge, adjitem)
        self.neo4j_report()

    def report(self):
        print("Total number of nodes: {}".format(len(self.graph.nodes())))
        print("Nodes: {}".format(self.graph.nodes()))
        print("Total number of edges: {}".format(len(self.graph.edges())))
        print("Edges: {}".format(self.graph.edges()))

    def neo4j_report(self):
        """
        Give a summary on the number of nodes and edges in neo4j database
        """
        with self.driver.session() as session:
            for r in session.run("MATCH (n) RETURN COUNT(*)"):
                logging.info("Number of Nodes: {}".format(r.values()[0]))
            for r in session.run("MATCH (s)-->(o) RETURN COUNT(*)"):
                logging.info("Number of Edges: {}".format(r.values()[0]))

    @staticmethod
    def parse_properties(properties, delim = '|'):
        propertyList = []
        for key in properties:
            if key in ['subject', 'predicate', 'object']:
                continue
            if delim in properties[key]:
                values = properties[key].split(delim)
                pair = "{}: {}".format(key, values)
            else:
                values = properties[key]
                pair = "{}: '{}'".format(key, values)
            propertyList.append(pair)
        return ','.join(propertyList)

class MonarchNeoTransformer(NeoTransformer):
    """
    TODO: do we need a subclass, or just make parent configurable?

    In contrast to a generic import/export, the Monarch neo4j graph
    uses reification (same as Richard's semmeddb implementation in neo4j).
    This transform should de-reify.

    Also:

     - rdf:label to name
     - neo4j label to category
    """
    