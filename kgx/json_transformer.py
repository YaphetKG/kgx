import json
import logging
from .transformer import Transformer
from .pandas_transformer import PandasTransformer  # Temp

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (tuple, set)):
            return list(obj)
        else:
            return json.JSONEncoder.default(self, obj)

class JsonTransformer(PandasTransformer):
    """
    """

    def parse(self, filename, input_format='json', **args):
        """
        Parse a JSON file
        """
        with open(filename, 'r') as f:
            obj = json.load(f)
            if 'nodes' in obj:
                self.load_nodes(obj['nodes'])
            if 'edges' in obj:
                self.load_edges(obj['edges'])

    def load_nodes(self, nodes):
        for d in nodes:
            n = d['id']
            del d['id']
            self.graph.add_node(n, **d)

    def load_edges(self, edges):
        for d in edges:
            s = d['subject']
            o = d['object']
            del d['subject']
            del d['object']
            self.graph.add_edge(s, o, **d)

    def export(self):
        nodes=[]
        edges=[]
        for id, data in self.graph.nodes(data=True):
            node = data.copy()
            node['id'] = id
            nodes.append(node)
        for o, s, data in self.graph.edges(data=True):
            edge = data.copy()
            edge['subject'] = s
            edge['object'] = o
            edges.append(edge)

        return {'nodes':nodes, 'edges':edges}

    def save(self, filename, **args):
        """
        Write a JSON file
        """
        obj = self.export()
        with open(filename,'w') as file:
            file.write(json.dumps(obj, indent=4, sort_keys=True, cls=JSONEncoder))
