import kgx
import os, sys, click, logging, itertools, pickle, json, yaml
import pandas as pd
from typing import List
from urllib.parse import urlparse
from kgx import Transformer, Validator, map_graph, Filter, FilterLocation

from kgx.cli.decorators import handle_exception
from kgx.cli.utils import get_file_types, get_type, get_transformer_constructor, is_writable

from kgx.cli.utils import Config

from neo4j.v1 import GraphDatabase
from neo4j.v1.types import Node, Record

import pandas as pd

from collections import defaultdict

pass_config = click.make_pass_decorator(Config, ensure=True)

def error(msg):
    click.echo(msg)
    quit()

@click.group()
@click.option('--debug', is_flag=True, help='Prints the stack trace if error occurs')
@click.version_option(version=kgx.__version__, prog_name=kgx.__name__)
@pass_config
def cli(config, debug):
    """
    Knowledge Graph Exchange
    """
    config.debug = debug
    if debug:
        logging.basicConfig(level=logging.DEBUG)

from collections import Counter
from terminaltables import AsciiTable

def get_prefix(curie:str) -> str:
    if ':' in curie:
        p, _ = curie.rsplit(':', 1)
        return p
    else:
        return None

@cli.command('node-summary')
@click.argument('input', type=click.Path(exists=True), required=True)
@click.option('--input-type', type=click.Choice(get_file_types()))
def node_summary(input, input_type):
    """
    Loads and summarizes a knowledge graph node set
    """
    t = load_transformer([input], input_type)

    g = t.graph

    tuples = []
    with click.progressbar(g.nodes(), label='Reading knowledge graph') as bar:
        for n in bar:
            categories = g.node[n].get('category')
            curie = g.node[n].get('id')
            prefix = None

            if ':' in curie:
                prefix = get_prefix(curie)

            if not isinstance(categories, (list, tuple, set)):
                categories = [categories]

            for category in categories:
                tuples.append((prefix, category))

    tuple_count = Counter(tuples)

    headers = [['Prefix', 'Category', 'Frequency']]
    rows = [[*k, v] for k, v in tuple_count.items()]
    print(AsciiTable(headers + rows).table)

    category_count = defaultdict(lambda: 0)
    prefix_count = defaultdict(lambda: 0)

    for (prefix, category), frequency in tuple_count.items():
        category_count[category] += frequency
        prefix_count[prefix] += frequency

    headers = [['Category', 'Frequency']]
    rows = [[k, v] for k, v in category_count.items()]
    print(AsciiTable(headers + rows).table)

    headers = [['Prefixes', 'Frequency']]
    rows = [[k, v] for k, v in prefix_count.items()]
    print(AsciiTable(headers + rows).table)

@cli.command('edge-summary')
@click.argument('input', type=click.Path(exists=True), required=True)
@click.option('--input-type', type=click.Choice(get_file_types()))
def edge_summary(input, input_type):
    """
    Loads and summarizes a knowledge graph edge set
    """
    t = load_transformer([input], input_type)

    g = t.graph

    tuples = []
    with click.progressbar(g.edges(data=True), label='Reading knowledge graph') as bar:
        for s, o, edge_attr in bar:
            subject_attr = g.node[s]
            object_attr = g.node[o]

            subject_prefix = get_prefix(s)
            object_prefix = get_prefix(o)

            subject_categories = subject_attr.get('category')
            object_categories = object_attr.get('category')
            predicates = edge_attr.get('predicate')

            if not isinstance(subject_categories, (list, set, tuple)):
                subject_categories = [subject_categories]

            if not isinstance(object_categories, (list, set, tuple)):
                object_categories = [object_categories]

            if not isinstance(predicates, (list, set, tuple)):
                predicates = [predicates]

            for subject_category in subject_categories:
                for object_category in object_categories:
                    for predicate in predicates:
                        tuples.append((subject_prefix, subject_category, predicate, object_prefix, object_category))

    tuple_count = Counter(tuples)

    headers = [['Subject Prefix', 'Subject Category', 'Predicate', 'Object Prefix', 'Object Category', 'Frequency']]
    rows = [[*k, v] for k, v in tuple_count.items()]
    print(AsciiTable(headers + rows).table)

@cli.command(name='neo4j-node-summary')
@click.option('-a', '--address', type=str, required=True)
@click.option('-u', '--username', type=str, required=True)
@click.option('-p', '--password', type=str, required=True)
@click.option('-o', '--output', type=click.Path(exists=False))
@pass_config
def neo4j_node_summary(config, address, username, password, output=None):
    if output is not None and not is_writable(output):
        error(f'Cannot write to {output}')

    bolt_driver = GraphDatabase.driver(address, auth=(username, password))

    query = """
    MATCH (x) RETURN DISTINCT x.category AS category
    """

    with bolt_driver.session() as session:
        records = session.run(query)

    categories = set()

    for record in records:
        category = record['category']
        if isinstance(category, str):
            categories.add(category)
        elif isinstance(category, (list, set, tuple)):
            categories.update(category)
        elif category is None:
            continue
        else:
            error('Unrecognized value for node.category: {}'.format(category))

    rows = []
    with click.progressbar(categories, length=len(categories)) as bar:
        for category in bar:
            query = """
            MATCH (x) WHERE x.category = {category} OR {category} IN x.category
            RETURN DISTINCT
                {category} AS category,
                split(x.id, ':')[0] AS prefix,
                COUNT(*) AS frequency
            ORDER BY category, frequency DESC;
            """

            with bolt_driver.session() as session:
                records = session.run(query, category=category)

            for record in records:
                rows.append({
                    'category' : record['category'],
                    'prefix' : record['prefix'],
                    'frequency' : record['frequency']
                })

    df = pd.DataFrame(rows)
    df = df[['category', 'prefix', 'frequency']]

    if output is None:
        click.echo(df)
    else:
        df.to_csv(output, sep='|', header=True)
        click.echo('Saved report to {}'.format(output))

@cli.command(name='neo4j-edge-summary')
@click.option('-a', '--address', type=str, required=True)
@click.option('-u', '--username', type=str, required=True)
@click.option('-p', '--password', type=str, required=True)
@click.option('-o', '--output', type=click.Path(exists=False))
@pass_config
def neo4j_edge_summary(config, address, username, password, output=None):
    if output is not None and not is_writable(output):
        error(f'Cannot write to {output}')

    bolt_driver = GraphDatabase.driver(address, auth=(username, password))

    query = """
    MATCH (x) RETURN DISTINCT x.category AS category
    """

    with bolt_driver.session() as session:
        records = session.run(query)

    categories = set()

    for record in records:
        category = record['category']
        if isinstance(category, str):
            categories.add(category)
        elif isinstance(category, (list, set, tuple)):
            categories.update(category)
        elif category is None:
            continue
        else:
            error('Unrecognized value for node.category: {}'.format(category))

    categories = list(categories)

    query = """
    MATCH (n)-[r]-(m)
    WHERE
        (n.category = {category1} OR {category1} IN n.category) AND
        (m.category = {category2} OR {category2} IN m.category)
    RETURN DISTINCT
        {category1} AS subject_category,
        {category2} AS object_category,
        type(r) AS edge_type,
        split(n.id, ':')[0] AS subject_prefix,
        split(m.id, ':')[0] AS object_prefix,
        COUNT(*) AS frequency
    ORDER BY subject_category, object_category, frequency DESC;
    """

    combinations = [(c1, c2) for c1 in categories for c2 in categories]

    rows = []
    with click.progressbar(combinations, length=len(combinations)) as bar:
        for category1, category2 in bar:
            with bolt_driver.session() as session:
                records = session.run(query, category1=category1, category2=category2)

                for r in records:
                    rows.append({
                        'subject_category' : r['subject_category'],
                        'object_category' : r['object_category'],
                        'subject_prefix' : r['subject_prefix'],
                        'object_prefix' : r['object_prefix'],
                        'frequency' : r['frequency']
                    })

    df = pd.DataFrame(rows)
    df = df[['subject_category', 'subject_prefix', 'object_category', 'object_prefix', 'frequency']]

    if output is None:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            click.echo(df)
    else:
        df.to_csv(output, sep='|', header=True)
        click.echo('Saved report to {}'.format(output))

@cli.command()
@click.option('--input-type', type=click.Choice(get_file_types()))
@click.argument('inputs', nargs=-1, type=click.Path(exists=False), required=True)
@pass_config
def validate(config, inputs, input_type):
    v = Validator()
    t = load_transformer(inputs, input_type)
    result = v.validate(t.graph)
    click.echo(result)

@cli.command(name='neo4j-download')
@click.option('--output-type', type=click.Choice(get_file_types()))
@click.option('-d', '--directed', is_flag=True, help='Enforces subject -> object edge direction')
@click.option('-lb', '--labels', type=(click.Choice(FilterLocation.values()), str), multiple=True, help='For filtering on labels. CHOICE: {}'.format(', '.join(FilterLocation.values())))
@click.option('-pr', '--properties', type=(click.Choice(FilterLocation.values()), str, str), multiple=True, help='For filtering on properties (key value pairs). CHOICE: {}'.format(', '.join(FilterLocation.values())))
@click.option('-a', '--address', type=str, required=True)
@click.option('-u', '--username', type=str)
@click.option('-p', '--password', type=str)
@click.option('--host', type=str)
@click.option('--port', type=str)
@click.option('--scheme', type=str)
@click.option('--start', type=int, default=0)
@click.option('--end', type=int)
@click.option('-o', '--output', type=click.Path(exists=False), required=True)
@pass_config
def neo4j_download(config, address, username, password, host, port, scheme, output, output_type, labels, properties, directed, start, end):
    if not is_writable(output):
        error(f'Cannot write to {output}')

    t = make_neo4j_transformer(address, host, port, scheme, username, password)

    set_transformer_filters(transformer=t, labels=labels, properties=properties)

    t.load(is_directed=directed, start=start, end=end)
    t.report()
    transform_and_save(t, output, output_type)

def set_transformer_filters(transformer:Transformer, labels:list, properties:list) -> None:
    for location, label in labels:
        if location == FilterLocation.EDGE.value:
            target = '{}_label'.format(location)
            transformer.set_filter(target=target, value=label)
        else:
            target = '{}_category'.format(location)
            transformer.set_filter(target=target, value=label)

    for location, property_name, property_value in properties:
        target = '{}_property'.format(location)
        transformer.set_filter(target=target, value=(property_name, property_value))

def make_neo4j_transformer(address, host, port, scheme, username, password):
    o = urlparse(address)

    if o.password is None and password is None:
        error('Could not extract the password from the address, please set password argument')
    elif password is None:
        password = o.password

    if o.username is None and username is None:
        error('Could not extract the username from the address, please set username argument')
    elif username is None:
        username = o.username

    if o.port is None and port is None:
        error('Could not extract port from the address, please set port argument')
    elif port is None:
        port = o.port

    if o.scheme is None and host is None:
        error('Could not extract host from the address, please set host argument')
    elif scheme is None:
        scheme = o.scheme

    if o.hostname is None and host is None:
        error('Could not extract host from the address, please set host argument')
    elif host is None:
        host = o.hostname + o.path

    return kgx.NeoTransformer(
        host=host,
        ports={scheme : port},
        username=username,
        password=password
    )

@cli.command(name='neo4j-upload')
@click.option('--input-type', type=click.Choice(get_file_types()))
@click.option('--use-unwind', is_flag=True, help='Loads using UNWIND, which is quicker')
@click.option('-a', '--address', type=str)
@click.option('-u', '--username', type=str)
@click.option('-p', '--password', type=str)
@click.option('--host', type=str)
@click.option('--port', type=str)
@click.option('--scheme', type=str)
@click.option('--edge-property', multiple=True, type=click.Tuple([str, str]), help='A property name and value that all edges will have')
@click.option('--node-property', multiple=True, type=click.Tuple([str, str]), help='A property name and value that all nodes will have')
@click.argument('inputs', nargs=-1, type=click.Path(exists=False), required=True)
@pass_config
def neo4j_upload(config, address, host, port, scheme, username, password, inputs, input_type, use_unwind, node_property, edge_property):
    t = load_transformer(inputs, input_type)

    neo_transformer = make_neo4j_transformer(address, host, port, scheme, username, password)
    neo_transformer.graph = t.graph

    if node_property is not None:
        for name, value in node_property:
            for n in neo_transformer.graph.nodes():
                neo_transformer.graph.node[n][name] = value

    if edge_property is not None:
        for name, value in edge_property:
            for subject_node, object_node, edge_attr_dict in neo_transformer.graph.edges.data():
                edge_attr_dict[name] = value

    if use_unwind:
        neo_transformer.save_with_unwind()
    else:
        neo_transformer.save()

@cli.command()
@click.option('--input-type', type=click.Choice(get_file_types()))
@click.option('--output-type', type=click.Choice(get_file_types()))
@click.option('--mapping', type=str)
@click.option('--preserve', is_flag=True)
@click.argument('inputs', nargs=-1, type=click.Path(exists=False), required=True)
@click.option('-o', '--output', type=click.Path(exists=False))
@pass_config
def dump(config, inputs, output, input_type, output_type, mapping, preserve):
    """\b
    Transforms a knowledge graph from one representation to another
    """
    if not is_writable(output):
        error(f'Cannot write to {output}')

    t = load_transformer(inputs, input_type)
    if mapping != None:
        path = get_file_path(mapping)
        with click.open_file(path, 'rb') as f:
            d = pickle.load(f)
            click.echo('Performing mapping: ' + mapping)
            map_graph(G=t.graph, mapping=d, preserve=preserve)
    transform_and_save(t, output, output_type)

@cli.command(name='load-mapping')
@click.argument('name', type=str)
@click.argument('csv', type=click.Path())
@click.option('--no-header', is_flag=True, help='Indicates that the given CSV file has no header, so that the first row will not be ignored')
@click.option('--columns', type=(int, int), default=(None, None), required=False, help="The zero indexed input and output columns for the mapping")
@click.option('--show', is_flag=True, help='Shows a small slice of the mapping')
@pass_config
def load_mapping(config, name, csv, columns, no_header, show):
    header = None if no_header else 0
    data = pd.read_csv(csv, header=header)
    source, target = (0, 1) if columns == (None, None) else columns
    d = {row[source] : row[target] for index, row in data.iterrows()}

    if show:
        for key, value in itertools.islice(d.items(), 5):
            click.echo(str(key) + ' : ' + str(value))

    path = get_file_path(name)

    with open(path, 'wb') as f:
        pickle.dump(d, f)
        click.echo('Mapping \'{name}\' saved at {path}'.format(name=name, path=path))

@cli.command(name='load-and-merge')
@click.argument('merge_config', type=str)
@click.option('--destination-uri', type=str)
@click.option('--destination-username', type=str)
@click.option('--destination-password', type=str)
def load_and_merge(merge_config, destination_uri, destination_username, destination_password):
    """
    Load nodes and edges from KGs, as defined in a config YAML, and merge them into a single graph
    """

    with open(merge_config, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    transformers = []
    for key in cfg['target']:
        logging.info("Connecting to {}".format(cfg['target'][key]))
        uri = "{}:{}".format(cfg['target'][key]['neo4j']['host'], cfg['target'][key]['neo4j']['port'])
        n = kgx.NeoTransformer(None, uri, cfg['target'][key]['neo4j']['username'],
                               cfg['target'][key]['neo4j']['password'])
        transformers.append(n)

        if 'target_filter' in cfg['target'][key]:
            for target_filter in cfg['target'][key]['target_filter']:
                # Set filters
                n.set_filter(target_filter, cfg['target'][key]['target_filter'][target_filter])

        start = 0
        end = None
        if 'query_limits' in cfg['target'][key]:
            if 'start' in cfg['target'][key]['query_limits']:
                start = cfg['target'][key]['query_limits']['start']
            if 'end' in cfg['target'][key]['query_limits']:
                end = cfg['target'][key]['query_limits']['end']

        n.load(start=start, end=end)

    mergedTransformer = Transformer()
    mergedTransformer.merge([x.graph for x in transformers])

    if destination_uri and destination_username and destination_password:
        destination = kgx.NeoTransformer(mergedTransformer.graph, uri=destination_uri, username=destination_username, password=destination_password)
        destination.save_with_unwind()

def get_file_path(name:str) -> str:
    app_dir = click.get_app_dir(__name__)

    if not os.path.exists(app_dir):
        os.makedirs(app_dir)

    return os.path.join(app_dir, name + '.pkl')

def transform_and_save(t:Transformer, output_path:str, output_type:str=None):
    """
    Creates a transformer with the appropraite file type from the given
    transformer, and applies that new transformation and saves to file.
    """
    if output_type is None:
        output_type = get_type(output_path)

    output_transformer = get_transformer_constructor(output_type)

    if output_transformer is None:
        error('Output does not have a recognized type: ' + str(get_file_types()))

    kwargs = {
        'extention' : output_type
    }

    w = output_transformer(t.graph)
    result_path = w.save(output_path, **kwargs)

    if result_path is not None and os.path.isfile(result_path):
        click.echo("File created at: " + result_path)
    elif os.path.isfile(output_path):
        click.echo("File created at: " + output_path)
    else:
        error("Could not create file.")

def _get_type(path):
    t = get_type(path)
    if t is None:
        error('Path do not have a recognized transformer type: {}'.format(path))
    else:
        return t

def load_transformer(input_paths:List[str], input_type:str=None) -> Transformer:
    """
    Creates a transformer for the appropriate file type and loads the data into
    it from file.
    """
    if input_type is None:
        input_types = [_get_type(i) for i in input_paths]
    else:
        input_types = [input_type for i in input_paths]

    t = None
    for path, input_type in zip(input_paths, input_types):
        constructor = get_transformer_constructor(input_type)

        if t is None:
            t = constructor()
        else:
            t = constructor(t)

        t.parse(path, input_type)

    t.report()

    return t
