import networkx as nx
import numpy as np
import util


def create_graphs(adjacency_folder_path=None, list_attr_path=None):
    """
    Parameters:
    - file_path: path to adjacency matrices
    - start_ID: starting ID for graph nodes
    - list_attr_path: path to the list of vec features associated to graph nodes
    
    Return: a list of undirected, unweighted graphs
    """
    if list_attr_path is not None:
        list_attr = np.loadtxt(list_attr_path)
    filenames = util.list_files_in_folder(adjacency_folder_path)
    start_ID = 0
    graphs = []
    for filename in filenames:
        AM = np.loadtxt(adjacency_folder_path + "/" + filename)
        N = AM.shape[0]  
        g = nx.Graph()
        if list_attr_path is not None:
            for idx, nodeid in enumerate(range(start_ID, N+start_ID)):
                g.add_nodes_from(range(start_ID, N+start_ID), label="")
                g.add_node(nodeid, label="", vec=list(list_attr[idx]))
        else:
            g.add_nodes_from(range(start_ID, N + start_ID), label="")
        list_edges = []
        for u in range(N-1):
            for v in range(u+1, N):
                w = AM[u, v]
                if w != 0.0:
                    list_edges.append((u+start_ID, v+start_ID))
        g.add_edges_from(list_edges, label="")
        graphs.append(g)
        
        start_ID+= start_ID + N
            
    return graphs


def kcore_decompose(g=None, deg_threshold=None, cli_threshold=None, max_node_ID=None):
    """
    Parameters:
    - deg_threshold: Degree threshold
    - cli_threshold: Clique threshold
    - max_node_ID: Starting ID for new added nodes
    
    Return: decomposed graph
    """    
    max_node_ID = [max_node_ID]
    dict_newnode_clinodes = {}
    remove_edges = []
    
    high_degree_nodes = []
    low_degree_nodes = []
    degrees = [d[1] for d in g.degree()]
    m = min(degrees)
    t = max(deg_threshold, m)
    
    for n in g.nodes():       
        if len(list(g.neighbors(n))):
            high_degree_nodes.append(n)
        else:
            low_degree_nodes.append(n)
    g_high_degree0 = g.subgraph(high_degree_nodes)
    n_nodes = len(high_degree_nodes)
    
    g_low_degree0 = g.subgraph(low_degree_nodes).copy()
    clique_decompose(g_low_degree0, max_node_ID, cli_threshold, dict_newnode_clinodes, remove_edges)
    
    g_union = g_low_degree0.copy()
    while n_nodes > 0:        
        high_degree_nodes = []
        low_degree_nodes = []
        degrees = [d[1] for d in g_high_degree0.degree()]
        m = min(degrees)
        t = max(deg_threshold, m)
        
        for n in g_high_degree0.nodes():      
            if len(list(g_high_degree0.neighbors(n))) > t:
                high_degree_nodes.append(n)
            else:
                low_degree_nodes.append(n)

        g_high_degree = g_high_degree0.subgraph(high_degree_nodes)
        n_nodes = len(high_degree_nodes)
        
        g_low_degree = g_high_degree0.subgraph(low_degree_nodes).copy()

        clique_decompose(g_low_degree, max_node_ID, cli_threshold, dict_newnode_clinodes,remove_edges)

        g_union = nx.union(g_union, g_low_degree)
        
        g_high_degree0 = g_high_degree
  
    edges = g.edges()
    edges_union = g_union.edges()
    edges_nesting = list((set(edges) - set(edges_union))-set(remove_edges))
    g_union.add_edges_from(edges_nesting, label="", nesting='True')
    # Remove edges inside cliques
    clique_edges = []
    for nodes in dict_newnode_clinodes.values():
        for idx1 in range(len(nodes)-1):
            for idx2 in range(idx1+1,len(nodes)):
                if idx1 != idx2:
                    clique_edges.append((nodes[idx1],nodes[idx2]))
    g_union.remove_edges_from(clique_edges)
    g_union.remove_edges_from(remove_edges)
    return [g_union, dict_newnode_clinodes]  


def clique_decompose(g, max_node_ID, cli_threshold, dict_newnode_clinodes, remove_edges):
    """
    Clique decomposition
    """    
    list_cliques = [cli for cli in nx.find_cliques(g) if len(cli) > cli_threshold]
    new_node_IDs = []
    nesting_edges = []
    new_edges = []    
    for cli in list_cliques:     
        # Updage dict_nodeID_clinodes
        dict_newnode_clinodes[max_node_ID[0]] = cli
        # Add a new node
        new_node_IDs.append(max_node_ID[0])        
        # Add nesting edges connecting nodes in clique with the new node and edges
        # connecting all neighbors of cliques'nodes to the new node
        for n in cli:
            nesting_edges.append((n, max_node_ID[0]))
            for v in nx.neighbors(g, n):
                if v not in cli:
                    new_edges.append((v, max_node_ID[0]))
                    remove_edges.append((n, v))
        max_node_ID[0] = max_node_ID[0]+1
    g.add_nodes_from(new_node_IDs, label="")
    g.add_edges_from(nesting_edges, label="", nesting ='True')
    g.add_edges_from(new_edges, label="")


def union_graphs(graphs=None, deg_threshold=None, cli_threshold=None): 
    """Decompose a list of graphs by using kcore and clique techniques
     
       Return: 
       The decomposed graph
    """
    max_node_ID = len(graphs)*len(graphs[0].nodes())+1
    
    # Decompose graphs
    dict_newnode_clinodes_temp = {}
    new_graphs = []
    
    for g in graphs:
        [g_dec, dict_newnode_clinodes_0] = kcore_decompose(g=g, deg_threshold=deg_threshold, cli_threshold=cli_threshold, max_node_ID=max_node_ID)
        new_graphs.append(g_dec)
        dict_newnode_clinodes_temp.update(dict_newnode_clinodes_0)
        
        if len(dict_newnode_clinodes_0.keys()) != 0:
            max_node_ID = max(dict_newnode_clinodes_0.keys())+1
        else:
            max_node_ID+=1
    
    # Union obtained decompose graphs
    or_edges0 = set(nx.get_edge_attributes(new_graphs[0], 'nesting').keys())
    and_edges0 = set(new_graphs[0].edges()) - or_edges0    
    node0 = set(new_graphs[0].nodes())

    for g in new_graphs[1:]:
        or_edges1 = set(nx.get_edge_attributes(g, 'nesting').keys())
        and_edges1 = set(g.edges()) - or_edges1    
        node1 = set(g.nodes())
        
        or_edges0 = or_edges0.union(or_edges1)
        and_edges0 = and_edges0.union(and_edges1)
        node0 = node0.union(node1)            
    
    g_union = nx.Graph()
    g_union.add_nodes_from(list(node0), label="")
    g_union.add_edges_from(list(and_edges0), label="")
    g_union.add_edges_from(list(or_edges0), label="", nesting=True)
                      
    return g_union
