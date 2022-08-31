from julia import Main
import networkx as nx
import scipy as sp
import scipy.sparse
import numpy as np
import scipy.linalg.lapack as la
from disjoint_set import DisjointSet
from operator import itemgetter


Main.include("function.jl")


def calculate_aproximation(vgraph, matrix):
    #subtract degree matrix from laplacian matrix
    nbr_nodes = len(vgraph)
    for ind in range(0, nbr_nodes):
        matrix[ind][ind] = 0
    matrix[np.tril_indices(matrix.shape[0], -1)] = 0
    Laplacian = sp.sparse.csc_matrix(matrix)

    #get number of edges, to build incidence matrix
    nbr_edges = sp.sparse.csc_matrix.getnnz(Laplacian)
    IncidenceMatrix = np.zeros((nbr_edges, nbr_nodes))
    AdjMatrix = np.zeros((nbr_nodes, nbr_nodes))


    #building the incidence matrix. Each row is an edge, to be in the correct shape for the multiplication
    #also building the adjacency matrix
    edgesInc = zip(*Laplacian.nonzero())
    
    i = 0
    for row, column in edgesInc:
        IncidenceMatrix[i][row] = 1
        IncidenceMatrix[i][column] = -1
        i = i + 1
        AdjMatrix[row][column] = 1
        AdjMatrix[column][row] = 1

    IncMatrix = sp.sparse.csc_matrix(IncidenceMatrix)
    AdjacencyMatrix = sp.sparse.csc_matrix(AdjMatrix)
    
    
    return Main.scipyCSC_to_julia(AdjacencyMatrix, IncMatrix, nbr_edges, list((zip(*Laplacian.nonzero()))), np.array(vgraph))
    



def seb_weighted(G):
    size = len(G.nodes())  
    edges = []
    for e in G.edges():
        edges.append([e[0],e[1],G[e[0]][e[1]]['weight']])
    edges = sorted(edges, key = itemgetter(2))
    nEdges = len(edges)
    current_weight = edges[0][2]

    ds = DisjointSet()
    mapsets = np.arange(size)
    mapaux = np.full(size, -1)
    vaux = []
    seb_values = {}
    temp_matrix = np.zeros((size, size))
    encoding = {}
    decoding = {}

    prev = 0
    nEdge = 0

    while(True):


        if(nEdge < nEdges and edges[nEdge][2] == current_weight): 
            
            u, v = edges[nEdge][0], edges[nEdge][1]
            if u not in vaux:
                vaux.append(u)
            if v not in vaux:
                vaux.append(v)

            #the edge's vertexes are transformed into the corresponding connected component values
            s = mapsets[u]
            d = mapsets[v]

            #correspondence between edge and cc
            encoding.update({(u,v): (s,d)})

            #correspondence between cc and edge. Seems redundant but later will be important for performance reasons
            if (s,d) in decoding:
                decoding[(s,d)] = decoding[(s,d)] + [(u,v)]
            else:
                decoding[(s,d)] = [(u,v)]


            #building the Laplacian matrix for this weight level's edges
            temp_matrix[s][d] += -1
            temp_matrix[d][s] += -1
            temp_matrix[s][s] += 1
            temp_matrix[d][d] += 1

            if(not ds.connected(u,v)):
                ds.union(u,v)
                
            nEdge += 1
            
        elif prev != nEdge:
            #give each vertex the corresponding connected component identifier
            mapid = 0
            for i in range(size):
                setid = ds.find(i)
                if mapaux[setid] == -1:
                    mapaux[setid] = mapid
                    mapaux[i] = mapid
                    mapid += 1
                else:
                    mapaux[i] = mapaux[setid]


            #create space for the connected components formed in the processed weight level
            calcdet = []
            for i in range(mapid):
                calcdet.append([])

            #vaux contains the edges that have been processed
            #mapset = mapaux from the previous iteration, i.e., the node is represented by the id of its connected component
            #keep in mind that no repeated connected component's id will be added, but there can be multiple edges between the same connected components
            for i in range(len(vaux)):
                if mapsets[vaux[i]] not in calcdet[mapaux[vaux[i]]]:
                    calcdet[mapaux[vaux[i]]].append(mapsets[vaux[i]])


            #calculating the aproximate values for each connected component
            for i in range(len(calcdet)):
                if len(calcdet[i]) != 0:
                    vgraph = calcdet[i].copy()
                    matrix = temp_matrix[vgraph, :] [:, vgraph]
                    results = calculate_aproximation(vgraph, matrix)

                    #the results need to be matched to the original vertexes numbers, so the information can be correctly updated to the graph
                    for result in list(results):
                        #print(result, "->", list(encoding.keys())[list(encoding.values()).index(result)])
                        value = results.pop(result)
                        #if there are multiple edges between the same connected components results will only have 1 of those
                        #so decoding and encoding its used to check if multiple vertexes have been translated to the same pair of ids
                        edges_aux = decoding[encoding[list(encoding.keys())[list(encoding.values()).index(result)]]]
                        size_of_edges_aux = len(edges_aux)
                        #if there are then the result should be equally divided among all the vertexes
                        if len(edges_aux) > 1:
                            for decode in edges_aux:
                                results[decode] = value/size_of_edges_aux
                        #if there are not then the final value is the value returned
                        else:
                            results[list(encoding.keys())[list(encoding.values()).index(result)]] = value
                    
                    #print("results", results)
                    seb_values.update(results)
                    

            prev = nEdge
            if (nEdge >= nEdges):
                break

            current_weight += 1
            temp_matrix = np.zeros((mapid, mapid))
            vaux = []
            encoding = {}
            decoding = {}
            mapsets = mapaux.copy()
            mapaux = np.full(size, -1)

        else:
            current_weight += 1

    nx.set_edge_attributes(G, seb_values,'SEB')
    return seb_values
    


G = nx.read_weighted_edgelist("simplenet", nodetype=int)

print(seb_weighted(G))