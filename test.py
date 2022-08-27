from julia import Main
import networkx as nx
import scipy as sp
import scipy.sparse
import numpy as np
import scipy.linalg.lapack as la
from disjoint_set import DisjointSet
from operator import itemgetter


Main.include("function.jl")


def testfunc(vgraph, matrix):
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

    prev = 0
    nEdge = 0

    while(True):


        if(nEdge < nEdges and edges[nEdge][2] == current_weight): 
            
            u, v = edges[nEdge][0], edges[nEdge][1]
            if u not in vaux:
                vaux.append(u)
            if v not in vaux:
                vaux.append(v)

            s = mapsets[u]
            d = mapsets[v]

            encoding.update({(u,v): (s,d)})


            temp_matrix[s][d] += -1
            temp_matrix[d][s] += -1
            temp_matrix[s][s] += 1
            temp_matrix[d][d] += 1

            if(not ds.connected(u,v)):
                ds.union(u,v)
                
            nEdge += 1
            
        elif prev != nEdge:
            mapid = 0
            for i in range(size):
                setid = ds.find(i)
                if mapaux[setid] == -1:
                    mapaux[setid] = mapid
                    mapaux[i] = mapid
                    mapid += 1
                else:
                    mapaux[i] = mapaux[setid]


            #criar espaco para os cc formados neste nivel
            calcdet = []
            for i in range(mapid):
                calcdet.append([])

            #vaux contem os nodes cujos edges foram processados
            #mapset = mapaux da iteracao anterior, ou seja, o node e representado pelo id do seu cc
            for i in range(len(vaux)):
                if mapsets[vaux[i]] not in calcdet[mapaux[vaux[i]]]:
                    calcdet[mapaux[vaux[i]]].append(mapsets[vaux[i]])



            for i in range(len(calcdet)):
                if len(calcdet[i]) != 0:
                    vgraph = calcdet[i].copy()
                    matrix = temp_matrix[vgraph, :] [:, vgraph]
                    results = testfunc(vgraph, matrix)
                    for result in list(results):
                        #print(result, "->", list(encoding.keys())[list(encoding.values()).index(result)])
                        results[list(encoding.keys())[list(encoding.values()).index(result)]] = results.pop(result)
                    print("results", results)
                    seb_values.update(results)
                    

            prev = nEdge
            if (nEdge >= nEdges):
                break

            current_weight += 1
            temp_matrix = np.zeros((mapid, mapid))
            vaux = []
            encoding = {}
            mapsets = mapaux.copy()
            mapaux = np.full(size, -1)

        else:
            current_weight += 1

    nx.set_edge_attributes(G, seb_values,'SEB')
    return seb_values
    





G = nx.read_weighted_edgelist("simplenet", nodetype=int)

print(seb_weighted(G))

#A = nx.adjacency_matrix(G)

#L = nx.laplacian_matrix(G)


#print(L)

#Adjacency = sp.sparse.csc_matrix(nx.adjacency_matrix(G))

#Laplacian = sp.sparse.csc_matrix(nx.laplacian_matrix(G))

#L2 = [[4,-3,-1,0], [-2,4,-1,-1], [-2,-3,6,-1], [0,-3,-1,4]]


#ret = Main.scipyCSC_to_julia(Adjacency)

#print(ret)


#Main.eval("A = 5")
#Main.eval("A = A + 1")
#Main.eval("println(A)")

#LAP = lap(A)

#print(Main.methods(approxchol_lap))

#print(LAP)

#a = approxchol_lap(Laplacian)

#print(a)

#Main.println("I'm printing from a Julia function!")
#Main.eval('using Pkg; Pkg.add("Laplacians")')

#print("hy from python")