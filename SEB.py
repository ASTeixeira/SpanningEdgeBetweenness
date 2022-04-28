import networkx as nx
import scipy as sp
import scipy.sparse
import numpy as np
import scipy.linalg.lapack as la
from scipy.linalg import lu
from operator import itemgetter
from disjoint_set import DisjointSet
import time


def calc_edges_SEB(temp_matrix, edges, prev, nEdge, calcdet, calcNMSTsDet, mapaux, mapsets):
    seb = {}
    
    for edge in np.arange(prev, nEdge):
        u, v = edges[edge][0], edges[edge][1]
        s = mapsets[u]
        d = mapsets[v]
        
        if s == d:
            value = {(u, v) : 0}
            seb.update(value)
            continue
        
        vgraph = []
        for el in calcdet[mapaux[u]]:
            if el != s and el != d:
                vgraph.append(el)
                
        matrix = temp_matrix[vgraph, :] [:, vgraph]
        #calculated as in the Java version using Lapack factorization
        detCalc = 0
        if len(matrix) > 0:
            LU, piv, info = la.dgetrf(matrix)
            for x in np.diag(LU):
                detCalc += np.log10(np.abs(x))
        eMSTs = detCalc

        value = {(u, v) : round(10**(eMSTs-calcNMSTsDet[mapaux[u]]),3)}
        seb.update(value)
        
    return seb


def weighted_spark(edge, edges, temp_matrix, mapsets, calcdet, mapaux, calcNMSTsDet):
    u, v = edges[edge][0], edges[edge][1]
    s = mapsets[u]
    d = mapsets[v]
    
    if s == d:
        value = {(u, v) : 0}
        return value
    
    vgraph = []
    for el in calcdet[mapaux[u]]:
        if el != s and el != d:
            vgraph.append(el)
            
    matrix = temp_matrix[vgraph, :] [:, vgraph]
    #calculated as in the Java version using Lapack factorization
    detCalc = 0
    if len(matrix) > 0:
        LU, piv, info = la.dgetrf(matrix)
        for x in np.diag(LU):
            detCalc += np.log10(np.abs(x))
    eMSTs = detCalc

    value = {(u, v) : round(10**(eMSTs-calcNMSTsDet[mapaux[u]]),3)}
    return value


def seb_weighted_spark(G):

    
    conf = SparkConf().setAppName('test')
    sc = SparkContext(conf=conf)
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

    nmsts = 0

    f = open("SEBstats","w+")

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


            calcdet = []
            for i in range(mapid):
                calcdet.append([])

            for i in range(len(vaux)):
                if mapsets[vaux[i]] not in calcdet[mapaux[vaux[i]]]:
                    calcdet[mapaux[vaux[i]]].append(mapsets[vaux[i]])

            calcNMSTsDet = np.zeros(mapid)
            for i in range(len(calcdet)):
                if len(calcdet[i]) != 0:
                    vgraph = calcdet[i].copy()
                    vgraph = vgraph[1:]
                    matrix = temp_matrix[vgraph, :] [:, vgraph]
                    detCalc = 0
                    if len(matrix) > 0:
                        LU, piv, info = la.dgetrf(matrix)
                        for x in np.diag(LU):
                            detCalc += np.log10(np.abs(x))
                    msts = detCalc
                    nmsts += msts
                    calcNMSTsDet[i] = msts


            ###############################
            rangeRDD = sc.parallelize(np.arange(prev, nEdge))
            results = rangeRDD.map(partial(weighted_spark, temp_matrix=temp_matrix, edges=edges, mapsets=mapsets, calcdet=calcdet, mapaux=mapaux, calcNMSTsDet=calcNMSTsDet)).collect()
            for edge_value in results:
                seb_values.update(edge_value)
            ##############################

            prev = nEdge
            if (nEdge >= nEdges):
                break

            current_weight += 1
            temp_matrix = np.zeros((mapid, mapid))
            vaux = []
            mapsets = mapaux.copy()
            mapaux = np.full(size, -1)

        else:
            current_weight += 1

    nx.set_edge_attributes(G, seb_values,'SEB')
    f.write("Total of Minimum Spanning Trees: 10^" + str(round(nmsts, 3)) + "\n")
    for e in G.edges():
        f.write("" + str(e[0]) + " " + str(e[1]) + " " + str(G[e[0]][e[1]]['SEB']) + "\n")
    f.close()

    return G


def seb_weighted_mp(G):
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

    nmsts = 0

    f = open("SEBstats","w+")

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


            calcdet = []
            for i in range(mapid):
                calcdet.append([])

            for i in range(len(vaux)):
                if mapsets[vaux[i]] not in calcdet[mapaux[vaux[i]]]:
                    calcdet[mapaux[vaux[i]]].append(mapsets[vaux[i]])

            calcNMSTsDet = np.zeros(mapid)
            for i in range(len(calcdet)):
                if len(calcdet[i]) != 0:
                    vgraph = calcdet[i].copy()
                    vgraph = vgraph[1:]
                    matrix = temp_matrix[vgraph, :] [:, vgraph]
                    detCalc = 0
                    if len(matrix) > 0:
                        LU, piv, info = la.dgetrf(matrix)
                        for x in np.diag(LU):
                            detCalc += np.log10(np.abs(x))
                    msts = detCalc
                    nmsts += msts
                    calcNMSTsDet[i] = msts


            ###############################
            if __name__ == '__main__':
                pool = multiprocessing.Pool(multiprocessing.cpu_count())

                result_async = [pool.apply_async(weighted_spark, args = (edge, edges, temp_matrix, mapsets, calcdet, mapaux, calcNMSTsDet)) for edge in np.arange(prev, nEdge)]
                results = [r.get() for r in result_async]
                for edge_value in results:
                   seb_values.update(edge_value)
            ##############################

            prev = nEdge
            if (nEdge >= nEdges):
                break

            current_weight += 1
            temp_matrix = np.zeros((mapid, mapid))
            vaux = []
            mapsets = mapaux.copy()
            mapaux = np.full(size, -1)

        else:
            current_weight += 1

    nx.set_edge_attributes(G, seb_values,'SEB')
    f.write("Total of Minimum Spanning Trees: 10^" + str(round(nmsts, 3)) + "\n")
    for e in G.edges():
        f.write("" + str(e[0]) + " " + str(e[1]) + " " + str(G[e[0]][e[1]]['SEB']) + "\n")
    f.close()

    return G


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

    nmsts = 0

    f = open("SEBstats","w+")

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


            calcdet = []
            for i in range(mapid):
                calcdet.append([])

            for i in range(len(vaux)):
                if mapsets[vaux[i]] not in calcdet[mapaux[vaux[i]]]:
                    calcdet[mapaux[vaux[i]]].append(mapsets[vaux[i]])

            calcNMSTsDet = np.zeros(mapid)
            for i in range(len(calcdet)):
                if len(calcdet[i]) != 0:
                    vgraph = calcdet[i].copy()
                    vgraph = vgraph[1:]
                    matrix = temp_matrix[vgraph, :] [:, vgraph]
                    detCalc = 0
                    if len(matrix) > 0:
                        LU, piv, info = la.dgetrf(matrix)
                        for x in np.diag(LU):
                            detCalc += np.log10(np.abs(x))
                    msts = detCalc
                    nmsts += msts
                    calcNMSTsDet[i] = msts

            ###############################
            seb = calc_edges_SEB(temp_matrix, edges, prev, nEdge, calcdet, calcNMSTsDet, mapaux, mapsets)
            seb_values.update(seb)
            ###############################


            prev = nEdge
            if (nEdge >= nEdges):
                break

            current_weight += 1
            temp_matrix = np.zeros((mapid, mapid))
            vaux = []
            mapsets = mapaux.copy()
            mapaux = np.full(size, -1)

        else:
            current_weight += 1

    nx.set_edge_attributes(G, seb_values,'SEB')
    f.write("Total of Minimum Spanning Trees: 10^" + str(round(nmsts, 3)) + "\n")
    for e in G.edges():
        f.write("" + str(e[0]) + " " + str(e[1]) + " " + str(G[e[0]][e[1]]['SEB']) + "\n")
    f.close()


def unweighted_spark(e, matrix, value):
    eMSTs = calc_SEB(matrix, e, value)
    return {(e[0], e[1]) : eMSTs}
    

def seb_unweighted_mp(G): 
    f = open("SEBstats","w+")

    #Kirchhoff's Theorem
    Laplacian = sp.sparse.csr_matrix.toarray(nx.laplacian_matrix(G))
    toMSTs = np.delete(Laplacian, 0, 0)
    toMSTs = np.delete(toMSTs, 0, 1)


    detCalc = 0
    if len(toMSTs) > 0:
        LU, piv, info = la.dgetrf(toMSTs)
        for x in np.diag(LU):
            detCalc += np.log10(np.abs(x))
        nMSTs = detCalc

    f.write("Total of Minimum Spanning Trees: 10^" + str(round(nMSTs,3)) + "\n")


    if __name__ == '__main__':
        pool = multiprocessing.Pool(multiprocessing.cpu_count())

        result_async = [pool.apply_async(unweighted_spark, args = (e, Laplacian, nMSTs)) for e in G.edges()]
        results = [r.get() for r in result_async]
        results = dict(ChainMap(*results))
        nx.set_edge_attributes(G, results,'SEB')

    return G


def seb_unweighted_spark(G): 
    f = open("SEBstats","w+")

    conf = SparkConf().setAppName('test')
    sc = SparkContext(conf=conf)

    #Kirchhoff's Theorem
    Laplacian = sp.sparse.csr_matrix.toarray(nx.laplacian_matrix(G))
    toMSTs = np.delete(Laplacian, 0, 0)
    toMSTs = np.delete(toMSTs, 0, 1)


    detCalc = 0
    if len(toMSTs) > 0:
        LU, piv, info = la.dgetrf(toMSTs)
        for x in np.diag(LU):
            detCalc += np.log10(np.abs(x))
        nMSTs = detCalc

    f.write("Total of Minimum Spanning Trees: 10^" + str(round(nMSTs,3)) + "\n")


    edgelistRDD = sc.parallelize(G.edges())
    results = edgelistRDD.map(partial(unweighted_spark, matrix=Laplacian, value=nMSTs)).collect()
    results = dict(ChainMap(*results))
    nx.set_edge_attributes(G, results,'SEB')

    return G
    

def seb_unweighted(G): 
    f = open("SEBstats","w+")

    #Kirchhoff's Theorem
    Laplacian = sp.sparse.csr_matrix.toarray(nx.laplacian_matrix(G))
    toMSTs = np.delete(Laplacian, 0, 0)
    toMSTs = np.delete(toMSTs, 0, 1)

    detCalc = 0
    if len(toMSTs) > 0:
        LU, piv, info = la.dgetrf(toMSTs)
        for x in np.diag(LU):
            detCalc += np.log10(np.abs(x))
        nMSTs = detCalc

    f.write("Total of Minimum Spanning Trees: 10^" + str(round(nMSTs,3)) + "\n")

    for e in G.edges():
        eMSTs = calc_SEB(Laplacian, e, nMSTs)
        G[e[0]][e[1]]['SEB'] = eMSTs
        f.write("" + str(e[0]) + " " + str(e[1]) + " 10^" + str(eMSTs) + "\n")
    f.close()

    return G


def calc_SEB(matrix, e, nMSTs):


G = nx.read_edgelist("example_network", nodetype=int)
seb_unweighted(G)
