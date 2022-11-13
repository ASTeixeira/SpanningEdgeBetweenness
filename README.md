# SpanningEdgeBetweenness
Implementation of Spanning Edge Betweenness Centrality 

Papers published:

-- Teixeira, Andreia Sofia, et al. "Spanning edge betweenness." Workshop on mining and learning with graphs. Vol. 24. 2013

-- Teixeira, Andreia Sofia, et al. "Not seeing the forest for the trees: size of the minimum spanning trees (MSTs) forest and branch significance in MST-based phylogenetic analysis." PloS one 10.3 (2015).

-- Teixeira, Andreia Sofia, Francisco C. Santos, and Alexandre P. Francisco. "Spanning edge betweenness in practice." Complex Networks VII. Springer, Cham, 2016. 3-10.
APA	

# Java Implementation
Java implementation can be found here: https://bitbucket.org/phyloviz/popsim-analysis/src/master/


# SEB implementation details

# General Instructions

Nodes should be sequential as implementation uses indexation (based on node_id, should be integer) and number of nodes.  
Node's id should be integer.

## Exact Measure

### Dependencies:
```
import networkx as nx  
import scipy as sp  
import numpy as np  
from operator import itemgetter  
from disjoint_set import DisjointSetx  
import time  
```

### How to run:

```
python3 SEB.py
```

As of now it computes the Spanning Edge Betweenness for "example_network", also present in this repository.  
Either generate a new "example_network" with intended network's edgelist or change the code, to the new edgelist file's name.  
Another option is to delete the final 2 lines of the code, and use the remaining code as an API.


## Aproximate Measure

Approximated calculation uses Julia for the calculation of the centrality values.  
[Setup instructions](https://www.peterbaumgartner.com/blog/incorporating-julia-into-python-programs/).  
Or alternativelly, if you want to do as was done for this work:
```
pip install julia jill ipython --no-cache-dir
jill install 1.6.2 --confirm
python -c "import julia; julia.install()"
julia -e 'using Pkg; Pkg.add(["Revise", "BenchmarkTools", "PyCall", "SparseArrays", "Laplacians", "Statistics", "LinearAlgebra"])'
```

### How to run:
```
python-jl test.py <network_edgelist_name>
```

Also made to calculate the centrality values for <network_edgelist_name>. If not desired same options as exact implementation also apply here, only difference is that edgelist file's name is not hardcoded.

Versions:
```
julia version 1.6.2
Python 3.8.10
```
