# Matlab function for the Minimum absolute spectral similarity (MASS)
# Author Xiaoran Yan

The MASS function absSpecSim(A, threshold) takes a symmetric adjacency matrix A and a threshold percentile in the range [0,1] as inputs. It returns a MASS value in the range [0,1], measuring the similarity between the original graph A and the its thresholded counterpart.

The main.m provides a running example on a sythetic LFR network. It will provide a threhold profile similar to the MASS curve in Figure 1(b).

# Update: added a Python implementation
# Author Lucas Jeub

Simply run sparsify.py for a MASS profile. 

The MASS function is:
linear_sparsification_profile(edges, x_values):

    Parameters
    ----------
    edges : iterable of tuples
    x_values : iterable, fraction of edges or weight to remove
    
    Returns
    -------
    iterable
