using Combinatorics:partitions, multiset_permutations
using Base.Iterators: flatten, map


# enumerates the monomials of order n 
function enumerate_monomials(n, m)
    # The problem is that partitions(m, n) returns the partitions 
    # of up to m positive integers that sum to n
    # In order to instead work with nonnegative integers, 
    # we compute partitions(m, n + m). 
    # We then subtract one from each of the output vectors. 
    
    # This forms all partitions into nonnegative integers
    # It then replaces each of them with an iterator, iterating 
    # over all possible permutations.
    permuted_iterators = map(x -> multiset_permutations(x, m), partitions(n + m, m))
    # Here, we combine the iterators (that take care of the permutations) into 
    # one large iterator, iterating over all permutations of all partitions
    flattened_iterator = flatten(permuted_iterators)
    # Finally, we remove the spurious +1, which was only necessary to account for the 
    # fact that the partition function works with positive, instead of 
    # nonnegative partitions. 
    corrected_iterator = map(x -> x .- 1, flattened_iterator)  
    return corrected_iterator
end

function enumerate_monomials_up_to(n, m)
    return flatten([enumerate_monomials(k, m) for k = 0 : n])
end 

function evaluate_monomials(x,monomials)
    return [ prod(x.^p) for p in monomials ]
end