include("enumeration_polynomials.jl")
using Distributions
using Random: rand
using DynamicPolynomials
using Cumulants: moment
using CumulantsUpdates: moms2cums!, cums2moms
using LinearAlgebra: dot
using SymmetricTensors
using Base.Iterators: product, flatten
using Combinatorics: partitions, with_replacement_combinations,multiset_permutations,permutations
using CairoMakie

######################
#define function here 
######################
function func(x, y)
    return sin(x)cos(y)
end 


###############################################
#get samples of multi-variate distribution here
###############################################
function get_samples_from_distribution(num_samples, type_of_distribution)
    return rand(type_of_distribution, num_samples)
end 

function generate_multivariate_data(num_samples, type_of_distribution, num_variables)
    arr_rand_variables = []
    for i in 1:num_variables 
        X = get_samples_from_distribution(num_samples, type_of_distribution)
        push!(arr_rand_variables, X)
    end 

    Z = zeros(num_samples, num_variables)
    for i in 1:num_variables 
        for j in 1:num_samples
            Z[j, i] = arr_rand_variables[i][j]
        end 
    end 

    return Z 
end 

#################################################################################
#get samples of multi-variate distribution while also using central limit theorem 
#################################################################################

function generate_central_limit_multivariate_data(n, num_samples, type_of_distribution, num_variables)
    arr_rand_variables = []
    for i in 1:num_variables 
        Z = get_samples_from_distribution(num_samples, type_of_distribution)
        for j in 1:(n-1)
            X = get_samples_from_distribution(num_samples, type_of_distribution)
            Z += X 
        end 
        Z /= n
        push!(arr_rand_variables, Z)
    end 

    Z = zeros(num_samples, num_variables)
    for i in 1:num_variables 
        for j in 1:num_samples
            Z[j, i] = arr_rand_variables[i][j]
        end 
    end 

    return Z 
end 

#################################################
#plot data function here (only for bivariate case)
#################################################
function plot_data_samples(bivariate_data)
    X = bivariate_data[:, 1]
    Y = bivariate_data[:, 2]

    f = Figure()
    hist(f[1, 1], X, bins = 40)
    hist(f[1, 2], Y, bins = 40)
    f
end 


#################################################
#stored moments calculation
#################################################
function compute_stored_moments(data, number_stored)
    return [moment(data, i) for i = 1 : number_stored]
end 

#################################################
#naive cumulant truncation 
#################################################
function compute_pseudo_moments_cumulant_truncation(data, number_stored, number_pseudo)
    num_variables = size(data, 2)
    moments_array = compute_stored_moments(data, number_stored)

    moms2cums!(moments_array)

    
    for j in number_stored+1:number_pseudo 
        dimensions = ones(Int8, 1, j)
        for k in 1:j 
            dimensions[k] *= num_variables
        end 
        cumulant = SymmetricTensor(zeros(Float64, Tuple(dimensions)))
        push!(moments_array, cumulant)
    end 

    return cums2moms(moments_array) 
end 

#################################################
#cumulant product order pseudo moments
#################################################
function complementary_partitions_upto_order(τ::AbstractArray{T,1},order::T) where T<: Integer
    # generate complementary partitions of a ordinary cumulant index set τ. τ contains only one block so all partitions
    # are are sub-paritions of the maximal partition
    # inputs -
    #       τ - vector of ordinary cumulant index set
    #       order - maximum cumulant product order
    # returns -
    #       complementary partitions - vector of partitions of the index set such that each partition and index 
    #       set are not both sub-partitions of any partition other than the maximal partition containing one block
    return partitions_of_order(τ,order)
end

function complementary_partitions_upto_order(τ::AbstractArray{T},order::Integer) where T <: AbstractArray
    # generate complementary partitions of a ordinary cumulant index set τ    
    # inputs -
    #       τ - vector of ordinary cumulant index set
    #       order - maximum cumulant product order
    # returns -
    #       complementary partitions - vector of partitions of the index set such that each partition and index 
    #       set are not both sub-partitions of any partition other than the maximal partition containing one block

    maximal_parition = collect(flatten(τ))
    all_partions = partitions_of_order(maximal_parition,order)
    if size(τ,1) == 1
        # τ contains only one block so all partitions are are sub-paritions of the maximal partition
        return all_partions
    else
        # generate sub partitions of τ
        partitions_of_partition_blocks = map( x -> partitions(x) , τ )
        combination_of_partitions_of_partition_blocks = product(partitions_of_partition_blocks...)
        sub_partition = map( tuple -> collect(flatten(tuple)) , combination_of_partitions_of_partition_blocks  ) 
        return [ i for i in  all_partions if i ∉ sub_partition ]
    end 
end


function partitions_of_order(τ::AbstractVector{T},order::T) where T<: Integer
    # generate partitions of an ordinary cumulant index set τ    
    # inputs -
    #       τ - vector of ordinary cumulant index set
    #       order - maximum cumulant product order
    # returns -
    #       partitions - vector of partitions of the index set that are of cumulant product order less than order
    nτ = size(τ,1) # size of index set
    #set order of size equal to or less than size of index set
    if order > nτ
        order = nτ
    end
    # paritions by the cumulant product order
    partition_by_order = map( i -> partitions(τ,i) , nτ-order+1:nτ)
    # combine all partitions below order
    parition_less_than_order = flatten(partition_by_order)
    return parition_less_than_order
end

function compute_generalized_moment(τ,order,cummulant_array)
    # compute generalized moment for index set τ  
    # inputs -
    #       τ - vector of generalized moment index set
    #       order - maximum cumulant product order
    #       cumulant_array - vector of cumulant arrays up to at least order
    # returns -
    #       κτ - generalized moment at index set τ 
 
    # obtain generalized moment complementary paritions up to cumulant product order
    p = complementary_partitions_upto_order(τ,order)
    κτ = 0
    for σ in p
        # compute cumulant product for partition σ
        κσ = 1
        for σi in σ
            # cumulants at parition sub-block
            nσi = size(σi,1)
            κσ *= cummulant_array[nσi][σi...]
        end 
        κτ += κσ
    end
    return κτ
end

function pseudo_moments_upto_cumulant_product_order(cummulant_array,order)
    # compute pseudo-moment array from cumulant array by only including cumulant up to certain cumulant product order
    # inputs -
    #       cumulant_array - vector of cumulant arrays
    #       order - maximum cumulant product order
    # returns -
    #       moment_array - vector of moment arrays 

    n = size(cummulant_array,1) #maximum order of cumulant
    m = size(cummulant_array[1],1) #dimension of cumulants

    #initalize moment array based on cumualnt array
    moment_array = copy(cummulant_array).*0

    for nτ in 1:n
        #compute pseudo-moments of order nτ
        for τ in  enumerate_unique_coefficients(1:m,nτ)
            κτ = compute_generalized_moment(τ,order,cummulant_array)
            set_index_symmetric_array!(moment_array[nτ],κτ,τ)
        end
    end

    return moment_array
end

function set_index_symmetric_array!(symmetric_array::AbstractArray{T},update,idx::Vector{Int64}) where T <: AbstractFloat
    # updates value in all matching locations in a symmetric tensor
    # inputs - 
    #       symmetric_array - symmetric array to be updated at a specific location
    #       update - value to be set at block_idx in symmetric_array
    #       idx - index of array to update
    n = ndims(symmetric_array) #tensor dimension
    for i in multiset_permutations(idx,n)
        symmetric_array[i...] = update #update symmetric value in array
    end 
end

function enumerate_unique_coefficients(iz,n)
    return with_replacement_combinations(iz,n)
end

#finalized function computing pseudo moments using cumulant product order 
function compute_pseudo_moments_cumulant_order(data, number_stored, max_product_order) 
    moments_tensors = compute_stored_moments(data, number_stored)
    moms2cums!(moments_tensors)

    moments_array = [Array(moments_tensors[1]), Array(moments_tensors[2])]

    for i in 3:number_stored 
        push!(moments_array, Array(moments_tensors[i]))
    end 

    return pseudo_moments_upto_cumulant_product_order(moments_array, max_product_order)
end 


###################################################
#select specific moment values from moment tensors 
###################################################

function compute_moments_vector(pseudo_moment_tensors, num_variables, order)
    moments_vector = [1.0]

    for i in 1:order
        parts = enumerate_monomials(i, num_variables)
        for (j, current) in enumerate(parts)
            #current = parts[j]
            indices = []
            for (k, power) in enumerate(current)
                while power > 0 
                    push!(indices, k)
                    power-=1
                end 
            end 
            moment = pseudo_moment_tensors[i][indices...]
            push!(moments_vector, moment)
        end 
    end 
    return moments_vector
end 

#################################################
#least squares regression for coefficients 
################################################
function compute_coefficients_general(func, num_variables:: Int64, order ::Int64, left, right, samples :: Int64)
    parts = collect(enumerate_monomials_up_to(order, num_variables))
    values_array = Array{Float64}(undef, samples^num_variables, length(parts))
    samples_vector = LinRange(left, right, samples)
    
    z_vector = Array{Float64}(undef, samples^num_variables)
    for (current_row, i) in enumerate(Iterators.product(repeat([samples_vector],num_variables)...))
        for j in (1:length(parts))
            values_array[current_row, j] = compute_value(i, parts[j])
            z_vector[current_row] = func(i...)
        end       
    end

    coefficients = values_array \ z_vector
    return coefficients
end 

function compute_value(variable_tuple, power_array)
    value = 1
    for i in axes(power_array, 1)
        value *= variable_tuple[i]^(power_array[i])
    end 
    return value
end 


function compute_coefficients_array(func, num_variables, samples)
    return [compute_coefficients_general(func, num_variables,i, -1.5, 1.5, samples) for i = 2:8]
end 

#####################################################
#compute approximation using moments and coefficients
#####################################################
function calculate_moments_approximation(coefficients_vector, moments_vector)
    return dot(moments_vector, coefficients_vector)
end


##############################################
#calculate expectation of function here
#############################################
function calculate_expectatioin(data, func)
    expectation = 0
    for i in 1:size(data, 1)
        expectation += func(data[i, :]...)
    end 

    return expectation/size(data, 1)
end 


##############################################################################################################################
#calculate the error of stored moments, naive truncation, and cumulant product order truncation with the actual expectation
#############################################################################################################################

function calculate_error(num_variables, distribution, num_samples, number_of_sets, func, coefficients_array)
   data_sets = [generate_multivariate_data(num_samples, distribution, num_variables) for i = 1 : number_of_sets]
   data_sets = [generate_central_limit_multivariate_data(20, num_samples, distribution, num_variables) for i = 1 : number_of_sets]
   final_stored_error_array = zeros(7)

   final_two_stored_six_pseudo_error = 0
   final_three_stored_six_pseudo_error = 0
   final_three_stored_seven_pseudo_error = 0
   final_two_stored_seven_pseudo_error = 0

   final_two_cumulant_order_six_order = 0
   final_three_cumulant_order_six_order = 0
   final_two_cumulant_order_seven_order = 0
   final_three_cumulant_order_seven_order = 0


   for i in 1:size(data_sets, 1)
       #calculating true expectation 
       true_expectation = calculate_expectatioin(data_sets[i], func)

       #calculating stored moments approximate expectation
       moments_tensor = compute_stored_moments(data_sets[i], 8)
       moments_vector_array = [compute_moments_vector(moments_tensor, num_variables, k) for k = 2:8]
       approx_expectation_array = [calculate_moments_approximation(coefficients_array[l-1], moments_vector_array[l-1]) for l = 2:8]
       stored_error_array = [abs(true_expectation - approx_expectation_array[l]) for l = 1:7]
       final_stored_error_array = final_stored_error_array .+ stored_error_array

       #calculating naive truncation for 2 stored moments, 6 pseudo moments 
       pseudo_moments_tensor = compute_pseudo_moments_cumulant_truncation(data_sets[i], 2, 6)
       moments_vector = compute_moments_vector(pseudo_moments_tensor, num_variables, 6)
       approx_expectation = calculate_moments_approximation(coefficients_array[5], moments_vector)
       final_two_stored_six_pseudo_error += abs(true_expectation - approx_expectation)

       #calculating naive truncation for 2 stored moments, 7 pseudo moments 
       pseudo_moments_tensor = compute_pseudo_moments_cumulant_truncation(data_sets[i], 2, 7)
       moments_vector = compute_moments_vector(pseudo_moments_tensor, num_variables, 7)
       approx_expectation = calculate_moments_approximation(coefficients_array[6], moments_vector)
       final_two_stored_seven_pseudo_error += abs(true_expectation - approx_expectation)

       #calculating naive truncation for 3 stoed moments, 6 pseudo moments 
       pseudo_moments_tensor = compute_pseudo_moments_cumulant_truncation(data_sets[i], 3, 6)
       moments_vector = compute_moments_vector(pseudo_moments_tensor, num_variables, 6)
       approx_expectation = calculate_moments_approximation(coefficients_array[5], moments_vector)
       final_three_stored_six_pseudo_error += abs(true_expectation - approx_expectation)

       #calculating naive truncation for 3 stored moments, 7 pseudo moments 
       pseudo_moments_tensor = compute_pseudo_moments_cumulant_truncation(data_sets[i], 3, 7)
       moments_vector = compute_moments_vector(pseudo_moments_tensor, num_variables, 7)
       approx_expectation = calculate_moments_approximation(coefficients_array[6], moments_vector)
       final_three_stored_seven_pseudo_error += abs(true_expectation - approx_expectation)

       #calculating cumulant product order truncation for order 6, max cumulant order 2 
       pseudo_moments_tensor = compute_pseudo_moments_cumulant_order(data_sets[i], 6, 2)
       moments_vector = compute_moments_vector(pseudo_moments_tensor, num_variables, 6)
       approx_expectation = calculate_moments_approximation(coefficients_array[5], moments_vector)
       final_two_cumulant_order_six_order += abs(true_expectation - approx_expectation)

       #calculating cumulant product order truncation for order 6, max cumulant order 3
       pseudo_moments_tensor = compute_pseudo_moments_cumulant_order(data_sets[i], 6, 3)
       moments_vector = compute_moments_vector(pseudo_moments_tensor, num_variables, 6)
       approx_expectation = calculate_moments_approximation(coefficients_array[5], moments_vector)
       final_three_cumulant_order_six_order += abs(true_expectation - approx_expectation)

       #calculating cumulant product order truncation for order 7, max cumulant order 2 
       pseudo_moments_tensor = compute_pseudo_moments_cumulant_order(data_sets[i], 7, 2)
       moments_vector = compute_moments_vector(pseudo_moments_tensor, num_variables, 7)
       approx_expectation = calculate_moments_approximation(coefficients_array[6], moments_vector)
       final_two_cumulant_order_seven_order += abs(true_expectation - approx_expectation)

       #calculating cumulant product order truncation for order 7, max cumulant order 3 
       pseudo_moments_tensor = compute_pseudo_moments_cumulant_order(data_sets[i], 7, 3)
       moments_vector = compute_moments_vector(pseudo_moments_tensor, num_variables, 7)
       approx_expectation = calculate_moments_approximation(coefficients_array[6], moments_vector)
       final_three_cumulant_order_seven_order += abs(true_expectation - approx_expectation)




   end
   final_stored_error_array /= size(data_sets, 1)
   for i in 1:7 
        println("The error for ", i + 1, " stored moments is ", final_stored_error_array[i])
   end 

   final_two_stored_six_pseudo_error /= size(data_sets, 1)
   println("The error for two stored six pseudo naive cumulant truncation: ", final_two_stored_six_pseudo_error)
   final_three_stored_six_pseudo_error /= size(data_sets, 1)
   println("The error for three stored six pseudo naive cumulant truncation: ", final_three_stored_six_pseudo_error)
   final_three_stored_seven_pseudo_error /= size(data_sets, 1)
   println("The error for three stored seven pseudo naive cumulant truncation: ", final_three_stored_seven_pseudo_error)
   final_two_stored_seven_pseudo_error /= size(data_sets, 1)
   println("The error for two stored seven pseudo naive cumulant truncation: ", final_two_stored_seven_pseudo_error)

   final_two_cumulant_order_six_order /= size(data_sets, 1)
   println("The error for two max six order cumulant product order truncation : ", final_two_cumulant_order_six_order)
   final_three_cumulant_order_six_order /= size(data_sets, 1)
   println("The error for three max six order cumulant product order truncation : ", final_three_cumulant_order_six_order)
   final_two_cumulant_order_seven_order /= size(data_sets, 1)
   println("The error for two max seven order cumulant product order truncation : ", final_two_cumulant_order_seven_order)
   final_three_cumulant_order_seven_order /= size(data_sets, 1)
   println("The error for three max seven order cumulant product order truncation : ", final_three_cumulant_order_seven_order)

end 


