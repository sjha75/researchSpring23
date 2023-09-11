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
using CSV
using DataFrames
using CairoMakie
using SymPy
using Integrals


##################################################
#get data
##################################################
function create_synthetic_data(samples, normal_skewness, func1, func2)
    skewed_normal = SkewNormal(0.5, 0.5, normal_skewness)
    X = rand(skewed_normal, samples)
    Y = rand(skewed_normal, samples)
    Z = zeros(samples, 2)

    for i in 1:samples 
        Z[i, 1] = func1(X[i], Y[i])
    end 
    for j in 1:samples 
        Z[j, 2] = func2(X[j], Y[j])
    end 
    
    return Z/n
    
end 

function central_limitize_synthetic_data(samples, normal_skewness, func1, func2, n_value)
    skewed_normal = SkewNormal(0.5, 0.5, normal_skewness)
    Z = zeros(samples, 2)

    for i in 1:n_value
        Z_temp = zeros(samples, 2)
        X = rand(skewed_normal, samples)
        Y = rand(skewed_normal, samples)
        for i in 1:samples 
            Z_temp[i, 1] = func1(X[i], Y[i])
        end 
        for j in 1:samples 
            Z_temp[j, 2] = func2(X[j], Y[j])
        end 
        Z = Z + Z_temp

    end 

    return Z/n_value

end 

function get_data(filename) 
    data = CSV.read(filename, DataFrame)
    data = Matrix(data)
    return data
end 

function get_specific_columns_data(array_of_indices, data)
    Z = zeros(size(data, 1), length(array_of_indices))
    for i in 1:length(array_of_indices)
       Z[:, i] = data[:, array_of_indices[i]]
    end 
    return Z
end  

function get_random_rows_data(data, samples)
    total_rows = size(data, 1)
    random_rows = Array{Float64}(undef, samples, size(data, 2))
    for i in 1:samples
        selected_row = rand(1:total_rows)
        for j in 1:size(data, 2)
            random_rows[i, j] = data[selected_row, j]
        end
    end 
    return random_rows
end 


######################
#define function here 
######################
function func0(x, y, parameters)
    return tan(parameters[1] * x + parameters[2]*y)
end 


function func1(x, y, parameters)
    return sin(parameters[1] * x) * cos(parameters[2] * y)
end 

function func2(x, y, parameters)
    if (parameters[2] * y > 0.3)
        return parameters[1] * (x^2) 
    else 
        return 0
    end 
end 
#################################################
#plot data function here (only for bivariate case)
#################################################
function plot_data_samples(bivariate_data)
    X = bivariate_data[:, 1]
    Y = bivariate_data[:, 2]

    f = Figure()
    ax1 = Axis(f[1, 1])
    scatter!(ax1, X, Y)
    f
end 

##################################################
#plot errors for moments approximation here 
#################################################
function plot_error(error_array)
    f = Figure()
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    y = error_array
    ax = Axis(f[1, 1], yscale = log10)
    ax.xticks = 1:10
    ax.xtickformat = x -> ["2", "3", "4", "(2, 6)", "(3, 6)", "(2, 7)", "(3, 7)", "cpo (2, 6)", "cpo (3, 6)", "cpo (2, 7)", "cpo (3, 7)"]
    scatter!(ax, x, y)
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
function compute_coefficients_general(func, num_variables:: Int64, order ::Int64, left, right, samples :: Int64, parameters_vector)
    parts = collect(enumerate_monomials_up_to(order, num_variables))
    values_array = Array{Float64}(undef, samples^num_variables, length(parts))
    samples_vector = LinRange(left, right, samples)
    
    z_vector = Array{Float64}(undef, samples^num_variables)
    for (current_row, i) in enumerate(Iterators.product(repeat([samples_vector],num_variables)...))
        for j in (1:length(parts))
            values_array[current_row, j] = compute_value(i, parts[j])
            z_vector[current_row] = func(i..., parameters_vector)
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


function compute_coefficients_array(func, num_variables, samples, parameters_vector, start_range, end_range)
    return [compute_coefficients_general(func, num_variables,i, start_range, end_range, samples, parameters_vector) for i = 2:8]
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
function calculate_expectation(data, func, parameters_vector)
    expectation = 0
    for i in 1:size(data, 1)
        expectation += func(data[i, :]..., parameters_vector)
    end 

    return expectation/size(data, 1)
end 


##############################################################################################################################
#calculate the error of stored moments, naive truncation, and cumulant product order truncation with the actual expectation
#############################################################################################################################

function calculate_error(num_variables, func, parameters_vector, coefficients_array, data)
    true_expectation = calculate_expectation(data, func, parameters_vector)
    error_array = []

    moments_tensors = compute_stored_moments(data, 2)
    moments_vector = compute_moments_vector(moments_tensors, num_variables, 2)
    approx_expectation = calculate_moments_approximation(coefficients_array[1], moments_vector)
    push!(error_array, abs(true_expectation - approx_expectation))

    moments_tensors = compute_stored_moments(data, 3)
    moments_vector = compute_moments_vector(moments_tensors, num_variables, 3)
    approx_expectation = calculate_moments_approximation(coefficients_array[2], moments_vector)
    push!(error_array, abs(true_expectation - approx_expectation))

    moments_tensors = compute_stored_moments(data, 4)
    moments_vector = compute_moments_vector(moments_tensors, num_variables, 4)
    approx_expectation = calculate_moments_approximation(coefficients_array[3], moments_vector)
    push!(error_array, abs(true_expectation - approx_expectation))

    moments_tensors = compute_pseudo_moments_cumulant_truncation(data, 2, 6)
    moments_vector = compute_moments_vector(moments_tensors, num_variables, 6)
    approx_expectation = calculate_moments_approximation(coefficients_array[5], moments_vector)
    push!(error_array, abs(true_expectation - approx_expectation))
    
    moments_tensors = compute_pseudo_moments_cumulant_truncation(data, 3, 6)
    moments_vector = compute_moments_vector(moments_tensors, num_variables, 6)
    approx_expectation = calculate_moments_approximation(coefficients_array[5], moments_vector)
    push!(error_array, abs(true_expectation - approx_expectation))

    moments_tensors = compute_pseudo_moments_cumulant_truncation(data, 2, 7)
    moments_vector = compute_moments_vector(moments_tensors, num_variables, 7)
    approx_expectation = calculate_moments_approximation(coefficients_array[6], moments_vector)
    push!(error_array, abs(true_expectation - approx_expectation))

    moments_tensors = compute_pseudo_moments_cumulant_truncation(data, 3, 7)
    moments_vector = compute_moments_vector(moments_tensors, num_variables, 7)
    approx_expectation = calculate_moments_approximation(coefficients_array[6], moments_vector)
    push!(error_array, abs(true_expectation - approx_expectation))

    moments_tensors = compute_pseudo_moments_cumulant_order(data, 6, 2)
    moments_vector = compute_moments_vector(moments_tensors, num_variables, 6)
    approx_expectation = calculate_moments_approximation(coefficients_array[5], moments_vector)
    push!(error_array, abs(true_expectation - approx_expectation))

    moments_tensors = compute_pseudo_moments_cumulant_order(data, 6, 3)
    moments_vector = compute_moments_vector(moments_tensors, num_variables, 6)
    approx_expectation = calculate_moments_approximation(coefficients_array[5], moments_vector)
    push!(error_array, abs(true_expectation - approx_expectation))

    moments_tensors = compute_pseudo_moments_cumulant_order(data, 7, 2)
    moments_vector = compute_moments_vector(moments_tensors, num_variables, 7)
    approx_expectation = calculate_moments_approximation(coefficients_array[6], moments_vector)
    push!(error_array, abs(true_expectation - approx_expectation))

    moments_tensors = compute_pseudo_moments_cumulant_order(data, 7, 3)
    moments_vector = compute_moments_vector(moments_tensors, num_variables, 7)
    approx_expectation = calculate_moments_approximation(coefficients_array[6], moments_vector)
    push!(error_array, abs(true_expectation - approx_expectation))

    return error_array
end 

function calculate_all_coefficients_array(num_variables, func, start_param, end_param, total_params, start_range, end_range)
    x = range(start_param, stop = end_param, length=total_params)
    all_coefficients_array = []
    for i in x
        for j in x
            parameters_vector = [i, j]
            coefficients_array = compute_coefficients_array(func, num_variables, 1000, parameters_vector, start_range, end_range)
            push!(all_coefficients_array, coefficients_array)
        end 
    end
    return all_coefficients_array
end 

function calculate_error_generalized(num_variables, func, data, start_param, end_param, total_params, all_coefficients_array)
    current_coefficient = 0
    x = range(start_param, stop = end_param, length = total_params)
    error_generalized_array = zeros(11)
    for i in x
        for j in x
            current_coefficient+=1 
            parameters_vector = [i, j]
            current_coefficient_array = all_coefficients_array[current_coefficient]
            current_error_array = calculate_error(num_variables, func, parameters_vector, current_coefficient_array, data)
            error_generalized_array += current_error_array
        end 
    end 
    return error_generalized_array/(total_params^2)
   
end 

function bootstrap(data, func, num_variables, bootstrap_size, samples, start_param, end_param, total_params, all_coefficients_array)
    bootstrap_dist = []
    for i in 1:samples
        bootstrap_data = get_random_rows_data(data, bootstrap_size)
        bootstrap_data_point = calculate_error_generalized(num_variables, func, bootstrap_data, start_param, end_param, total_params, all_coefficients_array)
        push!(bootstrap_dist, bootstrap_data_point)
    end
    return bootstrap_standard_error(bootstrap_dist, samples)

end

function bootstrap_standard_error(bootstrap_dist, samples)
    standard_error_array = []
    for i in 1:10 
        current_dist = zeros(samples)
        for j in 1:samples 
            current_dist[j] = bootstrap_dist[j][i]
        end
        standard_deviation = std(current_dist)
        standard_error = (standard_deviation/sqrt(samples))
        push!(standard_error_array, standard_error)
    end
    return standard_error_array

end 



function data_func1(x, y)
    return x^2 + y^4
end 

function data_func2(x, y)
    return x^2 + y
end 

function data_func3(x, y)
    return 0.5x + y^3
end 

function data_func4(x, y)
    return x^2 + y^2
end 

function data_func5(x, y)
    return 0.7x^2 + 0.4y
end 

function data_func6(x, y)
    return sin(x)cos(y)
end 


