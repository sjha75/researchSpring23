using Distributions
using Random: rand
using DynamicPolynomials
using Cumulants: moment
using CumulantsUpdates: moms2cums!, cums2moms
using LinearAlgebra: dot
using SymmetricTensors
using CairoMakie



#function being tested on 
function func(x, y)
    return sin(x)cos(y)
end 

#generating multi-variate data, given type of distribution and number of samples
function get_samples_data(samples, distribution)
    return rand(distribution, samples)
end 

function get_multivariate_data(samples, distribution)
    X = get_samples_data(samples, distribution)
    Y = get_samples_data(samples, distribution)

    Z = zeros(samples, 2)
    length = samples 
    for i in 1:length
        Z[i] = X[i]
    end
    
    for i in 1:length
       Z[i, 2] = Y[i]
    end

    #return matrix 
    return Z

end

function plot_data_samples(multivariate_data)
    X = multivariate_data[:, 1]
    Y = multivariate_data[:, 2]

    f = Figure()
    hist(f[1, 1], X, bins = 40)
    hist(f[1, 2], Y, bins = 40)
    f
end 

#computes stored moments, returning array of symmetric tensors
function compute_stored_moments(data, number_stored)
    return [moment(data, i) for i = 1 : number_stored]
end 

#computes pseudo moments, returning array of symmetric tensors
function compute_pseudo_moments(data, number_stored, number_pseudo)
    moments_array = [moment(data, 1), moment(data, 2)]
    for i in 3:number_stored
        push!(moments_array, moment(data, i))
    end 

    moms2cums!(moments_array)
     #= Sets cumulants of order from number_stored + 1 up to number_pseudo 
        to 0 and adds it to cumulant array =#
        for j in number_stored+1:number_pseudo 
            #sets dimension of symmetric tensor 
            dimensions = ones(Int8, 1, j)
            #symmetric tensor requires block size to be greater than or equal 2
            for k in 1:j 
                dimensions[k] *= 2
            end 
            #create symmetric tensor of 0 
            cumulant = SymmetricTensor(zeros(Float64, Tuple(dimensions)))
            #add to cumulants array 
            push!(moments_array, cumulant)
        end   
    
    #converts cumulants back to moments and returns array of moments, now containing pseudo moments
    return cums2moms(moments_array) 

end 

#chooses certain indices of tensors, returning array of moments as floating points
function get_moments_vector(moments_array)
    moments_vector = Array{Float64}(undef, 0)
    current_index = length(moments_array)
    while(current_index > 0)
        indices = ones(Int64, current_index)
        get_moments_helper(indices, moments_vector, moments_array)
        current_index -= 1
    end 

    #for constant term 
    push!(moments_vector, 1)

    return moments_vector
end 

#helper for choosing moments 
function get_moments_helper(indices, moments_vector, moments_array)
    size = length(indices) + 1
    for i in 1:size
        push!(moments_vector,moments_array[length(indices)][indices...])
        if (i < size)
            indices[i] = 2
        end 
    end
end 


#returns vector of coefficients for least squares regression for given order
function compute_coefficients_vector(order, samples, left, right, func)
    values_array = Array{Float64}(undef, samples*samples, binomial(order+2, order))
    x_vect = LinRange(left, right, samples)
    y_vect = LinRange(left, right, samples)
    z_vect = Array{Float64}(undef, samples * samples)
    current_row = 0

    for i in (1: length(x_vect))
        for j in (1:length(y_vect))
            current_row += 1
            z_vect[current_row] = func(x_vect[i], y_vect[j])
            current_monomial = compute_monomial(x_vect[i], y_vect[j], order)
            for k in (1: length(current_monomial))
                values_array[current_row, k] = current_monomial[k]
            end 
        end 
    end 

    return values_array \ z_vect

end 

#helper for least squares regression
function compute_monomial(x_value, y_value, order)
    @polyvar x y 
    xy_monomial = monomials([x, y], 0:order)
    values = subs(xy_monomial, x => x_value, y => y_value)
    return values 
end 

function compute_coefficients_array(func)
    return [compute_coefficients_vector(i, 1000, -1.5, 1.5, func) for i = 2:8]
end


#calculates approximate expectation of function of random variable
function calculate_moments_approximation(coefficients_vector, moments_vector)
    return dot(moments_vector, coefficients_vector)
end

#calculates actual expectation of function of random variable 
function calculate_expectation(data, func)
    expectation = 0 
    for i in (1:size(data, 1))
        expectation += func(data[i,1], data[i, 2])
    end 
    expectation /= size(data, 1)
    return expectation 
end


function calculate_error_stored(largest_order, coefficients_array, samples, distribution, func)
    data = get_multivariate_data(samples, distribution)
    error_array = []
    true_expectation = calculate_expectation(data, func)
    
    for i in 2:largest_order
        stored_tensors = compute_stored_moments(data, i)
        stored_vector = get_moments_vector(stored_tensors)
        approx_expectation = calculate_moments_approximation(coefficients_array[i-1], stored_vector)
        error = abs(true_expectation - approx_expectation)
        push!(error_array, error)
    end 

    return error_array
end 


function calculate_error_pseudo(coefficients_array, samples, distribution, func)
    data = get_multivariate_data(samples, distribution)
    true_expectation = calculate_expectation(data, func)

    stored_tensors = compute_stored_moments(data, 2)
    stored_vector = get_moments_vector(stored_tensors)
    approx_expectation = calculate_moments_approximation(coefficients_array[1], stored_vector)
    println("The error for 2 stored: ", abs(true_expectation - approx_expectation))

    pseudo_tensors_6 = compute_pseudo_moments(data, 2, 6)
    pseudo_vector_6 = get_moments_vector(pseudo_tensors_6)
    approx_expectation = calculate_moments_approximation(coefficients_array[5], pseudo_vector_6)
    println("The error for 2 stored, 6 pseudo: ", abs(true_expectation - approx_expectation))


    pseudo_tensors_7 = compute_pseudo_moments(data, 2, 7)
    pseudo_vector_7 = get_moments_vector(pseudo_tensors_7)
    approx_expectation = calculate_moments_approximation(coefficients_array[6], pseudo_vector_7)
    println("The error for 2 stored, 7 pseudo: ", abs(true_expectation - approx_expectation))

    stored_tensors = compute_stored_moments(data, 3)
    stored_vector = get_moments_vector(stored_tensors)
    approx_expectation = calculate_moments_approximation(coefficients_array[2], stored_vector)
    println("The error for 3 stored: ", abs(true_expectation - approx_expectation))

    pseudo_tensors_6 = compute_pseudo_moments(data, 3, 6)
    pseudo_vector_6 = get_moments_vector(pseudo_tensors_6)
    approx_expectation = calculate_moments_approximation(coefficients_array[5], pseudo_vector_6)
    println("The error for 3 stored, 6 pseudo: ", abs(true_expectation - approx_expectation))


    pseudo_tensors_7 = compute_pseudo_moments(data, 3, 7)
    pseudo_vector_7 = get_moments_vector(pseudo_tensors_7)
    approx_expectation = calculate_moments_approximation(coefficients_array[6], pseudo_vector_7)
    println("The error for 3 stored, 7 pseudo: ", abs(true_expectation - approx_expectation))
end 

function least_squares_error(data, order, coefficients_array)
    coefficients = coefficients_array[order - 1]
    @polyvar x y 
    xy_monomial = monomials([x,y], 0: order)   
    polynomial = mapreduce(*, +, coefficients, xy_monomial) 
    x_vector = data[:,1]    
    y_vector = data[:,2]

    error = 0 
    for i in 1:(size(data, 1))
        temp = polynomial(x_vector[i], y_vector[i]) - sin(x_vector[i])*cos(y_vector[i])
        error += abs(temp)
    end
    error /= size(data, 1)

    return error
end 

distribution = Exponential(0.5)
data = get_multivariate_data(100000, distribution)
plot_data_samples(data)
coefficients_array = compute_coefficients_array(func)
print(coefficients_array[1])
coefficients_array[2]

calculate_error_stored(7, coefficients_array, 1000000, distribution, func)
calculate_error_pseudo(coefficients_array, 100000, distribution, func)



