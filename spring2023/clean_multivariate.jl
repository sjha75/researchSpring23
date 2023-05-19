using Distributions: Normal, Exponential, Gamma
using Random: rand
using DynamicPolynomials
using Cumulants: moment
using CumulantsUpdates: moms2cums!, cums2moms
using LinearAlgebra: dot
using SymmetricTensors



#testing pseudo-distributions for Z = sin(X)cos(Y)
#returns given number of samples of non normal distribution
function generate_data(samples)
    mu = 0
    sigma = 1

    normal_distribution = Normal(mu, sigma)
    x_samples = rand(normal_distribution, samples)
    y_samples = rand(normal_distribution, samples)

    z = x_samples .* y_samples 
    return z
end

#given distribution, returns number of samples from distribution
function get_samples_data(samples, distribution)
    return rand(distribution, samples)
end 



#returns bivariate non-gaussian distribution of given number of samples
function generate_multivariate_data(samples)
    X = generate_data(samples)
    Y = generate_data(samples)
    
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

#given two samples of distributions, returns bivariate non-gaussian distribution of given number of samples 
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

#returns vector of coefficients for least squares regression for given order
function compute_coefficients_vector(order, samples, left, right)
    values_array = Array{Float64}(undef, samples*samples, binomial(order+2, order))
    x_vect = LinRange(left, right, samples)
    y_vect = LinRange(left, right, samples)
    z_vect = Array{Float64}(undef, samples * samples)
    current_row = 0

    for i in (1: length(x_vect))
        for j in (1:length(y_vect))
            current_row += 1
            z_vect[current_row] = sin(x_vect[i])*cos(y_vect[j])
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

#calculates approximate expectation of function of random variable
function calculate_moments_approximation(coefficients_vector, moments_vector)
    return dot(moments_vector, coefficients_vector)
end 

#calculates actual expectation of function of random variable 
function calculate_expectation(data)
    expectation = 0 
    for i in axes(data, 1)
        expectation += sin(data[i, 1])*cos(data[i, 2])
    end 
    expectation /= size(data, 1)
    return expectation 
end 

#calculates average error of approx expectation using stored moments
function calculate_error_stored(number_stored, coefficients_vector, data_array) 
    error = 0 
    for i in 1: length(data_array)
        expectation = calculate_expectation(data_array[i])
        stored_moments_tensors = compute_stored_moments(data_array[i], number_stored)
        stored_moments_vector = get_moments_vector(stored_moments_tensors)
        approx_expectation = calculate_moments_approximation(coefficients_vector, stored_moments_vector)
        error += abs(expectation - approx_expectation)
      
    end 

    return error /= length(data_array)
end



function calculate_error_pseudo(number_stored, number_pseudo, coefficients_vector, data_array)
    error = 0
    for i in 1:length(data_array) 
        expectation = calculate_expectation(data_array[i])
        pseudo_moments = compute_pseudo_moments(data_array[i], number_stored, number_pseudo)
        pseudo_vector = get_moments_vector(pseudo_moments)
        pseudo_approximation = calculate_moments_approximation(coefficients_vector, pseudo_vector)
        error += abs(expectation - pseudo_approximation)
    end 
    error /= length(data_array)
    return error
end 

function create_data_array(amount, samples, distribution) 
    data1 = generate_multivariate_data(samples)
    data2 = generate_multivariate_data(sample)
    data_array = [data1, data2]
    for i in 3:amount 
        data_new = generate_multivariate_data(samples)
        push!(data_array, data_new)
    end 

    return data_array
end 


function main(samples, amount, distribution)

    coefficient_vector_order2 = compute_coefficients_vector(2, 1000, -1.5, 1.5)
    coefficient_vector_order4 = compute_coefficients_vector(4, 1000, -1.5, 1.5)
    data_array = create_data_array(amount, samples, distribution)
    println(calculate_error_stored(2, coefficient_vector_order2, data_array))
    println(calculate_error_stored(4, coefficient_vector_order4, data_array))

end 

function test_stored_moments(coefficients_vector, order, data)
    num_samples = length(data[:,1])
    @polyvar x y 
    xy_monomial = monomials([x,y], 0:order)

    func = mapreduce(*, +, xy_monomial, coefficients_vector)

    sum = 0 
    for i in 1:num_samples
        sum += func(data[i, 1], data[i, 2])
    end 

    return sum / num_samples
end 

function least_squares_error(order, data) 
    coefficients = compute_coefficients_vector(order, 1000, -1.5, 1.5) 
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


coefficients_vector_eigth_order = compute_coefficients_vector(8, 1000, -1.5, 1.5)
coefficients_vector_seventh_order = compute_coefficients_vector(7, 1000, -1.5, 1.5)
coefficients_vector_fourth_order = compute_coefficients_vector(4, 1000, -1.5, 1.5)
coefficients_vector_third_order = compute_coefficients_vector(3, 1000, -1.5, 1.5)
coefficients_vector_second = compute_coefficients_vector(2, 1000, -1.5, 1.5)

error_second_order = 0 
error_third_order = 0 
error_fourth_order = 0
error_seventh_order = 0
error_eigth_order = 0


for i in 1:1 
    data = generate_multivariate_data(1000000)
    true_expectation = calculate_expectation(data)


    expectation_fourth_order = test_stored_moments(coefficients_vector_fourth_order, 4, data)
    expectation_third_order = test_stored_moments(coefficients_vector_third_order, 3, data)
    expectation_second_order = test_stored_moments(coefficients_vector_second, 2, data)
    expectation_seventh_order = test_stored_moments(coefficients_vector_seventh_order, 7, data)
    expectation_eigth_order = test_stored_moments(coefficients_vector_eigth_order, 8, data)


    error_second_order += abs(expectation_second_order - true_expectation)
    error_third_order += abs(expectation_third_order - true_expectation)
    error_fourth_order += abs(expectation_fourth_order - true_expectation)
    error_seventh_order += abs(expectation_seventh_order - true_expectation)
    error_eigth_order += abs(expectation_eigth_order - true_expectation)
end 

println(error_second_order)
println(error_third_order)
println(error_fourth_order)
println(error_seventh_order)
println(error_eigth_order)


distribution = Gamma(0.5)
data = get_multivariate_data(1000, distribution)

error2 = least_squares_error(2, data)
error3 = least_squares_error(3, data)
error4 = least_squares_error(4, data)
error5 = least_squares_error(5, data)


println(error3)