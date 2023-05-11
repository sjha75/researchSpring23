using Distributions
using StatsBase
import Statistics as stats
using DataFrames
using CSV
using LinearAlgebra
using Expectations
using Cumulants
using Combinatorics
using SymmetricTensors
using Polynomials
using DynamicPolynomials
using CumulantsUpdates

#chooses random samples from each of the normal distributions and multiplies them together to get 
function generateData(max_mu, max_sigma, samples) 
    #randomly selects the means and standard deviations
    mu1 = rand(0:max_mu)
    mu2 = rand(0:max_mu)
    sigma1 = rand(0.1:max_sigma)
    sigma2 = rand(0.1:max_sigma)

#generate normal distributions using randomly chosen mean and standard deviation 
    X = Normal(mu1, sigma1)
    Y = Normal(mu2, sigma2)
    x_samples = rand(X, samples)
    y_samples = rand(Y, samples)
    z = x_samples .* y_samples
    return z 
end

#places two random variables 
function generateMultivariate(max_mu, max_sigma, samples) 
    #generate two non-normal random variables 
    X = generateData(max_mu, max_sigma, samples)
    Y = generateData(max_mu, max_sigma, samples)

    z_matrix = zeros(samples, 2)
    length = samples
    for i in 1:length
        z_matrix[i] = X[i]
    end
    
    for i in 1:length
        z_matrix[i, 2] = Y[i]
    end

    #return matrix 
    return z_matrix
end 

#calculates expectation of z = sin(x) * cos(y)
function calculateExpectation(data) 
    expectation = 0 
    for i in (1:size(data, 1))
        expectation += sin(data[i,1])*cos(data[i,2])
    end 
    expectation = expectation/size(data, 1)
    
    return expectation
end

#returns z = sin(x)cos(y) in matrix form 
function calculateZvector(x_vect, y_vect) 
    z_vector = Array{Float64}(undef, size(x_vect, 1))
    for i in 1:size(z_vector, 1)
        z_vector[i] = sin(x_vect[i])*cos(y_vect[i])
    end 
    return z_vector 
end


function coefficientsVector(data, number_pseudo) 
    Values_Array = Array{Float64}(undef, size(data, 1), binomial(number_pseudo+2, number_pseudo))
    x_vect = LinRange(-10, 10, 1000)
    y_vect = LinRange(-10, 10, 1000)
    for i in (1:size(x_vect, 1))
        monomial_array = calculateMonomial(x_vect[i], y_vect[i], number_pseudo)
        for j in (1:length(monomial_array))
            Values_Array[i, j] = monomial_array[j]
        end 
    end 

    z_vector = calculateZvector(x_vect, y_vect)
    coefficients_vector = Values_Array \ z_vector

    return coefficients_vector
end 



function calculateMonomial(x_value, y_value, order) 
    #creating two variables x and y 
    @polyvar x y 
    xy_monomial = monomials([x, y], 0:order)
    values = subs(xy_monomial, x => x_value, y => y_value)
    return values 
end


function findStoredMoments(number_stored, data) 
     #create moments array 
     array_of_moments = [moment(data, 1), moment(data, 2)]

     #add moments to array up to order number_stored
     for i in 3:number_stored
         push!(array_of_moments, moment(data, i))
     end 

     return array_of_moments
end 

function findPseudoMoments(number_stored, number_pseudo, data)
    #create moments array 
    array_of_moments = [moment(data, 1), moment(data, 2)]

    #add moments to array up to order number_stored
    for i in 3:number_stored
        push!(array_of_moments, moment(data, i))
    end 

    #converts moments to cumulants 
    moms2cums!(array_of_moments)

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
            push!(array_of_moments, cumulant)
        end   
    
        #converts cumulants back to moments and returns array of moments, now containing pseudo moments
        return cums2moms(array_of_moments) 
end 

function chooseMoments(pseudo_moments_array)
    moments_vector = Array{Float64}(undef, 0)
    current_index = length(pseudo_moments_array)
    while(current_index > 0)
        indices = ones(Int64, current_index)
        chooseMomentsHelper(indices, moments_vector, pseudo_moments_array)
        current_index -= 1
    end 
    push!(moments_vector, 0)
    return moments_vector
end 

function chooseMomentsHelper(indices, moments_vector, pseudo_moments_array)
    size = length(indices) + 1
    for i in 1:size
        push!(moments_vector,pseudo_moments_array[length(indices)][indices...])
        if (i < size)
            indices[i] = 2
        end 
    end
end 

function calculateApproximation(coefficients, moments_vector)
    return dot(moments_vector, coefficients)
end 


function stored_approx(std, mu, samples, number_stored, loops)
    error = 0
    for i in 1:loops 
        data = generateMultivariate(mu, std, samples)
        true_expectation = calculateExpectation(data)
        stored_moments = findStoredMoments(number_stored, data)
        stored_vector = chooseMoments(stored_moments)
        stored_coefficient_matrix = coefficientsMatrix(data, number_stored)
        stored_expectation = calculateApproximation(stored_coefficient_matrix, stored_vector)
        error += abs(true_expectation - stored_expectation)
    end 
    error /= loops
    return error   
end 

function pseudo_approx(std, mu, number_stored, number_pseudo, samples, loops)
    error = 0
    for i in 1:loops 
        data = generateMultivariate(mu, std, samples)
        true_expectation = calculateExpectation(data)
        pseudo_moments = findPseudoMoments(number_stored, number_pseudo, data)
        pseudo_vector = chooseMoments(pseudo_moments)
        pseudo_coefficient_matrix= coefficientsMatrix(data, number_pseudo)
        pseudo_approximation = calculateApproximation(pseudo_coefficient_matrix, pseudo_vector)
        error += abs(true_expectation - pseudo_approximation)
    end 
    error /= loops
    return error  

end 

function main(std, mu, number_stored, samples, number_pseudo, loops)
    error_stored = stored_approx(std, mu, samples, number_stored, loops)
    error_pseudo = pseudo_approx(std, mu, number_stored, number_pseudo, samples, loops)
    return_array = [error_stored, error_pseudo]
    return return_array
end












