using Distributions
using StatsBase
import Statistics as stats
using DataFrames
using CSV
using LinearAlgebra
using Expectations
using Cumulants
using Plots
using Combinatorics
using SymmetricTensors
using Polynomials
using DynamicPolynomials
using CumulantsUpdates


#Least Squares regression uses samples from multi-variate data itself

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


function coefficientsVector(samples, number_pseudo, data) 
    Values_Array = Array{Float64}(undef, samples, binomial(number_pseudo+2, number_pseudo))
    x_vect = data[:, 1]
    y_vect = data[:, 2]
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


#returns z = sin(x)cos(y) in matrix form 
function calculateZvector(x_vect, y_vect) 
    z_vector = Array{Float64}(undef, size(x_vect, 1))
    for i in 1:size(z_vector, 1)
        z_vector[i] = sin(x_vect[i])*cos(y_vect[i])
    end 
    return z_vector 
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

function calculateApproxForStored(number_stored, coefficient_vector, data_array, amount)
    error = 0 
    
    for i in 1:amount 
        true_expectation = calculateExpectation(data_array[i])
        stored_moments = findStoredMoments(number_stored, data_array[i])
        stored_moments_vector = chooseMoments(stored_moments)
        stored_expectation = calculateApproximation(coefficient_vector, stored_moments_vector)
        error += abs(true_expectation - stored_expectation)
    end 
    error /= amount
  
    return error
end

function calculateApproxForPseudo(number_stored, number_pseudo, coefficient_vector, data_array, amount)
    error = 0
    for i in 1:amount 
        true_expectation = calculateExpectation(data_array[i])
        pseudo_moments = findPseudoMoments(number_stored, number_pseudo, data_array[i])
        pseudo_vector = chooseMoments(pseudo_moments)
        pseudo_approximation = calculateApproximation(coefficient_vector, pseudo_vector)
        error += abs(true_expectation - pseudo_approximation)
    end 
    error /= amount
    return error  
end 

function calculateData(amount, max_mu, max_std, samples)
    data1 = generateMultivariate(max_mu, max_std, samples)
    data2 = generateMultivariate(max_mu, max_std, samples)
    data_array = [data1, data2]
    for i in 3:amount 
        data_new = generateMultivariate(max_mu, max_std, samples)
        push!(data_array, data_new)
    end 

    return data_array
end 



function main() 
   
    error_stored2 = 0
    error_pseudo6 = 0 
    error_pseudo7 = 0 
    error_stored3 = 0 
    error_pseudo6_stored3 = 0 
    error_pseudo7_stored3 = 0 

    for i in 1:100 
        data = generateMultivariate(1, 1, 1000)
        true_expectation = calculateExpectation(data)
        coefficients_vector_order2 = coefficientsVector(1000, 2, data)
        coefficients_vector_order3 = coefficientsVector(1000, 3, data)
        coefficients_vector_order6 = coefficientsVector(1000, 6, data)
        coefficients_vector_order7 = coefficientsVector(1000, 7, data)

        stored_moments_order2 = findStoredMoments(2, data)
        moments_order2 = chooseMoments(stored_moments_order2)
        stored_moments_order3 = findStoredMoments(3, data)
        moments_order3 = chooseMoments(stored_moments_order3)
        pseudo_moments_order6_stored2 = findPseudoMoments(2, 6, data)
        pseudo6 = chooseMoments(pseudo_moments_order6_stored2)
        pseudo_moments_order7_stored2 = findPseudoMoments(2, 7, data)
        pseudo7 = chooseMoments(pseudo_moments_order7_stored2)
        pseudo_moments_order6_stored3 = findPseudoMoments(3, 6, data)
        pseudo6_stored3 = chooseMoments(pseudo_moments_order6_stored3)
        pseudo_moments_order7_stored3 = findPseudoMoments(3, 7, data)
        pseudo7_stored3 = chooseMoments( pseudo_moments_order7_stored3)

        stored2approx = calculateApproximation(coefficients_vector_order2, moments_order2)
        stored3approx = calculateApproximation(coefficients_vector_order3, moments_order3)
        pseudo6approx = calculateApproximation(coefficients_vector_order6, pseudo6)
        pseudo7approx = calculateApproximation(coefficients_vector_order7, pseudo7)
        pseudo6stored3approx = calculateApproximation(coefficients_vector_order6, pseudo6_stored3)
        pseudo7stored3approx = calculateApproximation(coefficients_vector_order7, pseudo7_stored3)

        error_stored2 += abs(true_expectation - stored2approx)
        error_pseudo6 += abs(true_expectation - pseudo6approx)
        error_pseudo7 += abs(true_expectation - pseudo7approx)
        error_stored3 += abs(true_expectation - stored3approx)
        error_pseudo6_stored3 += abs(true_expectation - pseudo6stored3approx)
        error_pseudo7_stored3 += abs(true_expectation - pseudo7stored3approx)

    end

    

    println(error_stored2/100)
    println(error_pseudo6/100)
    println(error_pseudo7/100)
   
    println()
    println()

    println(error_stored3/100)
    println(error_pseudo6_stored3/100)
    println(error_pseudo7_stored3/100)



end 

data = generateMultivariate(1, 1, 1000)
coefficients_vector_order2 = coefficientsVector(1000, 2, data)
print(coefficients_vector_order2)
