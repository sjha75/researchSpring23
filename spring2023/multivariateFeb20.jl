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
using Random


#sin(x)cos(y) 
#x^3 + 4x^2y + 3xy^2 + 4y^3 + 2x^2 + xy + 5y^2 + 3x + 4y + 2 
#Least Squares regression uses Linear Range for x and y, between -4 and 4 
#chooses random samples from each of the normal distributions and multiplies them together to get 
function generateData1(max_mu, max_sigma, samples) 
    #randomly selects the means and standard deviations
    mu1 = 0
    mu2 = 0
    sigma1 = 1
    sigma2 = 1

#generate normal distributions using randomly chosen mean and standard deviation 
    X = Normal(mu1, sigma1)
    Y = Normal(mu2, sigma2)
    x_samples = rand(X, samples)
    y_samples = rand(Y, samples)
    z = x_samples .* y_samples
    return z 
end

#places two random variables 
function generateMultivariate1(max_mu, max_sigma, samples) 
    #generate two non-normal random variables 
    X = generateData1(max_mu, max_sigma, samples)
    Y = generateData1(max_mu, max_sigma, samples)

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
function calculateExpectation1(data) 
    expectation = 0 
    #=@polyvar x y 
    xy_monomial = monomials([x, y], 0:3)
    coefficients_vector = [1, 4, 3, 4, 2, 1, 5, 3, 4, 2] 
    third_order_polynomial = mapreduce(*, +, coefficients_vector, xy_monomial) =#

    for i in (1:size(data, 1))
        expectation += sin(data[i,1])*cos(data[i,2])
        #expectation += third_order_polynomial(data[i, 1], data[i, 2])
    end 
    expectation = expectation/size(data, 1)
    
    return expectation
end


function coefficientsVector1(samples, order, left, right) 
    Values_Array = Array{Float64}(undef, samples*samples, binomial(order+2, order))
    x_vect = LinRange(left, right, samples)
    y_vect = LinRange(left, right, samples)
    current_row = 0
    for i in (1:size(x_vect, 1))
        for j in (1:size(y_vect, 1))
            current_row += 1
            monomial_array = calculateMonomial1(x_vect[i], y_vect[j], order)
            for k in (1:length(monomial_array))
                Values_Array[current_row, k] = monomial_array[k]
            end 
        end 
    end 
    #=for i in (1:size(x_vect, 1))
        monomial_array = calculateMonomial1(x_vect[i], y_vect[i], order)
        for j in (1:length(monomial_array))
            Values_Array[i, j] = monomial_array[j]
        end 
    end =#

    z_vector = calculateZvector1(x_vect, y_vect)

    coefficients_vector = Values_Array \ z_vector
   #println(coefficients_vector)
   

    return coefficients_vector
end 

function calculateMonomial1(x_value, y_value, order) 
    #creating two variables x and y 
    @polyvar x y 
    xy_monomial = monomials([x, y], 0:order)
    values = subs(xy_monomial, x => x_value, y => y_value)
    return values 
end


#returns z = sin(x)cos(y) in matrix form 
function calculateZvector1(x_vect, y_vect) 
    #=@polyvar x y 
    xy_monomial = monomials([x, y], 0:3)
    coefficients_vector = [1, 4, 3, 4, 2, 1, 5, 3, 4, 2]
    third_order_polynomial = mapreduce(*, +, coefficients_vector, xy_monomial) =#
    z_vector = Array{Float64}(undef, size(x_vect, 1) * size(x_vect, 1))
    current_row = 0
    for i in 1:size(x_vect, 1)
        for j in 1:size(y_vect, 1)
            current_row +=1
            z_vector[current_row] = sin(x_vect[i])*cos(y_vect[j])
        end 
    end 

   #= z_vector = Array{Float64}(undef, size(x_vect, 1))
    for i in 1:size(z_vector, 1)
        #z_vector[i] = sin(x_vect[i])*cos(y_vect[i])
        z_vector[i] = third_order_polynomial(x_vect[i], y_vect[i])
    end  =#
    return z_vector 
end


function findStoredMoments1(number_stored, data) 
    #create moments array 
    array_of_moments = [moment(data, 1), moment(data, 2)]

    #add moments to array up to order number_stored
    for i in 3:number_stored
        push!(array_of_moments, moment(data, i))
    end 

    return array_of_moments
end 


function findPseudoMoments1(number_stored, number_pseudo, data)
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

function chooseMoments1(pseudo_moments_array)
    moments_vector = Array{Float64}(undef, 0)
    current_index = length(pseudo_moments_array)
    while(current_index > 0)
        indices = ones(Int64, current_index)
        chooseMomentsHelper1(indices, moments_vector, pseudo_moments_array)
        current_index -= 1
    end 
    push!(moments_vector, 1)
    return moments_vector
end 

function chooseMomentsHelper1(indices, moments_vector, pseudo_moments_array)
    size = length(indices) + 1
    for i in 1:size
        push!(moments_vector,pseudo_moments_array[length(indices)][indices...])
        if (i < size)
            indices[i] = 2
        end 
    end
end 

function calculateApproximation1(coefficients, moments_vector)
    return dot(moments_vector, coefficients)
end 

function calculateApproxForStored1(number_stored, coefficient_vector, data_array, amount)
    error = 0 
    
    for i in 1:amount 
        true_expectation = calculateExpectation1(data_array[i])
        stored_moments = findStoredMoments1(number_stored, data_array[i])
        stored_moments_vector = chooseMoments1(stored_moments)
        stored_expectation = calculateApproximation1(coefficient_vector, stored_moments_vector)
        error += abs(true_expectation - stored_expectation)
    end 
    error /= amount
  
    return error
end

function calculateApproxForPseudo1(number_stored, number_pseudo, coefficient_vector, data_array, amount)
    error = 0
    for i in 1:amount 
        true_expectation = calculateExpectation1(data_array[i])
        pseudo_moments = findPseudoMoments1(number_stored, number_pseudo, data_array[i])
        pseudo_vector = chooseMoments1(pseudo_moments)
        pseudo_approximation = calculateApproximation1(coefficient_vector, pseudo_vector)
        error += abs(true_expectation - pseudo_approximation)
    end 
    error /= amount
    return error  
end 

function calculateData1(amount, max_mu, max_std, samples)
    data1 = generateMultivariate1(max_mu, max_std, samples)
    data2 = generateMultivariate1(max_mu, max_std, samples)
    data_array = [data1, data2]
    for i in 3:amount 
        data_new = generateMultivariate1(max_mu, max_std, samples)
        push!(data_array, data_new)
    end 

    return data_array
end 

function checkRegression(order,data, samples)
    @polyvar x y 
    xy_monomial = monomials([x,y], 0:order) 
    coefficients_vector = coefficientsVector1(1000, order, -1.5, 1.5)
    regressionPolynomial = mapreduce(*, +, coefficients_vector, xy_monomial) 
    


    x_vector = data[:,1]
    y_vector = data[:,2]
    error = 0
    for i in 1:samples
        temp = regressionPolynomial(x_vector[i], y_vector[i]) - sin(x_vector[i])*cos(y_vector[i])
        error += abs(temp)
    end 
    error /= samples 
    return error
end 




function main1(amount, max_mu, max_std, samples) 
    coefficient_vector_order2 = coefficientsVector1(1000, 2, -1.5, 1.5)
    coefficient_vector_order3 = coefficientsVector1(1000, 3, -1.5, 1.5)
    coefficient_vector_order4 = coefficientsVector1(1000, 4, -1.5, 1.5)
    coefficient_vector_order5 = coefficientsVector1(1000, 5, -1.5, 1.5)
    coefficient_vector_order6 = coefficientsVector1(1000, 6, -1.5, 1.5)
    coefficient_vector_order7 = coefficientsVector1(1000, 7, -1.5, 1.5)


    
   
    data_array = calculateData1(amount, max_mu, max_std, samples)

    println(calculateApproxForStored1(2, coefficient_vector_order2, data_array, amount))
    println(calculateApproxForStored1(3, coefficient_vector_order3, data_array, amount))
    println(calculateApproxForStored1(4, coefficient_vector_order4, data_array, amount))
    println(calculateApproxForStored1(5, coefficient_vector_order5, data_array, amount))
    println(calculateApproxForStored1(6, coefficient_vector_order6, data_array, amount))
    println(calculateApproxForStored1(7, coefficient_vector_order7, data_array, amount))
    println()
    println(calculateApproxForPseudo1(2, 6, coefficient_vector_order6, data_array, amount))
    println(calculateApproxForPseudo1(2, 7, coefficient_vector_order7, data_array, amount))
    println(calculateApproxForPseudo1(3, 6, coefficient_vector_order6, data_array, amount))
    println(calculateApproxForPseudo1(3, 7, coefficient_vector_order7, data_array, amount))

    

end 


main1(20, 1, 1, 1000)













