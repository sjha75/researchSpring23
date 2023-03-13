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
using Tensors
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

#calculatues the expectation of sin(Z), where Z = XY
function expectation(data)
    total_sum = 0
    for i in data
        total_sum += sin(i)
    end 
    expectation = total_sum/length(data)
    return expectation
end 

#stores actual moments of data in array and returns array 
#I don't end up using this function
function calculateMoments(data, number_of_moments) 
    array_of_moments = [moment(data, 1), moment(data, 2)]
    for i in 3:number_of_moments 
        push!(array_of_moments, moment(data, i))
    end 
    return array_of_moments
end 

#calculates a certain order moment of given data 
#I don't end up using this function 
function calculateMomentOrder(data, order) 
    moment = 0
    for x in data 
        moment += x^order 
    end 
    moment = moment/length(data)
    return moment 
end 


#calculates cumulants from given moments
#I don't end up using this function
function moms2cumsUnivariate(moments_array) 
    cumulant_array = zeros(length(moments_array))
    if (length(moments_array) >= 1) 
        cumulant_array[1] = moments_array[1]
    end
    if (length(moments_array) >= 2) 
        cumulant_array[2] = moments_array[2] - (moments_array[1])^2
    end
    if (length(moments_array) >= 3) 
        cumulant_array[3] = moments_array[3]- 3*moments_array[2]*moments_array[1] + 2*((moments_array[1])^3)
    end
    if (length(moments_array) >= 4) 
        cumulant_array[4] = moments_array[4]-4*moments_array[3]*moments_array[1] -3*(moments_array[2]^2) + 12*moments_array[2]*(moments_array[1])^2 -6*(moments_array[1]^4)
    end
    if (length(moments_array) >= 5) 
        cumulant_array[5] = moments_array[5] - 5*moments_array[4]*moments_array[1] - 10*moments_array[3]*moments_array[2] + 20*moments_array[3]*(moments_array[1])^2 + 30*(moments_array[2]^2)*moments_array[1] - 60*moments_array[2]*(moments_array[1]^3) + 24*(moments_array[1]^5)
    end
    if (length(moments_array) == 6) 
        cumulant_array[6] = moments_array[6] - 6*moments_array[5]*moments_array[1] - 15*moments_array[4]*moments_array[2] + 80*moments_array[4]*(moments_array[1]^2) - 10*(moments_array[3]^2) + 120*moments_array[3]*moments_array[2]*moments_array[1] - 120*moments_array[3]*(moments_array[1]^3) + 30*(moments_array[2]^3) - 270*(moments_array[2]^2)*(moments_array[1]^2) + 360*moments_array[2]*(moments_array[1]^4) - 120*(moments_array[1]^6)
    end

    return cumulant_array 
end 

#calculates moments given array of cumulants
#I don't end up using this function 
function cums2momsUnivariate(cumulants_array)
    moments_array = zeros(length(cumulants_array))
    if (length(moments_array) >= 1) 
        moments_array[1] = cumulants_array[1]
    end
    if (length(moments_array) >= 2) 
       moments_array[2] = cumulants_array[2] + (cumulants_array[1]^2)
    end
    if (length(moments_array) >= 3) 
        moments_array[3] = cumulants_array[3] + 3*cumulants_array[2]*cumulants_array[1] + cumulants_array[1]^3
    end
    if (length(moments_array) >= 4) 
        moments_array[4] = cumulants_array[4] + 4*cumulants_array[3]*cumulants_array[1] + 3*(cumulants_array[2]^2) + 6*cumulants_array[2]*(cumulants_array[1]^2) + (cumulants_array[1]^4)
    end
    if (length(moments_array) >= 5) 
        moments_array[5] = cumulants_array[5] + 5*cumulants_array[4]*cumulants_array[1] + 10*cumulants_array[3]*cumulants_array[2] + 10*cumulants_array[3]*(cumulants_array[1]^2) + 15*(cumulants_array[2]^2)*cumulants_array[1] + 10*cumulants_array[2]*(cumulants_array[1]^3) + (cumulants_array[1]^5)
    end
    if (length(moments_array) == 6) 
        moments_array[6] = cumulants_array[6] + 6*cumulants_array[5]*cumulants_array[1] + 15*cumulants_array[4]*cumulants_array[2] + 15*cumulants_array[4]*(cumulants_array[1]^2) + 10*(cumulants_array[3]^2) + 60*cumulants_array[3]*cumulants_array[2]*cumulants_array[1] + 20*cumulants_array[3]*(cumulants_array[1]^3) + 15*(cumulants_array[2]^3) + 45*(cumulants_array[2]^2)*(cumulants_array[1]^2) + 15*cumulants_array[2]*(cumulants_array[1]^4) + (cumulants_array[1]^6)
    end

    return moments_array

end

#calculates the pseudo moments up to order specified 
function calculatePseudoMoments(data, number_moments, number_pseudo_moments) 
    moments_array = []
    for i in 1:number_moments
        push!(moments_array, moment(data, i))
    end 


end



#find polynomial approximation of data 
function polynomialApprox(data, degree)
    xs = convert(AbstractVector, @.sin(data))
    fit = Polynomials.fit(data, xs, degree)
    return fit

end

#finds expectation based on polynomial fit and moments 
function approxExpectation(moments_array, fit, num_moments) 
    total_sum = 0
    for i in 1:num_moments
        total_sum += moments_array[i] * fit[i]
    end 
    return total_sum
end


#=function that puts everything together 
  num_moments is  the number of actual moments to calculate of generated data 
  num_pseudo_moments is the number of pseudo moments to use 
  num_tests is the number of times to generate data and calculate approximations 
  
  this function returns an array with two elements. The first input 
  is the average difference between the approximation using actual moments and the 
  true expectation. The second input is the average difference between the approximation
  using pseudo moments and the true expectation=#
function test(num_moments, num_pseudo_moments, samples, number_tests, max_mu, max_sigma)
    test_output = Matrix(undef, number_tests, 3)
    true_expectation = 0 
    moments_expectation = 0
    pseudo_moments_expectation = 0 
    average_difference_moments = 0 
    average_difference_pseudo_moments = 0
    for i in 1:number_tests 
        data_generated = generateData(max_mu, max_sigma, samples)
        true_expectation = expectation(data_generated)
        stored_moments_array = calculateMoments(data_generated, num_moments)
        pseudo_moments_array = calculatePseudoMoments(stored_moments_array, num_pseudo_moments)
        polynomial_fit1 = polynomialApprox(data_generated, num_moments)
        polynomial_fit2 = polynomialApprox(data_generated, num_pseudo_moments)
        moments_expectation = approxExpectation(stored_moments_array, polynomial_fit1, num_moments)
        pseudo_moments_expectation = approxExpectation(pseudo_moments_array, polynomial_fit2, num_pseudo_moments)
        average_difference_moments += abs(true_expectation - moments_expectation)
        average_difference_pseudo_moments += abs(true_expectation - pseudo_moments_expectation)
        test_output[i, 1] = true_expectation
        test_output[i, 2] = moments_expectation
        test_output[i, 3] = pseudo_moments_expectation
    end 
    
    return [average_difference_moments/number_tests, average_difference_pseudo_moments/number_tests]
end 

data = generateData(1, 1, 10)
println(data)
reshape(data, 2, 5)
println(data)



#=Some notes and observations: 
- moment() function doesn't seem to work properly on univariate data 
    - need to read up on the documentation 
    - problem fixed: input must be a matrix, and not an array
- moms2cums and cums2moms require a symmetric tensor as the input argument 
    - symmetric tensors must be of at least 2nd order 
    - once again running into problems with univariate distributions 
- comparing actual moments approx with pseudo moments approx 
    - when max_mu and max_sigma = 1, the pseudo moments approx is closer to true expectation 
        - 
=#










