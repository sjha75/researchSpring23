using Plots 
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

function plotSinCos() 
    x = range(-4, 4, length = 100)
    y = range(-4, 4, length = 100)
    f(x,y) = sin(x) * cos(y)
    plot(x, y, f, st=:surface, camera=(-30, 30))
    savefig("plot.pdf")
end 

function plotLeastSquaresRegressionOrder2(coefficients_vector) 
    x = range(-4, 4, length = 100)
    y = range(-4, 4, length = 100)
    f(x, y) = coefficients_vector[1]x^2 * coefficients_vector[2]x*y + coefficients_vector[3]y^2 + coefficients_vector[4]x + coefficients_vector[5]y + coefficients_vector[6]
    plot(x, y, f, st=:surface, camera=(-30, 30))
    savefig("plot4.pdf")

    
    
end

coefficients_vector_datasamples = [-0.2323683139819531, -0.43872532104385137, -0.2077727279693869, 1.1892908699873561, 0.15993982592472095, -0.015849895532277412]
coefficients_vector_linsamples = [3.1727201931449345e-19, 7.714220061359648e-19, 7.714220061359648e-19, 0.006474889866210256, 0.006474889866210256, -1.0349694810292345e-17]
plotLeastSquaresRegressionOrder2(coefficients_vector_linsamples)

