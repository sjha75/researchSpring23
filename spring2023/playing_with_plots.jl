using GLMakie
include("multivariateFeb20.jl")
using Colors




function plotOriginalData(data, fig, axis) 
    x_vect = data[:,1]
    y_vect = data[:,2]
    z_vect = calculateZvector1(x_vect, y_vect) 
    #=aspect=(1, 1, 1)
    perspectiveness=0.5
    fig = Figure(; resolution=(1200, 400))=#
    scatter!(axis, x_vect, y_vect, z_vect; markersize=15)
    fig
end 

function plotLeastSquaresRegression(order, fig, axis)
    @polyvar x y 
    xy_monomial = monomials([x,y], 0:order)
    coefficients_vector = coefficientsVector1(100000, order, -1.5, 1.5)
    leastSquaresPolynomial = mapreduce(*, +, coefficients_vector, xy_monomial) 
    xs = LinRange(-10, 10, 1000)
    ys = LinRange(-10, 10, 1000)
    zs = [leastSquaresPolynomial(x,y) for x in xs, y in ys]
    surface!(axis, xs, ys, zs, color=fill(RGBA(0.,0.,1.,0.5),100,100))
    fig
end 

function plotOriginalFunction(fig, axis)
    @polyvar x y 
    xy_monomial = monomials([x,y], 0:3)
    coefficients_vector = [1, 4, 3, 4, 2, 1, 5, 3, 4, 2]
    original_monomial = mapreduce(*, +, coefficients_vector, xy_monomial) 
    xs = LinRange(-5, 5, 1000)
    ys = LinRange(-5, 5, 1000)
    zs = [original_monomial(x,y) for x in xs, y in ys]
    surface!(axis, xs, ys, zs, color=fill(RGBA(1.,0.,0.,0.5),100,100))
    fig
end 


aspect=(1, 1, 1)
perspectiveness=0.5
fig = Figure(; resolution=(1200, 400))
ax1 = Axis3(fig[1, 1]; aspect, perspectiveness)
ax2 = Axis3(fig[1, 2]; aspect, perspectiveness)
ax3 = Axis3(fig[2, 1]; aspect, perspectiveness)
ax4 = Axis3(fig[2, 2]; aspect, perspectiveness)
data1 = generateMultivariate1(1, 1, 100000)



plotLeastSquaresRegression(2, fig, ax1)
plotLeastSquaresRegression(3, fig, ax2)
plotLeastSquaresRegression(4, fig, ax3)
plotLeastSquaresRegression(5, fig, ax4)

plotOriginalData(data1, fig, ax1)
plotOriginalData(data1, fig, ax2)
plotOriginalData(data1, fig, ax3)
plotOriginalData(data1, fig, ax4)

plotOriginalFunction(fig, ax1)
plotOriginalFunction(fig, ax2)
plotOriginalFunction(fig, ax3)
plotOriginalFunction(fig, ax4)






