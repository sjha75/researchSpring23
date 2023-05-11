using GLMakie
using Colors
include("multivariateMarch10.jl")
include("multivariateFeb20.jl")


function plotOriginalData(data, fig, axis) 
    x_vect = data[:,1]
    y_vect = data[:,2]
    z_vect = calculateZvector(x_vect, y_vect) 
    scatter!(axis, x_vect, y_vect, z_vect; markersize=15, color = :pink)
    fig
end 

aspect=(1, 1, 1)
perspectiveness=0.5
fig = Figure(; resolution=(1200, 400))
ax1 = Axis3(fig[1, 1]; aspect, perspectiveness)
ax2 = Axis3(fig[1, 2]; aspect, perspectiveness)
ax3 = Axis3(fig[2, 1]; aspect, perspectiveness)
ax4 = Axis3(fig[2, 2]; aspect, perspectiveness)

xs = LinRange(-1.5, 1.5, 100)
ys = LinRange(-1.5, 1.5, 100)
data = generateMultivariate(1, 1, 100)
#data2 = generateMultivariate(1, 1, 100)
#data3 = generateMultivariate(1, 1, 100)
#data4 = generateMultivariate(1, 1, 100)

coefficients_vector_samples3 = coefficientsVector(100, 3, data)
coefficients_vector_samples2 = coefficientsVector(100, 2, data)
coefficients_vector_linRange3 = coefficientsVector1(100, 3, -0.5, 1.5)
coefficients_vector_linRange2 = coefficientsVector1(100,2, -0.5, 1.5)
coefficients_vector_linRange4 = coefficientsVector1(100, 4, -0.5, 1.5)
coefficients_vector_samples4 = coefficientsVector(100, 4, data)

order4linRange = [coefficients_vector_linRange4[1]x^4 + coefficients_vector_linRange4[2]x^3*y + coefficients_vector_linRange4[3]x^2*y^2 + coefficients_vector_linRange4[4]x*y^3 + coefficients_vector_linRange4[5]y^4 + coefficients_vector_linRange4[6]x^3 + coefficients_vector_linRange4[7]x^2*y + coefficients_vector_linRange4[8]x*y^2 + coefficients_vector_linRange4[9]y^3 + coefficients_vector_linRange4[10]x^2 + coefficients_vector_linRange4[11]x*y + coefficients_vector_linRange4[12]y^2 + coefficients_vector_linRange4[13]x + coefficients_vector_linRange4[14]y + coefficients_vector_linRange4[15] for x in xs, y in ys]
order4samples = [coefficients_vector_samples4[1]x^4 + coefficients_vector_samples4[2]x^3*y + coefficients_vector_samples4[3]x^2*y^2 + coefficients_vector_samples4[4]x*y^3 + coefficients_vector_samples4[5]y^4 + coefficients_vector_samples4[6]x^3 + coefficients_vector_samples4[7]x^2*y + coefficients_vector_samples4[8]x*y^2 + coefficients_vector_samples4[9]y^3 + coefficients_vector_samples4[10]x^2 + coefficients_vector_samples4[11]x*y + coefficients_vector_samples4[12]y^2 + coefficients_vector_samples4[13]x + coefficients_vector_samples4[14]y + coefficients_vector_samples4[15] for x in xs, y in ys]
order2linRange = [coefficients_vector_linRange2[1]x^2 + coefficients_vector_linRange2[2]x*y + coefficients_vector_linRange3[3]y^2 + coefficients_vector_linRange3[4]x + coefficients_vector_linRange3[5]y + coefficients_vector_linRange3[6] for x in xs, y in ys]
order2samples = [coefficients_vector_samples2[1]x^2 + coefficients_vector_samples2[2]x*y + coefficients_vector_samples2[3]y^2 + coefficients_vector_samples2[4]x + coefficients_vector_samples2[5]y + coefficients_vector_samples2[6] for x in xs, y in ys]

surface!(ax1, xs, ys, order4linRange, color=fill(RGBA(0.,0.,1.,0.5),100,100))
fig 
surface!(ax2, xs, ys, order4linRange, color=fill(RGBA(0.,0.,1.,0.5),100,100))
fig 
surface!(ax3, xs, ys, order4linRange, color=fill(RGBA(0.,0.,1.,0.5),100,100))
fig 
surface!(ax4, xs, ys, order4linRange, color=fill(RGBA(0.,0.,1.,0.5),100,100))
fig 

#=surface!(ax1, xs, ys, order4samples, color=fill(RGBA(1.,0.,0.,0.5),100,100))
fig=#

#=surface!(ax1, xs, ys, order2linRange, color=fill(RGBA(0.,0.,1.,0.5),100,100))
fig 

surface!(ax1, xs, ys, order2samples, color=fill(RGBA(1.,0.,0.,0.5),100,100))
fig=#

#=z1 = [coefficients_vector_samples3[1]x^3 + coefficients_vector_samples3[2]x^2*y + coefficients_vector_samples3[3]x*y^2 + coefficients_vector_samples3[4]y^3 + coefficients_vector_samples3[5]x^2 + coefficients_vector_samples3[6]x*y + coefficients_vector_samples3[7]y^2 + coefficients_vector_samples3[8]x + coefficients_vector_samples3[9]y + coefficients_vector_samples3[10] for x in xs, y in ys]
surface!(ax1, xs, ys, z1,  color=fill(RGBA(1.,0.,0.,0.5),100,100))
fig=#

#=z3 = [coefficients_vector_linRange3[1]x^3 + coefficients_vector_linRange3[2]x^2*y + coefficients_vector_linRange3[3]x*y^2 + coefficients_vector_linRange3[4]y^3 + coefficients_vector_linRange3[5]x^2 + coefficients_vector_linRange3[6]x*y + coefficients_vector_linRange3[7]y^2 + coefficients_vector_linRange3[8]x + coefficients_vector_linRange3[9]y + coefficients_vector_linRange3[10] for x in xs, y in ys]
surface!(ax1, xs, ys, z3, color=fill(RGBA(0.,0.,1.,0.5),100,100))
fig =#

#=z2 = [sin(x) * cos(y) for x in xs, y in ys]
surface!(ax1, xs, ys, z2, color=fill(RGBA(1.,1.,0.,0.5),100,100))
fig=#

plotOriginalData(data, fig, ax1)
plotOriginalData(data2, fig, ax2)
plotOriginalData(data3, fig, ax3)
plotOriginalData(data4, fig, ax4)


