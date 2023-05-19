using Combinatorics

array = [0, 1, 2, 3]
println(multiset_permutations(a, 3))

coefficients_vector_eigth_order = compute_coefficients_vector(8, 1000, -1.5, 1.5)

coefficients_vector_fourth_order = compute_coefficients_vector(4, 1000, -1.5, 1.5)
coefficients_vector_third_order = compute_coefficients_vector(3, 1000, -1.5, 1.5)
coefficients_vector_second = compute_coefficients_vector(2, 1000, -1.5, 1.5)

error_second_order = 0 
error_third_order = 0 
error_fourth_order = 0
error_eigth_order = 0


for i in 1:1 
    data = generate_multivariate_data(1000000)
    true_expectation = calculate_expectation(data)


    expectation_fourth_order = test_stored_moments(coefficients_vector_fourth_order, 4, data)
    expectation_third_order = test_stored_moments(coefficients_vector_third_order, 3, data)
    expectation_second_order = test_stored_moments(coefficients_vector_second, 2, data)
    expectation_eigth_order = test_stored_moments(coefficients_vector_eigth_order, 8, data)


    error_second_order += abs(expectation_second_order - true_expectation)
    error_third_order += abs(expectation_third_order - true_expectation)
    error_fourth_order += abs(expectation_fourth_order - true_expectation)
    error_eigth_order += abs(expectation_eigth_order - true_expectation)
end 

println(error_second_order)
println(error_third_order)
println(error_fourth_order)
println(error_eigth_order)

data = generate_multivariate_data(10000)
error2 = least_squares_error(2, data)
error3 = least_squares_error(3, data)
error4 = least_squares_error(4, data)
error5 = least_squares_error(5, data)