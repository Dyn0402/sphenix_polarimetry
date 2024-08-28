#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

double calculate_sigma(double sigma, double z, double beta_star) {
    if (beta_star != 0) {
        return sigma * std::sqrt(1 + (z * z) / (beta_star * 1e4 * beta_star * 1e4));
    }
    return sigma;
}

py::array_t<double> density(
    py::array_t<double> x, py::array_t<double> y, py::array_t<double> z,
    double r_x, double r_y, double r_z, double sigma_x, double sigma_y, double sigma_z,
    double angle, double beta_star
) {
    // Get the buffer info for the inputs
    auto buf_x = x.unchecked<3>();
    auto buf_y = y.unchecked<3>();
    auto buf_z = z.unchecked<3>();

    // Create an array to store the result, using the same shape as the input arrays
    std::vector<ssize_t> shape = { buf_x.shape(0), buf_x.shape(1), buf_x.shape(2) };
    py::array_t<double> result(shape);
    auto buf_result = result.mutable_unchecked<3>();

    // Precompute the cosine and sine of the rotation angle
    double cos_angle = std::cos(angle);
    double sin_angle = std::sin(angle);

    // Iterate over the grid points
    for (ssize_t i = 0; i < buf_x.shape(0); i++) {
        for (ssize_t j = 0; j < buf_x.shape(1); j++) {
            for (ssize_t k = 0; k < buf_x.shape(2); k++) {
                double z_rel = buf_z(i, j, k) - r_z;

                // Compute the modified sigmas
                double sigma_x_mod = calculate_sigma(sigma_x, buf_z(i, j, k), beta_star);
                double sigma_y_mod = calculate_sigma(sigma_y, buf_z(i, j, k), beta_star);

                // Rotate the coordinates
                double y_rot = (buf_y(i, j, k) - r_y) * cos_angle - z_rel * sin_angle;
                double z_rot = (buf_y(i, j, k) - r_y) * sin_angle + z_rel * cos_angle;

                // Calculate the relative x coordinate
                double x_rel = buf_x(i, j, k) - r_x;

                // Calculate the exponent for the Gaussian distribution
                double exponent = -0.5 * (x_rel * x_rel / (sigma_x_mod * sigma_x_mod) +
                                          y_rot * y_rot / (sigma_y_mod * sigma_y_mod) +
                                          z_rot * z_rot / (sigma_z * sigma_z));

                // Compute the density and store it in the result array
                buf_result(i, j, k) = std::exp(exponent) / (std::pow(2 * M_PI, 1.5) * sigma_x_mod * sigma_y_mod * sigma_z);
            }
        }
    }

    return result;
}

PYBIND11_MODULE(bunch_density_cpp, m) {
    m.def("density", &density, "Calculate the density of the bunch at given points in the lab frame");
}
