#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

double calculate_sigma(double sigma, double z, double beta_star, double angle_x, double angle_y) {
    if (beta_star != 0) {
        double distance_to_IP = z * std::sqrt(1 + std::tan(angle_x) * std::tan(angle_x) + std::tan(angle_y) * std::tan(angle_y));
        return sigma * std::sqrt(1 + (distance_to_IP * distance_to_IP) / (beta_star * 1e4 * beta_star * 1e4));
    }
    return sigma;
}

py::array_t<double> density(
    py::array_t<double> x, py::array_t<double> y, py::array_t<double> z,
    double r_x, double r_y, double r_z, double sigma_x, double sigma_y, double sigma_z,
    double angle_x, double angle_y, double beta_star
) {
    // Get the buffer info for the inputs
    auto buf_x = x.unchecked<3>();
    auto buf_y = y.unchecked<3>();
    auto buf_z = z.unchecked<3>();

    // Create an array to store the result, using the same shape as the input arrays
    std::vector<ssize_t> shape = { buf_x.shape(0), buf_x.shape(1), buf_x.shape(2) };
    py::array_t<double> result(shape);
    auto buf_result = result.mutable_unchecked<3>();

    // Precompute the cosine and sine of the rotation angles
    double cos_angle_xz = std::cos(angle_x);
    double sin_angle_xz = std::sin(angle_x);
    double cos_angle_yz = std::cos(angle_y);
    double sin_angle_yz = std::sin(angle_y);

    // Iterate over the grid points
    for (ssize_t i = 0; i < buf_x.shape(0); i++) {
        for (ssize_t j = 0; j < buf_x.shape(1); j++) {
            for (ssize_t k = 0; k < buf_x.shape(2); k++) {
                // Calculate the relative position vector
                double x_rel = buf_x(i, j, k) - r_x;
                double y_rel = buf_y(i, j, k) - r_y;
                double z_rel = buf_z(i, j, k) - r_z;

                // Apply rotation for the xz plane (rotation around the y-axis)
                double x_rot = x_rel * cos_angle_xz - z_rel * sin_angle_xz;
                double z_rot_xz = x_rel * sin_angle_xz + z_rel * cos_angle_xz;

                // Apply rotation for the yz plane (rotation around the x-axis)
                double y_rot = y_rel * cos_angle_yz - z_rot_xz * sin_angle_yz;
                double z_rot_yz = y_rel * sin_angle_yz + z_rot_xz * cos_angle_yz;

                // Compute the modified sigmas
                double sigma_x_mod = calculate_sigma(sigma_x, buf_z(i, j, k), beta_star, angle_x, angle_y);
                double sigma_y_mod = calculate_sigma(sigma_y, buf_z(i, j, k), beta_star, angle_x, angle_y);

                // Calculate the exponent for the Gaussian distribution
                double exponent = -0.5 * (x_rot * x_rot / (sigma_x_mod * sigma_x_mod) +
                                          y_rot * y_rot / (sigma_y_mod * sigma_y_mod) +
                                          z_rot_yz * z_rot_yz / (sigma_z * sigma_z));

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
