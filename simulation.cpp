#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const int Nx = 400;                 //X-resolution
const int Ny = 100;                 //Y-resolution
const double rho0 = 40.0;           //Avg Density
const double tau = 0.6;            //Collision Timescale
const int Nt = 4000;                //Number of Timestep Interations
const bool plotRealTime = true;     //Display vis
const int visT = 1;                //Timestep per vis

void initialize(vector<vector<vector<double>>> &F, vector<double> &cxs, vector<double> &cys, vector<double> &weights) {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> d(0, 0.01);

    F.resize(Ny, vector<vector<double>>(Nx, vector<double>(9, rho0 / 9.0)));

    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            for (int i = 0; i < 9; ++i) {
                F[y][x][i] += d(gen);
            }
            F[y][x][3] += 2 * (1 + 0.2 * cos(2 * M_PI * x / Nx * 4));
        }
    }
}

void calculateRhoUxUy(const vector<vector<vector<double>>> &F, vector<vector<double>> &rho, vector<vector<double>> &ux,
    vector<vector<double>> &uy, const vector<double> &cxs, const vector<double> &cys) {

    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            double sum_rho = 0.0;
            double sum_ux = 0.0;
            double sum_uy = 0.0;
            for (int i = 0; i < 9; ++i) {
                sum_rho += F[y][x][i];
                sum_ux += F[y][x][i] * cxs[i];
                sum_uy += F[y][x][i] * cys[i];
            }
            rho[y][x] = sum_rho;
            ux[y][x] = sum_ux / sum_rho;
            uy[y][x] = sum_uy / sum_rho;
        }
    }
}

void applyCollision(vector<vector<vector<double>>> &F, const vector<vector<double>> &rho, const vector<vector<double>> &ux,
    const vector<vector<double>> &uy, const vector<double> &cxs, const vector<double> &cys, const vector<double> &weights) {

    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            for (int i = 0; i < 9; ++i) {
                double cu = 3 * (cxs[i] * ux[y][x] + cys[i] * uy[y][x]);
                double cu2 = cu * cu;
                double u2 = ux[y][x] * ux[y][x] + uy[y][x] * uy[y][x];
                double feq = rho[y][x] * weights[i] * (1 + cu + 0.5 * cu2 - 1.5 * u2);
                F[y][x][i] += -(1.0 / tau) * (F[y][x][i] - feq);
            }
        }
    }
}

void stream(vector<vector<vector<double>>> &F, const vector<double> &cxs, const vector<double> &cys) {
    vector<vector<vector<double>>> F_temp = F;

    for (int i = 0; i < 9; ++i) {
        int cx = static_cast<int>(cxs[i]);
        int cy = static_cast<int>(cys[i]);
        for (int y = 0; y < Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                int newX = (x + cx + Nx) % Nx;
                int newY = (y + cy + Ny) % Ny;
                F[newY][newX][i] = F_temp[y][x][i];
            }
        }
    }
}

void setReflectiveBoundaries(vector<vector<vector<double>>> &F, const vector<vector<bool>> &cylinder) {
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            if (cylinder[y][x]) {
                vector<double> bndryF(9);
                for (int i = 0; i < 9; ++i) {
                    bndryF[i] = F[y][x][i];
                }
                F[y][x][1] = bndryF[5];
                F[y][x][2] = bndryF[6];
                F[y][x][3] = bndryF[7];
                F[y][x][4] = bndryF[8];
                F[y][x][5] = bndryF[1];
                F[y][x][6] = bndryF[2];
                F[y][x][7] = bndryF[3];
                F[y][x][8] = bndryF[4];
            }
        }
    }
}

void visualize(const vector<vector<double>> &ux, const vector<vector<double>> &uy, const vector<vector<bool>> &cylinder) {
    Mat vorticityImg(Ny, Nx, CV_64F);
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            double vorticity = (ux[(y+1) % Ny][x] - ux[(y-1+Ny) % Ny][x]) - (uy[y][(x+1) % Nx] - uy[y][(x-1+Nx) % Nx]);
            if (cylinder[y][x]) {
                vorticityImg.at<double>(y, x) = NAN;
            } else {
                vorticityImg.at<double>(y, x) = vorticity;
            }
        }
    }

    // Normalize and convert vorticity image to CV_8UC1
    Mat normalizedVorticityImg;
    normalize(vorticityImg, normalizedVorticityImg, 0, 1, NORM_MINMAX);
    normalizedVorticityImg *= 255;
    normalizedVorticityImg.convertTo(normalizedVorticityImg, CV_8UC1);

    // Apply color map
    Mat colorMappedImg;
    applyColorMap(normalizedVorticityImg, colorMappedImg, COLORMAP_JET);

    // Set cylinder area to white
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            if (cylinder[y][x]) {
                colorMappedImg.at<Vec3b>(y, x) = Vec3b(255, 255, 255);
            }
        }
    }

    // Display the image
    imshow("Vorticity", colorMappedImg);
    waitKey(1);
}

int main() {
    vector<double> cxs = {0, 0, 1, 1, 1, 0, -1, -1, -1};
    vector<double> cys = {0, 1, 1, 0, -1, -1, -1, 0, 1};
    vector<double> weights = {4.0 / 9, 1.0 / 9, 1.0 / 36, 1.0 / 9, 1.0 / 36, 1.0 / 9, 1.0 / 36, 1.0 / 9, 1.0 / 36};

    vector<vector<vector<double>>> F;
    initialize(F, cxs, cys, weights);

    vector<vector<double>> rho(Ny, vector<double>(Nx));
    vector<vector<double>> ux(Ny, vector<double>(Nx));
    vector<vector<double>> uy(Ny, vector<double>(Nx));

    vector<vector<bool>> cylinder(Ny, vector<bool>(Nx, false));
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            if (pow(x - Nx / 4, 2) + pow(y - Ny / 2, 2) < pow(Ny / 4, 2)) {
                cylinder[y][x] = true;
            }
        }
    }

    for (int it = 0; it < Nt; ++it) {
        cout << "Iteration: " << it << endl;

        stream(F, cxs, cys);

        setReflectiveBoundaries(F, cylinder);

        calculateRhoUxUy(F, rho, ux, uy, cxs, cys);

        applyCollision(F, rho, ux, uy, cxs, cys, weights);

        if (plotRealTime && (it % visT == 0 || it == Nt - 1)) {
            visualize(ux, uy, cylinder);
        }
    }

    return 0;
}

