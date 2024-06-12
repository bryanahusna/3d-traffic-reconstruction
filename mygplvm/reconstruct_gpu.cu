#define M_PI   3.14159265358979323846264338327950288
#define degToRad(angleInDegrees) ((angleInDegrees) * M_PI / 180.0)
#define radToDeg(angleInRadians) ((angleInRadians) * 180.0 / M_PI)

#define LAMBDA_SIZE 7

__device__ void multiplyMatricesVector(float m[4][4], float v[4], float result[4]) {
    for (int i = 0; i < 4; i++) {
        result[i] = 0;
        for (int j = 0; j < 4; j++) {
            result[i] += m[i][j] * v[j];
        }
    }
}

extern "C" __global__ void optimize_pose(const float * input_image, const float * sdfs, const float * grads, const float * lambdas, float * dE_dlis, float MI[4][4], const int dimension)
{
    // parameters
    float zeta_smoothing = 0.75f;

    // function parameters
    // float Tx = lambdas[0];
    // float Ty = lambdas[1];
    // float Tz = lambdas[2];
    float Rx = degToRad(lambdas[3]);
    float Ry = degToRad(lambdas[4]);
    float Rz = degToRad(lambdas[5]);
    float S = lambdas[6];

    int idx = blockIdx.x * blockDim.x  + threadIdx.x;
    int u = idx / dimension;
    int v = idx % dimension;
    float pf = input_image[idx] + 0.1;
    float pb = 1 - input_image[idx] + 0.1;
    float pf_min_pb = 2 * input_image[idx] - 1;
    
    int sigma_lray_dlis[20];
    float sigma_log_phi_l = 0;

    float x = 1.0f * v - dimension/2;
    // x /= dimension/2;
    float y = 1.0f * dimension/2 - u;
    // y /= dimension/2;
    for (int l = 0; l < dimension; l++){
        float z = 1.0f * (l - dimension/2);
        // z /= dimension/2;
        float xyz[] = {
            x,
            y,
            z,
            1
        };
        
        float xyz0[4];
        multiplyMatricesVector(MI, xyz, xyz0);
        float x0 = xyz0[0];
        float y0 = xyz0[1];
        float z0 = xyz0[2];

        int sdf_idx = u*dimension*dimension + v*dimension + l;
        float phi_l = sdfs[sdf_idx];
        float dl_dx = grads[sdf_idx*3 + 0];
        float dl_dy = grads[sdf_idx*3 + 1];
        float dl_dz = grads[sdf_idx*3 + 2];

        float exp_phi_l_zeta = exp(-phi_l * zeta_smoothing);
        sigma_log_phi_l += log(1 - (exp_phi_l_zeta)/(exp_phi_l_zeta + 1));


        // dx/dli, dy/dli, and dz/dli
        float dx_dlis[] = {
                1,
                0,
                0,
                0,
                -x0*S*sin(Ry)*cos(Rz) + y0*S*sin(Ry)*sin(Rz) + z0*S*cos(Ry),
                -x0*S*cos(Ry)*sin(Rz) - y0*S*cos(Ry)*cos(Rz),
                0// x0*cos(Ry)*cos(Rz) - y0*cos(Ry)*sin(Rz) + z0*sin(Ry)
        };
        float dy_dlis[] = {
                0,
                1,
                0,
                x0*S*(-sin(Rx)*sin(Rz) + cos(Rx)*sin(Ry)*cos(Rz)) + y0*S*(-sin(Rx)*cos(Rz) - cos(Rx)*sin(Ry)*sin(Rz)) - z0*S*cos(Rx)*cos(Ry),
                x0*S*(sin(Rx)*cos(Ry)*cos(Rz)) + y0*S*(-sin(Rx)*cos(Ry)*sin(Rz)) + z0*S*cos(Rx)*sin(Ry),
                x0*S*(cos(Rx)*cos(Rz) - sin(Rx)*sin(Ry)*sin(Rz)) + y0*S*(-cos(Rx)*sin(Rz) - sin(Rx)*sin(Ry)*cos(Rz)),
                0 // x0*(cos(Rx)*sin(Rz) + sin(Rx)*sin(Ry)*sin(Rz)) + y0*(cos(Rx)*cos(Rz) - sin(Rx)*sin(Ry)*sin(Rz)) - z0*sin(Rx)*cos(Ry)
        };
        float dz_dlis[] = {
                0,
                0,
                1,
                x0*S*(cos(Rx)*sin(Rz) + sin(Rx)*sin(Ry)*cos(Rz)) + y0*S*(cos(Rx)*cos(Rz) - sin(Rx)*sin(Ry)*sin(Rz)) + z0*S*(-sin(Rx)*cos(Ry)),
                x0*S*(-cos(Rx)*cos(Ry)*cos(Rz)) + y0*S*(cos(Rx)*cos(Ry)*sin(Rz)) + z0*S*(-cos(Rx)*sin(Ry)),
                x0*S*(sin(Rx)*cos(Rz) + cos(Rx)*sin(Ry)*sin(Rz)) + y0*S*(-sin(Rx)*sin(Rz) + cos(Rx)*sin(Ry)*cos(Rz)),
                0 // x0*(sin(Rx)*sin(Rz) - cos(Rx)*sin(Ry)*cos(Rz)) + y0*(sin(Rx)*cos(Rz) + cos(Rx)*sin(Ry)*sin(Rz)) + z0*(cos(Rx)*cos(Ry))
        };
        
        float dl_dlis[LAMBDA_SIZE];
        for (int i = 0; i < LAMBDA_SIZE; i++) {
            dl_dlis[i] = dl_dx * dx_dlis[i] + dl_dy * dy_dlis[i] + dl_dz * dz_dlis[i];
        }
        for (int i = 0; i < LAMBDA_SIZE; i++) {
            sigma_lray_dlis[i] += (exp_phi_l_zeta / (exp_phi_l_zeta + 1)) * dl_dlis[i];
        }
    }
    
    float he = 1 - exp(sigma_log_phi_l);
    for(int i = 0; i < LAMBDA_SIZE; i++) {
        // dE_dlis[idx*LAMBDA_SIZE + i] += -pf_min_pb /(he * pf_min_pb + pb) * exp(sigma_log_phi_l) * sigma_lray_dlis[i];
        dE_dlis[idx*LAMBDA_SIZE + i] += -pf_min_pb /(he * pf + (1 - he)*pb) * exp(sigma_log_phi_l) * sigma_lray_dlis[i];
    }

    //// Project the ray
    // dE_dlis[idx] = 1 - exp(sigma_log_phi_l);
}
