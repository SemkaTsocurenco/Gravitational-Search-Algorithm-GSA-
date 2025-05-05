#include "./init.h"

using namespace gsa;


// ---------------- Isomap‑MDS (быстрая схема) -----------------------------
// Проекция N×D -> N×d_low (d_low = 2)
Eigen::MatrixXd mds2D(const Matrix& pos)
{
    Eigen::MatrixXd B(N_PARTICLES, N_PARTICLES);
    // (1) геодезические = евклид here (ускоренная версия)
    #pragma omp parallel for
    for (int i = 0; i < N_PARTICLES; ++i)
        for (int j = i; j < N_PARTICLES; ++j) {
            double d2 = dist_sq(pos[i], pos[j]);
            B(i,j) = B(j,i) = std::sqrt(d2);
        }
    // (2) классическое MDS: двойное центрирование
    Eigen::VectorXd rowMean = B.rowwise().mean();
    double totalMean = rowMean.mean();
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(N_PARTICLES,N_PARTICLES);
    for(int i=0;i<N_PARTICLES;++i)
        for(int j=0;j<N_PARTICLES;++j)
            C(i,j) = -0.5*(B(i,j)*B(i,j)
                         - rowMean[i]*rowMean[i]
                         - rowMean[j]*rowMean[j]
                         + totalMean*totalMean);
    // (3) top‑2 eigen‑vectors
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(C);
    Eigen::VectorXd vals = es.eigenvalues().tail(2).cwiseMax(0.0).cwiseSqrt();
    Eigen::MatrixXd vecs = es.eigenvectors().rightCols(2);
    return vecs * vals.asDiagonal();
}

// ---------------- MGGSA‑специфические шаги -------------------------------
Vec lambda_coeff(const Matrix& pos)
{
    Eigen::MatrixXd Y = mds2D(pos);              // N×2 manifold map
    /* 1. pick the K_best indices (smallest fitness) */
    std::vector<int> idx(N_PARTICLES);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin()+K_BEST, idx.end(),
                      [&](int a,int b){ return target(pos[a]) < target(pos[b]); });
    idx.resize(K_BEST);

    /* 2. gather ‘scores’ for the same K particles in both spaces */
    std::vector<double> s_high, s_low;  s_high.reserve(K_BEST); s_low.reserve(K_BEST);
    for(int k : idx){
        s_high.push_back(target(pos[k]));            // original fitness
        s_low .push_back(Y.row(k).norm());           // radius in 2‑D
    }

    auto to_rank = [](const std::vector<double>& v){
        std::vector<int> id(v.size()); std::iota(id.begin(),id.end(),0);
        std::sort(id.begin(),id.end(),[&](int a,int b){ return v[a] < v[b]; });
        std::vector<int> rk(v.size()); for(int p=0;p<(int)v.size();++p) rk[id[p]] = p;
        return rk;
    };
    auto r_high = to_rank(s_high);
    auto r_low  = to_rank(s_low );

    /* 3. diff[d] = Σ|rank_high‑rank_low| over K_best */
    Vec diff(DIMENSIONS,0.0);
    for(int t=0;t<K_BEST;++t){
        double delta = std::abs(r_high[t]-r_low[t]);
        for(int d=0; d<DIMENSIONS; ++d) diff[d]+=delta;
    }

    /* 4. group → δ_im_norm → δ'_im (eqs. 26‑32 in the paper) */
    double diff_max = *std::max_element(diff.begin(),diff.end()) + EPS;
    Vec diff_norm(DIMENSIONS);
    std::transform(diff.begin(), diff.end(), diff_norm.begin(),
                   [&](double x){ return x/diff_max; });

    /* simple uniform grouping:   L = 5 gives good results */
    constexpr int L = 5;
    Vec lambda(DIMENSIONS);
    for(int g=0; g<L; ++g){
        double low = (double)g/L, high = (double)(g+1)/L;
        Vec group_idx;
        for(int d=0;d<DIMENSIONS;++d)
            if(diff_norm[d]>=low && diff_norm[d]<high) group_idx.push_back(d);

        if(group_idx.empty()) continue;

        double delta_im = 0.0;
        for(auto d:group_idx) delta_im += diff_norm[d];
        delta_im /= group_idx.size();

        double delta_prime = delta_im - 1.0/L + 1.0;     // eq. (32) simplified
        for(auto d:group_idx) lambda[d] = 1.0 / delta_prime;
    }
    /* guarantee ⟨λ⟩ = 1 */
    double mean = std::accumulate(lambda.begin(),lambda.end(),0.0)/DIMENSIONS + EPS;
    for(auto& l:lambda) l *= DIMENSIONS*1.0/mean;

    return lambda;
}


void MGGSA(){
    
    using namespace gsa;

}