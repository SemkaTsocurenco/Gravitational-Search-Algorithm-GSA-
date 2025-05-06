#include "./init.h"
#include "fitness_logger.h"
using namespace gsa;


/* ---------- Быстрый MDS (N×D → N×2) ------------- */
static Eigen::MatrixXd mds2D(const Matrix& pos)
{
    static Eigen::MatrixXd B(N_PARTICLES, N_PARTICLES);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N_PARTICLES; ++i) {
        B(i,i)=0.0;
        for (int j = i+1; j < N_PARTICLES; ++j) {
            double d2 = dist_sq(pos[i], pos[j]);
            double d  = std::sqrt(d2);
            B(i,j) = B(j,i) = d;
        }
    }
    // двойное центрирование
    Eigen::VectorXd rowMean = B.rowwise().mean();
    double totalMean = rowMean.mean();
    Eigen::MatrixXd C = -0.5*(B.array().square().matrix());
    C.colwise()      -= rowMean;
    C.rowwise()      -= rowMean.transpose();
    C.array()        += totalMean;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(C);
    Eigen::VectorXd vals = es.eigenvalues().tail(2).cwiseMax(0.0).cwiseSqrt();
    return es.eigenvectors().rightCols(2) * vals.asDiagonal();
}

/* ---------- λ‑коэффициенты ---------------------- */
static Vec lambda_coeff(const Matrix& pos, const Vec& fitness)
{
    static Eigen::MatrixXd Y;                 // переиспользуем память
    Y = mds2D(pos);

    /* 1. K_best индексы */
    std::vector<int> idx(N_PARTICLES);
    std::iota(idx.begin(), idx.end(), 0);
    int K = std::max(K_MIN, N_PARTICLES/10);
    std::partial_sort(idx.begin(), idx.begin()+K, idx.end(),
                      [&](int a,int b){ return fitness[a] < fitness[b]; });
    idx.resize(K);

    /* 2. ранги */
    auto to_rank = [](const std::vector<double>& v){
        std::vector<int> id(v.size()); std::iota(id.begin(), id.end(), 0);
        std::sort(id.begin(), id.end(), [&](int a,int b){ return v[a] < v[b]; });
        std::vector<int> r(v.size()); for(int p=0;p<(int)v.size();++p) r[id[p]]=p; return r; };

    std::vector<double> s_high(K), s_low(K);
    for(int t=0;t<K;++t){ s_high[t]=fitness[idx[t]];  s_low[t]=Y.row(idx[t]).norm(); }
    auto r_high = to_rank(s_high);  auto r_low  = to_rank(s_low);

    /* 3. diff[d] */
    Vec diff(DIMENSIONS, 0.0);
    for(int t=0;t<K;++t){ double dlt = std::abs(r_high[t]-r_low[t]);
        for(int d=0; d<DIMENSIONS; ++d) diff[d]+=dlt; }

    double diff_max = *std::max_element(diff.begin(), diff.end()) + EPS;
    for(auto& v:diff) v /= diff_max;                 // нормировка 0..1

    /* 4. группировка и λ */
    Vec lambda(DIMENSIONS,1.0);
    for(int g=0; g<L_GROUPS; ++g){
        double low=(double)g/L_GROUPS, high=(double)(g+1)/L_GROUPS;
        std::vector<int> dimlist;
        for(int d=0; d<DIMENSIONS; ++d) if(diff[d]>=low && diff[d]<high) dimlist.push_back(d);
        if(dimlist.empty()) continue;
        double delta_im=0.0; for(int d:dimlist) delta_im+=diff[d]; delta_im/=dimlist.size();
        double lambda_g = 1.0/(delta_im + 0.01);           // усиление эффективных, подавление слабых
        for(int d:dimlist) lambda[d]=lambda_g;
    }
    // среднее к 1
    double mean = std::accumulate(lambda.begin(), lambda.end(), 0.0)/DIMENSIONS + EPS;
    for(auto& l:lambda) l /= mean;
    return lambda;
}

/* ---------- Основная функция ------------------------------------------- */
void MGGSA()
{
    FitnessLogger flog("../results/MGGSA_DATA.csv");
    auto t0 = std::chrono::high_resolution_clock::now();

    Matrix pos(N_PARTICLES, Vec(DIMENSIONS));
    Matrix vel(N_PARTICLES, Vec(DIMENSIONS, 0.0));
    Matrix acc(N_PARTICLES, Vec(DIMENSIONS, 0.0));

    #pragma omp parallel for
    for (int i = 0; i < N_PARTICLES; ++i)
        for (int d = 0; d < DIMENSIONS; ++d)
            pos[i][d] = rand_double(MIN_RAND, MAX_RAND);

    Vec fitness(N_PARTICLES), mass(N_PARTICLES);
    Vec glob_best(DIMENSIONS,0.0);
    double glob_fit = std::numeric_limits<double>::infinity();
    const double V_MAX = (MAX_RAND-MIN_RAND)*V_MAX_FRAC;

    Vec lambda(DIMENSIONS,1.0);   // кэшируем между пересчётами

    for (int it = 0; it < ITERATIONS; ++it)
    {
        /* 1. fitness */
        #pragma omp parallel for schedule(static)
        for (int i=0;i<N_PARTICLES;++i) fitness[i]=target(pos[i]);

        int best_idx = std::min_element(fitness.begin(), fitness.end()) - fitness.begin();
        if (fitness[best_idx] < glob_fit) { glob_fit = fitness[best_idx]; glob_best = pos[best_idx]; }

        /* 2. masses */
        double fbest=*std::min_element(fitness.begin(),fitness.end());
        double fworst=*std::max_element(fitness.begin(),fitness.end());
        double den  = fworst-fbest+EPS;
        #pragma omp parallel for schedule(static)
        for(int i=0;i<N_PARTICLES;++i) mass[i]=(fworst-fitness[i])/den;
        double msum = std::accumulate(mass.begin(),mass.end(),0.0);
        for(auto& m:mass) m/= (msum+EPS);

        /* 3. λ every MDSTEP iterations */
        if(it%MDSTEP==0) lambda = lambda_coeff(pos, fitness);

        /* 4. forces */
        for (auto& a:acc) std::fill(a.begin(),a.end(),0.0);
        double G = G0*std::exp(-ALPHA*it/ITERATIONS);
        #pragma omp parallel for schedule(dynamic)
        for(int i=0;i<N_PARTICLES;++i){
            for(int j=0;j<N_PARTICLES;++j){ if(i==j) continue;
                double dist = std::sqrt(dist_sq(pos[i],pos[j])) + EPS;
                double factor = G*mass[j]/dist;
                double r = rand_double(0.0,1.0);
                for(int d=0; d<DIMENSIONS; ++d)
                    acc[i][d] += r*factor*lambda[d]*(pos[j][d]-pos[i][d]); }
        }

        /* 5. update */
        #pragma omp parallel for schedule(static)
        for(int i=0;i<N_PARTICLES;++i)
            for(int d=0; d<DIMENSIONS; ++d){
                double r = rand_double(0.0,1.0);
                vel[i][d] = r*vel[i][d] + acc[i][d];
                vel[i][d] = std::clamp(vel[i][d], -V_MAX, V_MAX);
                pos[i][d] += vel[i][d];
                pos[i][d]  = std::clamp(pos[i][d], MIN_RAND, MAX_RAND);
            }

        long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-t0).count();
        flog.push(it, ms, glob_fit);
    }

    std::cout << "\nMGGSA final best=" << glob_fit << '\n';
}
