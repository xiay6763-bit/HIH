#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <filesystem>
#include <cstring>
#include <iomanip>
#include <cmath>
#include "viper/viper.hpp"

using namespace viper;

const std::string PMEM_PATH = "/pmem0/viper_zipf"; 
const size_t NUM_KEYS = 1000000;
const size_t NUM_OPS  = 2000000;

std::string new_key(uint64_t i) {
    char buf[16];
    sprintf(buf, "%08lu", i); 
    return std::string(buf);
}

class ZipfianGenerator {
    double alpha, zeta_n, eta, theta;
    uint64_t n;
    double zeta(uint64_t n, double theta) {
        double sum = 0;
        for (uint64_t i = 1; i <= n; i++) sum += std::pow(1.0 / i, theta);
        return sum;
    }
public:
    ZipfianGenerator(uint64_t n, double theta) : n(n), theta(theta) {
        zeta_n = zeta(n, theta);
        alpha = 1.0 / (1.0 - theta);
        eta = (1.0 - std::pow(2.0 / n, 1.0 - theta)) / (1.0 - zeta(2, theta) / zeta_n);
    }
    uint64_t next() {
        static std::default_random_engine engine(42);
        static std::uniform_real_distribution<double> dist(0, 1);
        double u = dist(engine);
        double uz = u * zeta_n;
        if (uz < 1.0) return 0;
        if (uz < 1.0 + std::pow(0.5, theta)) return 1;
        return (uint64_t)(n * std::pow(eta * u - eta + 1, alpha));
    }
};

int main(int argc, char** argv) {
    double zipf_s = 0.99; // 默认值
    if (argc > 1) {
        zipf_s = std::atof(argv[1]);
    }

    std::filesystem::remove_all(PMEM_PATH);
    auto viper = Viper<std::string, std::string>::create(PMEM_PATH, 4UL * 1024 * 1024 * 1024);
    auto client = viper->get_client();

    // Phase 1: Load
    std::string val(64, 'x'); 
    for (size_t i = 0; i < NUM_KEYS; ++i) client.put(new_key(i), val);

    // Phase 2: Zipfian Sequence
    ZipfianGenerator zipf(NUM_KEYS - 1, zipf_s);
    std::vector<uint64_t> access_indices;
    access_indices.reserve(NUM_OPS);
    for (size_t i = 0; i < NUM_OPS; ++i) access_indices.push_back(zipf.next());

    // Phase 3: Run
    auto start_get = std::chrono::high_resolution_clock::now();
    uint64_t found_cnt = 0;
    std::string val_out;
    for (size_t i = 0; i < NUM_OPS; ++i) {
        if (client.get(new_key(access_indices[i]), &val_out)) found_cnt++;
    }
    auto end_get = std::chrono::high_resolution_clock::now();
    double dur = std::chrono::duration<double>(end_get - start_get).count();
    
    // 输出简炼格式，方便脚本抓取
    std::cout << "S=" << zipf_s << " | Throughput: " << std::fixed << std::setprecision(2) 
              << (NUM_OPS / dur) / 1000000.0 << " M ops/s | Latency: " << (dur * 1e9) / NUM_OPS << " ns" << std::endl;

    return 0;
}
