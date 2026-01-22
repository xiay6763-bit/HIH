#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <filesystem>
#include <cstring>
#include <iomanip>
#include "viper/viper.hpp"

using namespace viper;

// ==========================================
// 🎓 论文实验配置 (Experimental Setup)
// ==========================================
const std::string PMEM_PATH = "/pmem0/viper_bench"; 
const size_t NUM_KEYS = 1000000;   
const size_t NUM_OPS  = 1000000;   

int main() {
    // [Step 0] 环境清理
    std::filesystem::remove_all(PMEM_PATH);

    std::cout << "=========================================================" << std::endl;
    std::cout << "   HIH Viper Benchmark (Paper Standard Evaluation)       " << std::endl;
    std::cout << "=========================================================" << std::endl;
    std::cout << "Dataset Size : " << NUM_KEYS << " keys" << std::endl;
    std::cout << "Storage Path : " << PMEM_PATH << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;

    // [Step 1] 初始化数据库
    auto viper = Viper<uint64_t, uint64_t>::create(PMEM_PATH, 2UL * 1024 * 1024 * 1024);

    // 🔑【关键修改】获取 Client 对象！
    // 所有的 put/get 操作必须通过 client 进行，而不是 viper 指针
    auto client = viper->get_client();

    // [Step 2] 准备数据
    std::cout << "[Setup] Generating random keys..." << std::endl;
    std::vector<uint64_t> keys(NUM_KEYS);
    for (size_t i = 0; i < NUM_KEYS; ++i) keys[i] = i; 
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(keys.begin(), keys.end(), g);

    // [Step 3] Load Phase (Insert)
    std::cout << "[Phase 1] Starting LOAD (Insert) phase..." << std::endl;
    auto start_ins = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < NUM_KEYS; ++i) {
        // ✅ 修正：使用 client.put (小写)
        client.put(keys[i], keys[i] + 2026); 
    }
    
    auto end_ins = std::chrono::high_resolution_clock::now();
    double ins_duration = std::chrono::duration<double>(end_ins - start_ins).count();
    double ins_throughput = (NUM_KEYS / ins_duration) / 1000000.0;
    
    std::cout << ">>> [Result] Insert Throughput: " << std::fixed << std::setprecision(2) 
              << ins_throughput << " M ops/sec" << std::endl;

    // [Step 4] Run Phase (Get)
    std::shuffle(keys.begin(), keys.end(), g);
    
    std::cout << "[Phase 2] Starting RUN (Get) phase..." << std::endl;
    auto start_get = std::chrono::high_resolution_clock::now();

    uint64_t found_cnt = 0;
    uint64_t val;
    for (size_t i = 0; i < NUM_OPS; ++i) {
        // ✅ 修正：使用 client.get (小写)
        if (client.get(keys[i % NUM_OPS], &val)) {
            found_cnt++;
        }
    }

    auto end_get = std::chrono::high_resolution_clock::now();
    double get_duration = std::chrono::duration<double>(end_get - start_get).count();
    double get_throughput = (NUM_OPS / get_duration) / 1000000.0;
    double get_latency = (get_duration * 1e9) / NUM_OPS;

    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << ">>> [Result] GET Throughput : " << std::fixed << std::setprecision(2) 
              << get_throughput << " M ops/sec" << std::endl;
    std::cout << ">>> [Result] GET Avg Latency: " << std::fixed << std::setprecision(2) 
              << get_latency << " ns" << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;

    if (found_cnt != NUM_OPS) {
        std::cerr << "!!! ERROR: Correctness check failed! Found " << found_cnt << std::endl;
    } else {
        std::cout << "✅ Correctness check passed." << std::endl;
    }

    return 0;
}