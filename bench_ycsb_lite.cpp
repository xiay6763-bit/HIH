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

const std::string PMEM_PATH = "/pmem0/viper_bench"; 
const size_t NUM_KEYS = 1000000;   
const size_t NUM_OPS  = 1000000;   

// 🔴 修改点：生成 8 字节短字符串 (00000000 - 00999999)
std::string new_key(uint64_t i) {
    char buf[32];
    sprintf(buf, "%08lu", i); 
    return std::string(buf);
}

int main() {
    std::filesystem::remove_all(PMEM_PATH);

    std::cout << "=========================================================" << std::endl;
    std::cout << "   HIH Viper Benchmark (SHORT STRING Mode)               " << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    // 使用 String 类型
    auto viper = Viper<std::string, std::string>::create(PMEM_PATH, 4UL * 1024 * 1024 * 1024);
    auto client = viper->get_client();

    std::cout << "[Setup] Generating random SHORT keys (8 bytes)..." << std::endl;
    std::vector<std::string> keys;
    keys.reserve(NUM_KEYS);
    for (size_t i = 0; i < NUM_KEYS; ++i) {
        keys.push_back(new_key(i));
    }
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(keys.begin(), keys.end(), g);

    std::cout << "[Phase 1] Starting LOAD..." << std::endl;
    std::string val(64, 'x'); 
    for (size_t i = 0; i < NUM_KEYS; ++i) {
        client.put(keys[i], val); 
    }
    
    std::shuffle(keys.begin(), keys.end(), g);
    
    std::cout << "[Phase 2] Starting RUN (Get)..." << std::endl;
    auto start_get = std::chrono::high_resolution_clock::now();

    uint64_t found_cnt = 0;
    std::string val_out;
    for (size_t i = 0; i < NUM_OPS; ++i) {
        if (client.get(keys[i % NUM_OPS], &val_out)) {
            found_cnt++;
        }
    }

    auto end_get = std::chrono::high_resolution_clock::now();
    double get_duration = std::chrono::duration<double>(end_get - start_get).count();
    double get_throughput = (NUM_OPS / get_duration) / 1000000.0;
    double get_latency = (get_duration * 1e9) / NUM_OPS;

    std::cout << ">>> [Result] GET Throughput : " << std::fixed << std::setprecision(2) 
              << get_throughput << " M ops/sec" << std::endl;
    std::cout << ">>> [Result] GET Avg Latency: " << std::fixed << std::setprecision(2) 
              << get_latency << " ns" << std::endl;

    return 0;
}
