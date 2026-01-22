#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <filesystem>
#include <cstring>
#include <iomanip>  // 用于格式化输出
#include "viper/viper.hpp"

using namespace viper;

// ==========================================
// 🎓 论文实验配置 (Experimental Setup)
// ==========================================
// 路径：确保这是你的 PMem 挂载路径
const std::string PMEM_PATH = "/pmem0/viper_bench"; 

// 数据规模：论文通常测 100万 到 1亿。
// 建议先用 100万 (1M) 跑通，确认提升幅度；写论文图表时如果需要更震撼，可以改成 1000万 (10M)。
const size_t NUM_KEYS = 1000000;   
const size_t NUM_OPS  = 1000000;   

// ==========================================

int main() {
    // [Step 0] 环境清理 (Ensuring a Clean Slate)
    // 每次运行前删除旧数据，防止之前的实验干扰本次结果
    std::filesystem::remove_all(PMEM_PATH);

    std::cout << "=========================================================" << std::endl;
    std::cout << "   HIH Viper Benchmark (Paper Standard Evaluation)       " << std::endl;
    std::cout << "=========================================================" << std::endl;
    std::cout << "Dataset Size : " << NUM_KEYS << " keys" << std::endl;
    std::cout << "Storage Path : " << PMEM_PATH << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;

    // [Step 1] 初始化数据库
    // 分配 2GB 空间 (100万条数据通常占用几百MB，给2GB足够)
    auto viper = Viper<uint64_t, uint64_t>::create(PMEM_PATH, 2UL * 1024 * 1024 * 1024);

    // [Step 2] 准备数据 (Data Generation)
    std::cout << "[Setup] Generating random keys..." << std::endl;
    std::vector<uint64_t> keys(NUM_KEYS);
    for (size_t i = 0; i < NUM_KEYS; ++i) {
        keys[i] = i; 
    }
    
    // 关键步骤：打乱顺序！(Simulate Uniform Random Distribution)
    // 只有打乱了，才能测试出哈希表处理冲突和随机访问的真实能力
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(keys.begin(), keys.end(), g);

    // [Step 3] Load Phase - 测试写入吞吐量 (INSERT Performance)
    std::cout << "[Phase 1] Starting LOAD (Insert) phase..." << std::endl;
    auto start_ins = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < NUM_KEYS; ++i) {
        // Key 是随机顺序的，Value 随便填
        viper->put(keys[i], keys[i] + 2026); 
    }
    
    auto end_ins = std::chrono::high_resolution_clock::now();
    double ins_duration = std::chrono::duration<double>(end_ins - start_ins).count();
    double ins_throughput = (NUM_KEYS / ins_duration) / 1000000.0; // M ops/sec
    
    std::cout << ">>> [Result] Insert Throughput: " << std::fixed << std::setprecision(2) 
              << ins_throughput << " M ops/sec" << std::endl;

    // [Step 4] Run Phase - 测试查询吞吐量 (GET Performance)
    // 这是你【创新点一】的核心验证部分！
    
    // 再次打乱 Key，模拟完全随机的读取请求，而不是刚才插入的顺序
    std::shuffle(keys.begin(), keys.end(), g);
    
    std::cout << "[Phase 2] Starting RUN (Get) phase..." << std::endl;
    auto start_get = std::chrono::high_resolution_clock::now();

    uint64_t found_cnt = 0;
    uint64_t val;
    for (size_t i = 0; i < NUM_OPS; ++i) {
        // 执行查询：get(key, &val)
        // 使用 i % NUM_KEYS 确保我们查的 Key 都在库里，模拟 100% Hit Rate 的场景
        if (viper->get(keys[i % NUM_OPS], &val)) {
            found_cnt++;
        }
    }

    auto end_get = std::chrono::high_resolution_clock::now();
    double get_duration = std::chrono::duration<double>(end_get - start_get).count();
    double get_throughput = (NUM_OPS / get_duration) / 1000000.0; // M ops/sec
    double get_latency = (get_duration * 1e9) / NUM_OPS;          // ns

    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << ">>> [Result] GET Throughput : " << std::fixed << std::setprecision(2) 
              << get_throughput << " M ops/sec" << std::endl;
    std::cout << ">>> [Result] GET Avg Latency: " << std::fixed << std::setprecision(2) 
              << get_latency << " ns" << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;

    // 结果校验 (Sanity Check)
    if (found_cnt != NUM_OPS) {
        std::cerr << "!!! ERROR: Correctness check failed!" << std::endl;
        std::cerr << "    Expected to find " << NUM_OPS << " keys, but found " << found_cnt << std::endl;
    } else {
        std::cout << "✅ Correctness check passed." << std::endl;
    }

    return 0;
}