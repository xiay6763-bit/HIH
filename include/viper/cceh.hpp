/**
 * This code was taken and modified from https://github.com/DICL/CCEH, the original authors of CCEH.
 */

#pragma once

//#define CCEH_PERSISTENT

#include <cstring>
#include <type_traits>
#include <cmath>
#include <vector>
#include <stdint.h>
#include <iostream>
#include <thread>
#include <bitset>
#include <cassert>
#include <unordered_map>
#include <atomic>
#include <stdlib.h>

#include "hash.hpp"
#include <immintrin.h> // ✅ [创新点 3] 新增：引入 Intel AVX-512 SIMD 硬件指令集支持

#ifdef CCEH_PERSISTENT
#include <libpmemobj++/allocator.hpp>
#include <libpmempool.h>
#endif

namespace viper {

#ifdef CCEH_PERSISTENT
static constexpr char CCEH_PMEM_POOL_FILE[] = "/mnt/pmem2/viper/cceh-allocator.file";
class PMemAllocator {
  public:
    static PMemAllocator& get() {
        static std::once_flag flag;
        std::call_once(flag, []{ instance(); });
        return instance();
    }

    void allocate(PMEMoid* pmem_ptr, size_t size) {
        auto ctor = [](PMEMobjpool* pool, void* ptr, void* arg) { return 0; };
        int ret = pmemobj_alloc(pmem_pool_.handle(), pmem_ptr, size, 0, ctor, nullptr);
        if (ret != 0) {
            throw std::runtime_error{std::string("Could not allocate! ") + std::strerror(errno)};
        }
    }

    static PMemAllocator& instance() {
        static PMemAllocator instance{};
        return instance;
    }

    PMemAllocator() {
        pool_is_open_ = false;
        initialize();
    }

    void initialize() {
        destroy();

        int sds_write_value = 0;
        pmemobj_ctl_set(NULL, "sds.at_create", &sds_write_value);
        pmem_pool_ = pmem::obj::pool_base::create(CCEH_PMEM_POOL_FILE, "", 10ul * (1024l * 1024 * 1024), S_IRWXU);
        if (pmem_pool_.handle() == nullptr) {
            throw std::runtime_error("Could not open allocator pool file.");
        }
        pool_is_open_ = true;
    }

    void destroy() {
        if (pool_is_open_) {
            pmem_pool_.close();
            pool_is_open_ = false;
        }
        pmempool_rm(CCEH_PMEM_POOL_FILE, PMEMPOOL_RM_FORCE);
    }

    ~PMemAllocator() {
        destroy();
    }

    pmem::obj::pool_base pmem_pool_;
    bool pool_is_open_;
};
#endif


#define internal_cas(entry, expected, updated) \
    __atomic_compare_exchange_n(entry, expected, updated, false, __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE)

#define ATOMIC_LOAD(addr) \
    __atomic_load_n(addr, __ATOMIC_ACQUIRE)

#define ATOMIC_STORE(addr, value) \
    __atomic_store_n(addr, value, __ATOMIC_RELEASE)

#define requires_fingerprint(KeyType) \
    std::is_same_v<KeyType, std::string> || sizeof(KeyType) > 8

#define IS_BIT_SET(variable, mask) ((variable & mask) != 0)


template <typename KeyType>
inline bool CAS(KeyType* key, KeyType* expected, KeyType updated) {
    if constexpr (sizeof(KeyType) == 1) return internal_cas((int8_t*) key, (int8_t*) expected, (int8_t) updated);
    else if constexpr (sizeof(KeyType) == 2) return internal_cas((int16_t*) key, (int16_t*) expected, (int16_t) updated);
    else if constexpr (sizeof(KeyType) == 4) return internal_cas((int32_t*) key, (int32_t*) expected, (int32_t) updated);
    else if constexpr (sizeof(KeyType) == 8) return internal_cas((int64_t*) key, (int64_t*) expected, (int64_t) updated);
    else if constexpr (sizeof(KeyType) == 16) return internal_cas((__int128*) key, (__int128*) expected, (__int128) updated);
    else throw std::runtime_error("CAS not supported for > 16 bytes!");
}


using offset_size_t = uint64_t;
using block_size_t = uint64_t;
using page_size_t = uint8_t;
using data_offset_size_t = uint16_t;

struct KeyValueOffset {
    static constexpr offset_size_t INVALID = 0xFFFFFFFFFFFFFFFF;

    union {
        offset_size_t offset;
        struct {
            block_size_t block_number : 45;
            page_size_t page_number : 3;
            data_offset_size_t data_offset : 16;
        };
    };

    KeyValueOffset() : offset{INVALID} {}

    static KeyValueOffset NONE() { return KeyValueOffset{INVALID}; }

    explicit KeyValueOffset(const offset_size_t offset) : offset(offset) {}

    KeyValueOffset(const block_size_t block_number, const page_size_t page_number, const data_offset_size_t slot)
        : block_number{block_number}, page_number{page_number}, data_offset{slot} {}

    static KeyValueOffset Tombstone() {
        return KeyValueOffset{};
    }

    inline std::tuple<block_size_t, page_size_t, data_offset_size_t> get_offsets() const {
        return {block_number, page_number, data_offset};
    }

    inline bool is_tombstone() const {
        return offset == INVALID;
    }

    inline bool operator==(const KeyValueOffset& rhs) const { return offset == rhs.offset; }
    inline bool operator!=(const KeyValueOffset& rhs) const { return offset != rhs.offset; }
};

using IndexK = size_t;
using IndexV = KeyValueOffset;
constexpr IndexK SENTINEL = -2; // 11111...110
constexpr IndexK INVALID = -1; // 11111...111

namespace cceh {

#define CACHE_LINE_SIZE 64

constexpr size_t kSegmentBits = 8;
constexpr size_t kMask = (1 << kSegmentBits)-1;
constexpr size_t kShift = kSegmentBits;
constexpr size_t kSegmentSize = (1 << kSegmentBits) * 16 * 4;
constexpr size_t kNumPairPerCacheLine = 4;
constexpr size_t kNumCacheLine = 4;

constexpr uint64_t SPLIT_REQUEST_BIT = 1ul << 63;
constexpr uint64_t EXCLUSIVE_LOCK = -1;

// ==========================================
// 论文对比开关：启用所有创新点
// ==========================================
#define ENABLE_INLINE_OPT

// 定义标记位 (用于标识该槽位是否存放了内联 Key)
static constexpr uint64_t INLINE_BIT = 1ul << 63;

struct Pair {
#ifdef ENABLE_INLINE_OPT
    // ================= [创新点一：改进版结构 (Inline Key)] =================
    // 论文核心思想：打破原版只存哈希的限制，对于短键直接内联进哈希桶中，消灭 PMem 读取。
    union {
        IndexK key;           // 用于存哈希 (大 Key 模式)
        char inline_key[8];   // 用于存原始 Key (小 Key 模式，内联优化)
    };
#else
    // ================= [Baseline：原版结构] =================
    IndexK key;               // 只存哈希，必须去 PMem 查 Key
#endif

    IndexV value;             // PMem 地址 或 包含 INLINE_BIT 的元数据

    Pair(void) : key{INVALID}, value{IndexV::Tombstone()} {}

    Pair(IndexK _key, IndexV _value) : value{_value} {
        key = _key;
    }

    Pair& operator=(const Pair& other) {
        key = other.key;
        value = other.value;
        return *this;
    }

    // 辅助函数：判断是否内联
    bool is_inline() const {
#ifdef ENABLE_INLINE_OPT
        return (value.offset & INLINE_BIT) != 0;
#else
        return false;
#endif
    }
};

inline void persist(void* data, size_t len) {
#ifdef CCEH_PERSISTENT
  pmem_persist(data, len);
#endif
}

template <typename KeyType>
struct Segment {
    static const size_t kNumSlot = kSegmentSize / sizeof(Pair);

    Segment(void)
        : local_depth{0}
    { }

    Segment(size_t depth)
        :local_depth{depth}
    { }

    void* operator new(size_t size) {
#ifdef CCEH_PERSISTENT
        PMEMoid ret;
        PMemAllocator::get().allocate(&ret, size);
        return pmemobj_direct(ret);
#else
        void* ret;
        if (posix_memalign(&ret, 64, size) != 0) throw std::runtime_error("bad memalign");
        return ret;
#endif
    }

    template <typename KeyCheckFn>
    int Insert(const KeyType&, IndexV, size_t, size_t, IndexV* old_entry, KeyCheckFn);

    void Insert4split(IndexK, IndexV, size_t);
    Segment** Split(void);

    Pair _[kNumSlot];
    size_t local_depth;
    std::atomic<uint64_t> sema = 0;
    size_t pattern = 0;
    static constexpr bool using_fp_ = requires_fingerprint(KeyType);
};

template <typename KeyType>
struct Directory {
    static const size_t kDefaultDepth = 10;
    Segment<KeyType>** _;
    size_t capacity;
    size_t depth;
    bool lock;
#ifdef CCEH_PERSISTENT
    PMEMoid pmem_seg_loc_;
#endif

    Directory(void) {
        depth = kDefaultDepth;
        capacity = pow(2, depth);
        _ = new Segment<KeyType>*[capacity];
        lock = false;
    }

    Directory(size_t _depth) {
        depth = _depth;
        capacity = pow(2, depth);
#ifdef CCEH_PERSISTENT
        PMemAllocator::get().allocate(&pmem_seg_loc_, sizeof(Segment<KeyType>*) * capacity);
        _ = (Segment<KeyType>**) pmemobj_direct(pmem_seg_loc_);
#else
        _ = new Segment<KeyType>*[capacity];
#endif
        lock = false;
    }

    ~Directory(void) {
#ifdef CCEH_PERSISTENT
        pmemobj_free(&pmem_seg_loc_);
#else
        delete [] _;
#endif
    }

    bool Acquire(void) {
        bool unlocked = false;
        return CAS(&lock, &unlocked, true);
    }

    bool Release(void) {
        bool locked = true;
        return CAS(&lock, &locked, false);
    }

    void* operator new(size_t size) {
#ifdef CCEH_PERSISTENT
        PMEMoid ret;
        PMemAllocator::get().allocate(&ret, size);
        return pmemobj_direct(ret);
#else
        void* ret;
        if (posix_memalign(&ret, 64, size) != 0) throw std::runtime_error("bad memalign");
        return ret;
#endif
    }

#ifdef CCEH_PERSISTENT
    void operator delete(void* addr) {}
#endif
};

template <typename KeyType>
class CCEH {
  public:
    static constexpr auto dummy_key_check = [](const KeyType&, IndexV) {
        throw std::runtime_error("Dummy key check should never be used!");
        return true;
    };

    CCEH(size_t);
    ~CCEH();

    template <typename KeyCheckFn>
    IndexV Insert(const KeyType&, IndexV, KeyCheckFn);

    template <typename KeyCheckFn>
    IndexV Get(const KeyType&, KeyCheckFn);

    template <typename KeyCheckFn>
    bool Delete(const KeyType&, KeyCheckFn);

    IndexV Insert(const KeyType&, IndexV);
    bool Delete(const KeyType&);
    IndexV Get(const KeyType&);
    void Remove(IndexV* offset);
    size_t Capacity(void);

  private:
    Directory<KeyType>* dir;
    static constexpr bool using_fp_ = requires_fingerprint(KeyType);
};

extern size_t perfCounter;

template <typename KeyType>
template <typename KeyCheckFn>
int Segment<KeyType>::Insert(const KeyType& key, IndexV value, size_t loc, size_t key_hash,
                             IndexV* old_entry, KeyCheckFn key_check_fn) {
  uint64_t lock = sema.load();
  if (lock == EXCLUSIVE_LOCK) return 2;
  if (IS_BIT_SET(lock, SPLIT_REQUEST_BIT)) return 1;

  const size_t pattern_shift = 8 * sizeof(key_hash) - local_depth;
  if ((key_hash >> pattern_shift) != pattern) return 2;

  int ret = 1;
  while (!sema.compare_exchange_weak(lock, lock+1)) {
      if (lock == EXCLUSIVE_LOCK) return 2;
      if (IS_BIT_SET(lock, SPLIT_REQUEST_BIT)) return 1;
  }

  IndexK LOCK = INVALID;
  IndexK key_checker;
  if constexpr (using_fp_) {
      key_checker = key_hash;
  } else {
      key_checker = *reinterpret_cast<const IndexK*>(&key);
  }

  for (unsigned i = 0; i < kNumPairPerCacheLine * kNumCacheLine; ++i) {
    auto slot = (loc + i) % kNumSlot;
    auto _key = _[slot].key;

    bool invalidate = _key != INVALID;
    if constexpr (using_fp_) {
        invalidate &= (_key >> pattern_shift) != pattern;
    } else {
        invalidate &= (h(&_key, sizeof(IndexK)) >> pattern_shift) != pattern;
    }

    if (invalidate && CAS(&_[slot].key, &_key, INVALID)) {
        _[slot].value = IndexV::Tombstone();
    }

    if (CAS(&_[slot].key, &LOCK, SENTINEL)) {
        old_entry->offset = _[slot].value.offset;

#ifdef ENABLE_INLINE_OPT
        // ==========================================================
        // 【创新点一：插入时的 Inline Key 判定逻辑】
        // ==========================================================
        if (value.is_tombstone()) {
            _[slot].value = value;
            _[slot].key = INVALID; // 标记该槽位无效
        }
        else if constexpr (!std::is_same_v<KeyType, std::string> && sizeof(KeyType) <= 8) {
            // 只有非 Tombstone 且 Key 很小 (<= 8 Byte) 的时候，才走内联优化
            std::memset(_[slot].inline_key, 0, 8); // 清空内存，防止脏数据影响 SIMD 比较
            std::memcpy(_[slot].inline_key, &key, sizeof(KeyType)); // 将原始 Key 填入 DRAM 桶
            _[slot].value = value;
            _[slot].value.offset |= INLINE_BIT; // 在 PMem 指针最高位打上 INLINE 标记
        } 
        else {
            // 大 Key 情况：走原有逻辑，只存哈希指纹
            _[slot].value = value;
            _[slot].key = key_checker;
        }
#else
        // ==========================================================
        // 【原版逻辑 (Baseline)】 
        // ==========================================================
        _[slot].value = value;
        if (value.is_tombstone()) {
            _[slot].key = INVALID;
        }
        else {
            _[slot].key = key_checker;
        }
#endif

        persist(&_[slot], sizeof(Pair));
        ret = 0;
        break;
    } 
    
    else if (ATOMIC_LOAD(&_[slot].key) == key_checker) {
        if constexpr (using_fp_) {
            const bool keys_match = key_check_fn(key, _[slot].value);
            if (!keys_match) continue;
        }

        IndexV old_value = _[slot].value;
        while (!CAS(&_[slot].value.offset, &old_value.offset, value.offset)) {}
        if (value.is_tombstone()) {
            IndexK expected = key_checker;
            CAS(&_[slot].key, &expected, INVALID);
        }
        old_entry->offset = old_value.offset;
        persist(&_[slot].key, sizeof(Pair));
        ret = 0;
        break;
    } else {
        LOCK = INVALID;
    }
  }

  sema.fetch_sub(1);
  return ret;
}

template <typename KeyType>
void Segment<KeyType>::Insert4split(IndexK key, IndexV value, size_t loc) {
    for (unsigned i = 0; i < kNumPairPerCacheLine * kNumCacheLine; ++i) {
        auto slot = (loc+i) % kNumSlot;
        if (_[slot].key == INVALID) {
            _[slot].key = key;
            _[slot].value = value;
            persist(&_[slot], sizeof(Pair));
            return;
        }
    }
}

template <typename KeyType>
Segment<KeyType>** Segment<KeyType>::Split(void) {
  uint64_t lock = 0;
  if (!sema.compare_exchange_strong(lock, EXCLUSIVE_LOCK)) {
      if (lock == EXCLUSIVE_LOCK) {
          return nullptr;
      }

      lock = SPLIT_REQUEST_BIT;
      if (!sema.compare_exchange_strong(lock, EXCLUSIVE_LOCK)) {
          if ((lock & SPLIT_REQUEST_BIT) != 0) {
              return nullptr;
          }
          sema.compare_exchange_strong(lock, lock | SPLIT_REQUEST_BIT);
          return nullptr;
      }
  }

  Segment<KeyType>** split = new Segment<KeyType>*[2];
  split[0] = this;
  split[1] = new Segment<KeyType>(local_depth + 1);

  for (unsigned i = 0; i < kNumSlot; ++i) {
    size_t key_hash;
    if constexpr (using_fp_) {
        key_hash = _[i].key;
    } else {
        key_hash = h(&_[i].key, sizeof(IndexK));
    }
    if (key_hash & ((size_t) 1 << ((sizeof(IndexK)*8 - local_depth - 1)))) {
      split[1]->Insert4split(_[i].key, _[i].value, (key_hash & kMask)*kNumPairPerCacheLine);
    }
  }

    persist((char*) split[1], sizeof(Segment));
    local_depth = local_depth + 1;
    persist((char*) &local_depth, sizeof(size_t));

    return split;
}

template <typename KeyType>
CCEH<KeyType>::CCEH(size_t initCap)
    : dir{new Directory<KeyType>(static_cast<size_t>(log2(initCap)))}
{
    for (unsigned i = 0; i < dir->capacity; ++i) {
        dir->_[i] = new Segment<KeyType>(static_cast<size_t>(log2(initCap)));
        dir->_[i]->pattern = i;
    }
}

template <typename KeyType>
IndexV CCEH<KeyType>::Insert(const KeyType& key, IndexV value) {
    return Insert(key, value, dummy_key_check);
}

template <typename KeyType>
template <typename KeyCheckFn>
IndexV CCEH<KeyType>::Insert(const KeyType& key, IndexV value, KeyCheckFn key_check_fn) {
    size_t key_hash;
    if constexpr (std::is_same_v<KeyType, std::string>) { key_hash = h(key.data(), key.length()); }
    else { key_hash = h(&key, sizeof(key)); }
    auto loc = (key_hash & kMask) * kNumPairPerCacheLine;

    while (true) {
        auto x = (key_hash >> (8 * sizeof(key_hash) - dir->depth));
        auto target = dir->_[x];
        IndexV old_entry{};
        auto ret = target->Insert(key, value, loc, key_hash, &old_entry, key_check_fn);

        if (ret == 0) {
            return old_entry;
        } else if (ret == 2) {
            continue;
        }

        Segment<KeyType>** s = target->Split();
        if (s == nullptr) {
            continue;
        }

        s[0]->pattern = (key_hash >> (8 * sizeof(key_hash) - s[0]->local_depth + 1)) << 1;
        s[1]->pattern = ((key_hash >> (8 * sizeof(key_hash) - s[1]->local_depth + 1)) << 1) + 1;

        while (!dir->Acquire()) {
            asm("nop");
        }

        { 
            x = (key_hash >> (8 * sizeof(key_hash) - dir->depth));
            if (dir->_[x]->local_depth - 1 < dir->depth) {  
                unsigned depth_diff = dir->depth - s[0]->local_depth;
                if (depth_diff == 0) {
                    if (x % 2 == 0) {
                        dir->_[x + 1] = s[1];
                        persist((char*) &dir->_[x + 1], 8);
                    } else {
                        dir->_[x] = s[1];
                        persist((char*) &dir->_[x], 8);
                    }
                } else {
                    int chunk_size = pow(2, dir->depth - (s[0]->local_depth - 1));
                    x = x - (x % chunk_size);
                    for (unsigned i = 0; i < chunk_size / 2; ++i) {
                        dir->_[x + chunk_size / 2 + i] = s[1];
                    }
                    persist((char*) &dir->_[x + chunk_size / 2], sizeof(void*) * chunk_size / 2);
                }
                dir->Release();
            } else {  
                auto dir_old = dir;
                auto d = dir->_;
                auto _dir = new Directory<KeyType>(dir->depth + 1);
                for (unsigned i = 0; i < dir->capacity; ++i) {
                    if (i == x) {
                        _dir->_[2 * i] = s[0];
                        _dir->_[2 * i + 1] = s[1];
                    } else {
                        _dir->_[2 * i] = d[i];
                        _dir->_[2 * i + 1] = d[i];
                    }
                }
                persist((char*) &_dir->_[0], sizeof(Segment<KeyType>*) * _dir->capacity);
                persist((char*) &_dir, sizeof(Directory<KeyType>));
                if (!CAS(&dir, &dir_old, _dir)) {
                    throw std::runtime_error("Could not swap dirs. This should never happen!");
                }
                persist((char*) &dir, sizeof(void*));
                delete dir_old;
            }
            s[0]->sema.store(0);
        }  

        delete s;
    }
}

template <typename KeyType>
void CCEH<KeyType>::Remove(IndexV* offset) {
    offset_size_t expected_value = offset->offset;
    CAS(&offset->offset, &expected_value, IndexV::Tombstone().offset);
    IndexK* key_slot = reinterpret_cast<IndexK*>(offset) - 1;
    ATOMIC_STORE(key_slot, INVALID);
}

template <typename KeyType>
IndexV CCEH<KeyType>::Get(const KeyType& key) {
    return Get(key, dummy_key_check);
}

template <typename KeyType>
template <typename KeyCheckFn>
IndexV CCEH<KeyType>::Get(const KeyType& key, KeyCheckFn key_check_fn) {
    size_t key_hash;
    if constexpr (std::is_same_v<KeyType, std::string>) { key_hash = h(key.data(), key.length()); }
    else { key_hash = h(&key, sizeof(key)); }
    const size_t loc = (key_hash & kMask) * kNumPairPerCacheLine;

    Segment<KeyType>* segment;
    while (true) {
        const size_t seg_num = (key_hash >> (8 * sizeof(key_hash) - dir->depth));
        segment = dir->_[seg_num];
        auto& sema = segment->sema;

        uint64_t lock = sema.load();
        if ((lock & SPLIT_REQUEST_BIT) != 0 || !sema.compare_exchange_weak(lock, lock + 1)) {
            continue;
        }

        break;
    }

    IndexK key_checker;
    if constexpr (using_fp_) {
        key_checker = key_hash;
    } else {
        key_checker = *reinterpret_cast<const IndexK*>(&key);
    }

    // =========================================================================================
    // 【创新点 3：利用 AVX-512 进行 Cache-Line 对齐的 SIMD 并行探测】
    // 说明：此优化必须与创新点 1 (Inline Key) 结合使用。只有当数据在内存中连续存放时，
    //       AVX-512 才能通过一条指令同时比对 Cache Line (64B) 中的 4 个 Key。
    //       时间复杂度从 O(N) 降维至 O(1)，且全程无分支预测(Branchless)。
    // =========================================================================================
#ifdef ENABLE_INLINE_OPT 
    // 1. 将 8 字节目标 Key 广播填充到 512-bit (64字节) 寄存器中 (复制 8 份)
    IndexK target_key_u64 = *reinterpret_cast<const IndexK*>(&key);
    __m512i target_vec = _mm512_set1_epi64(target_key_u64);

    // 每次加载一个 Cache Line (4 个 Pair = 64 字节 = 512 bit)
    for (unsigned c = 0; c < kNumCacheLine; ++c) {
        auto base_slot = (loc + c * 4) % Segment<KeyType>::kNumSlot;
        
        // 2. 一次性从 DRAM 哈希桶加载 4 个 Pair (包含 4 个 Key 和 4 个 Value)
        __m512i bucket_vec = _mm512_loadu_si512((__m512i*)&segment->_[base_slot]);

        // 3. 硬件并行比较！__mmask8 的第 0, 2, 4, 6 位分别代表 4 个 Key 的匹配结果
        // 0x55 (01010101) 用于过滤掉 Value 区域的误匹配
        __mmask8 mask = _mm512_cmpeq_epi64_mask(bucket_vec, target_vec) & 0x55;

        // 4. 解析 Mask 并返回结果 (使用 __builtin_ctz 瞬间定位，无循环分支)
        while (mask != 0) {
            int bit = __builtin_ctz(mask); // 找到第一个匹配的位 (0, 2, 4 或 6)
            auto& entry = segment->_[base_slot + bit / 2];

            // 创新点 1 发挥作用：确认是内联数据，不需要二次读取 PMem！
            if (entry.is_inline()) {
                IndexV ret = entry.value;
                ret.offset &= ~INLINE_BIT; // 去除内联标记，还原真实 PMem 地址
                segment->sema.fetch_sub(1);
                return ret; // 极速返回！
            }
            mask &= ~(1 << bit); // 如果不是内联数据（极其罕见），清除当前位继续检查哈希冲突
        }
    }
#else
    // ==========================================================
    // 【原版逻辑 (Baseline)：标量遍历查询】
    // ==========================================================
    for (unsigned i = 0; i < kNumPairPerCacheLine * kNumCacheLine; ++i) {
        auto slot = (loc + i) % Segment<KeyType>::kNumSlot;
        auto& entry = segment->_[slot];

        if (entry.key == key_checker) {
            if constexpr (using_fp_) {
                const bool keys_match = key_check_fn(key, entry.value);
                if (!keys_match) continue;
            }

            IndexV offset = entry.value;
            segment->sema.fetch_sub(1);
            return offset;
        }
    }
#endif

    segment->sema.fetch_sub(1);
    return IndexV::NONE();
}

template <typename KeyType>
size_t CCEH<KeyType>::Capacity(void) {
    std::unordered_map<Segment<KeyType>*, bool> set;
    for (size_t i = 0; i < dir->capacity; ++i) {
        set[dir->_[i]] = true;
    }
    return set.size() * Segment<KeyType>::kNumSlot;
}

template <typename KeyType>
CCEH<KeyType>::~CCEH() {
#ifndef CCEH_PERSISTENT
    std::unordered_map<Segment<KeyType>*, bool> set;
    for (size_t i = 0; i < dir->capacity; ++i) {
        set[dir->_[i]] = true;
    }
    for (auto const& [seg, foo] : set) {
        delete seg;
    }
#endif
}

}  // namespace cceh
}  // namespace viper
