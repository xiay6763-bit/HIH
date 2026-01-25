import struct
import random
from tqdm import tqdm
import os

# 配置
NUM_RECORDS = 10000000  # 1000万条数据
VALUE_SIZE = 200        # 200字节的Value
DATA_DIR = "/root/HIH/data"

# 操作码
OP_INSERT = 0
OP_GET = 1
OP_UPDATE = 2

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)

# 1. 生成预热数据 (Prefill: 全部是 INSERT)
prefill_file = os.path.join(DATA_DIR, "ycsb_prefill.dat")
print(f"🚀 正在生成 1000万条 预热数据到 {prefill_file} ...")
with open(prefill_file, 'wb') as f:
    dummy_value = b'a' * VALUE_SIZE
    for key in tqdm(range(NUM_RECORDS)):
        f.write(struct.pack('<I', OP_INSERT) + struct.pack('<Q', key) + dummy_value)

# 2. 生成 Uniform 测试数据 (Run: 90% 查询, 10% 更新, 键值完全随机)
run_file = os.path.join(DATA_DIR, "ycsb_run_1090_uniform.dat")
print(f"🚀 正在生成 1000万条 Uniform 跑分数据到 {run_file} ...")
with open(run_file, 'wb') as f:
    dummy_value = b'b' * VALUE_SIZE
    for _ in tqdm(range(NUM_RECORDS)):
        # 90% 概率是 GET，10% 概率是 UPDATE
        op = OP_GET if random.random() < 0.9 else OP_UPDATE
        # Uniform 分布：完全随机的 Key
        key = random.randint(0, NUM_RECORDS - 1)
        f.write(struct.pack('<I', op) + struct.pack('<Q', key) + dummy_value)

print("✅ 数据生成完毕！现在可以去跑分了！")
