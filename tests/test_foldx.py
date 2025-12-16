# FoldX Debug Version - 保留输出文件用于诊断
# 只计算第一个候选,保存所有FoldX输出文件

import os
import subprocess
import shutil
from pathlib import Path
import time

# 配置
FOLDX_BINARY = "/usr/local/bin/foldx"
PDB_FILE = "/content/disulfide_results/XynA.pdb"  # 修改为你的PDB文件路径
DEBUG_OUTPUT = "/content/foldx_debug_output"

# 测试突变
TEST_MUTATIONS = ["SA49C", "SA68C"]  # S49C+S68C

print("=" * 70)
print("FoldX Debug Test")
print("=" * 70)
print(f"PDB: {PDB_FILE}")
print(f"Mutations: {TEST_MUTATIONS}")
print(f"Output: {DEBUG_OUTPUT}")
print("=" * 70)

# 清理并创建输出目录
if os.path.exists(DEBUG_OUTPUT):
    shutil.rmtree(DEBUG_OUTPUT)
os.makedirs(DEBUG_OUTPUT, exist_ok=True)

# 复制PDB
pdb_name = Path(PDB_FILE).name
work_pdb = os.path.join(DEBUG_OUTPUT, pdb_name)
shutil.copy(PDB_FILE, work_pdb)

# 创建突变列表
mut_file = os.path.join(DEBUG_OUTPUT, "individual_list.txt")
with open(mut_file, 'w') as f:
    mutation_str = ",".join(TEST_MUTATIONS)
    f.write(f"{mutation_str};\n")

print(f"\nMutation file content:")
with open(mut_file, 'r') as f:
    print(f.read())

# 运行FoldX
cmd = [
    FOLDX_BINARY,
    "--command=BuildModel",
    f"--pdb={pdb_name}",
    "--mutant-file=individual_list.txt",
    "--numberOfRuns=3"
]

print(f"\nFoldX command:")
print(" ".join(cmd))

print(f"\nRunning FoldX...")
start = time.time()

result = subprocess.run(
    cmd,
    cwd=DEBUG_OUTPUT,
    capture_output=True,
    text=True,
    timeout=300
)

elapsed = time.time() - start

print(f"Completed in {elapsed:.1f}s")
print(f"Return code: {result.returncode}")

# 显示stdout/stderr
print(f"\n{'=' * 70}")
print("STDOUT:")
print(result.stdout[:500])

print(f"\n{'=' * 70}")
print("STDERR:")
print(result.stderr[:500])

# 列出所有输出文件
print(f"\n{'=' * 70}")
print("Output files:")
print("=" * 70)

for f in sorted(os.listdir(DEBUG_OUTPUT)):
    size = os.path.getsize(os.path.join(DEBUG_OUTPUT, f))
    print(f"  {f:<40} {size:>10,} bytes")

# 查看所有.fxout文件
fxout_files = [f for f in os.listdir(DEBUG_OUTPUT) if f.endswith('.fxout')]

print(f"\n{'=' * 70}")
print(f"Found {len(fxout_files)} .fxout files")
print("=" * 70)

for filename in sorted(fxout_files):
    filepath = os.path.join(DEBUG_OUTPUT, filename)
    
    print(f"\n{'=' * 70}")
    print(f"File: {filename}")
    print("=" * 70)
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    print(f"Lines: {len(lines)}\n")
    
    # 显示所有内容(如果不太长)
    if len(lines) <= 30:
        for i, line in enumerate(lines, 1):
            print(f"{i:3d} | {line.rstrip()}")
    else:
        # 显示前20行和后10行
        print("First 20 lines:")
        for i, line in enumerate(lines[:20], 1):
            print(f"{i:3d} | {line.rstrip()}")
        
        print(f"\n... ({len(lines) - 30} lines omitted) ...\n")
        
        print("Last 10 lines:")
        for i, line in enumerate(lines[-10:], len(lines) - 9):
            print(f"{i:3d} | {line.rstrip()}")
    
    print()

print(f"\n{'=' * 70}")
print("Debug output saved to:")
print(DEBUG_OUTPUT)
print("=" * 70)
print("\nYou can now:")
print("1. Examine the files manually")
print("2. Share the .fxout file contents for further analysis")
print("3. Run the diagnostic script on this directory")
