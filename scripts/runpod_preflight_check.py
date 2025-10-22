#!/usr/bin/env python3
"""Preflight check for RunPod out-of-domain testing"""
import os
import sys

print("=" * 80)
print("Brie 3B Out-of-Domain Testing - Preflight Check")
print("=" * 80)

checks_passed = 0
checks_total = 0

def check(name, condition, fix_hint=""):
    global checks_passed, checks_total
    checks_total += 1
    if condition:
        print(f"✅ {name}")
        checks_passed += 1
        return True
    else:
        print(f"❌ {name}")
        if fix_hint:
            print(f"   Fix: {fix_hint}")
        return False

print("\n[1/7] Checking Python environment...")
check("Python 3.8+", sys.version_info >= (3, 8), "Upgrade Python")

print("\n[2/7] Checking CUDA availability...")
try:
    import torch
    check("PyTorch installed", True)
    check("CUDA available", torch.cuda.is_available(), "Use a GPU instance")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
except ImportError:
    check("PyTorch installed", False, "Run: pip install torch")

print("\n[3/7] Checking required packages...")
packages_ok = True
try:
    import transformers
    print(f"   transformers: {transformers.__version__}")
except ImportError:
    check("transformers", False, "Run: pip install transformers")
    packages_ok = False

try:
    import peft
    print(f"   peft: {peft.__version__}")
except ImportError:
    check("peft", False, "Run: pip install peft")
    packages_ok = False

if packages_ok:
    check("Required packages", True)

print("\n[4/7] Checking directory structure...")
check("exports/ exists", os.path.exists("exports"), "Run: mkdir -p exports")
check("scripts/ exists", os.path.exists("scripts"), "In wrong directory?")
check("runs/ exists", os.path.exists("runs"), "Run: mkdir -p runs")

print("\n[5/7] Checking Brie 3B model...")
brie_path = "runs/brie-v2-3b"
check(f"{brie_path}/ exists", os.path.exists(brie_path),
      f"Upload model to {brie_path}/")
check(f"{brie_path}/adapter_model.safetensors",
      os.path.exists(f"{brie_path}/adapter_model.safetensors"),
      "Model file missing - re-upload Brie 3B")
check(f"{brie_path}/adapter_config.json",
      os.path.exists(f"{brie_path}/adapter_config.json"),
      "Config file missing - re-upload Brie 3B")

print("\n[6/7] Checking test script...")
test_script = "scripts/runpod_out_of_domain_3b.py"
check(f"{test_script} exists", os.path.exists(test_script),
      "Pull latest from git")
check(f"{test_script} executable", os.access(test_script, os.X_OK),
      f"Run: chmod +x {test_script}")

print("\n[7/7] Estimating disk space...")
if os.path.exists("runs/brie-v2-3b"):
    try:
        import subprocess
        result = subprocess.run(['du', '-sh', 'runs/brie-v2-3b'],
                              capture_output=True, text=True)
        size = result.stdout.split()[0]
        print(f"   Brie 3B size: {size}")
    except:
        pass

# Check available space
try:
    import shutil
    stat = shutil.disk_usage(".")
    free_gb = stat.free / (1024**3)
    check(f"Free disk space: {free_gb:.1f}GB", free_gb > 10,
          "Need at least 10GB free for models + results")
except:
    pass

print("\n" + "=" * 80)
print(f"PREFLIGHT CHECK: {checks_passed}/{checks_total} passed")
print("=" * 80)

if checks_passed == checks_total:
    print("\n✅ All checks passed! Ready to run:")
    print("   python scripts/runpod_out_of_domain_3b.py")
else:
    print(f"\n⚠️  {checks_total - checks_passed} checks failed. Fix issues above before proceeding.")
    sys.exit(1)
