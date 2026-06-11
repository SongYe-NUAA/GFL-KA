#!/usr/bin/env python3
"""
Test script to verify analyze_similarity_logs.py works from different directories
"""

import os
import sys
import subprocess

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n🧪 {description}")
    print(f"Command: {cmd}")
    print(f"Working directory: {os.getcwd()}")
    
    try:
        # Use the same Python executable that's running this script
        if cmd.startswith("python "):
            cmd = cmd.replace("python ", f'"{sys.executable}" ', 1)
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Command executed successfully")
            # Check if plot was created
            if "test_" in cmd:
                plot_name = cmd.split("test_")[1].split(".png")[0] + ".png"
                if os.path.exists(plot_name):
                    print(f"✅ Plot file created: {plot_name}")
                    return True
                else:
                    print(f"❌ Plot file not found: {plot_name}")
                    return False
            return True
        else:
            print(f"❌ Command failed with code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def main():
    print("🚀 Testing analyze_similarity_logs.py from different directories\n")
    
    results = []
    
    # Test 1: From project root
    print("="*60)
    print("Test 1: Running from project root directory")
    print("="*60)
    
    if os.path.basename(os.getcwd()) == 'tools':
        os.chdir('..')  # Go to project root
    
    cmd1 = "python tools/analyze_similarity_logs.py --log_file log/68.log --output_plot test_root.png"
    result1 = run_command(cmd1, "Testing from root with relative paths")
    results.append(("From root directory", result1))
    
    # Test 2: From tools directory
    print("\n" + "="*60)
    print("Test 2: Running from tools directory")
    print("="*60)
    
    os.chdir('tools')
    
    cmd2 = "python analyze_similarity_logs.py --log_file ../log/68.log --output_plot test_tools.png"
    result2 = run_command(cmd2, "Testing from tools with relative paths")
    results.append(("From tools directory", result2))
    
    # Test 3: Direct path test
    print("\n" + "="*60)
    print("Test 3: Using absolute path (from tools)")
    print("="*60)
    
    log_abs_path = os.path.abspath("../log/68.log")
    cmd3 = f"python analyze_similarity_logs.py --log_file \"{log_abs_path}\" --output_plot test_absolute.png"
    result3 = run_command(cmd3, "Testing with absolute path")
    results.append(("Using absolute path", result3))
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY OF RESULTS")
    print("="*60)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:<25}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 All tests passed! The script works correctly from different directories.")
    else:
        print(f"\n⚠️ Some tests failed. The path handling may need more work.")
    
    print("\nGenerated test files:")
    for f in ["test_root.png", "test_tools.png", "test_absolute.png"]:
        abs_path = os.path.abspath(f)
        if os.path.exists(abs_path):
            size = os.path.getsize(abs_path)
            print(f"  ✅ {abs_path} ({size} bytes)")
        else:
            # Check in parent directory too
            parent_path = os.path.abspath(f"../{f}")
            if os.path.exists(parent_path):
                size = os.path.getsize(parent_path)
                print(f"  ✅ {parent_path} ({size} bytes)")
            else:
                print(f"  ❌ {f} (not found)")

if __name__ == "__main__":
    main()