#!/usr/bin/env python3
"""
Debug script to test the analyze_similarity_logs.py functionality
"""

import sys
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def test_basic_functionality():
    print("🔍 Debugging analyze_similarity_logs.py functionality...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    
    # Test if we can import the module
    try:
        from analyze_similarity_logs import SimilarityLogAnalyzer
        print("✅ Successfully imported SimilarityLogAnalyzer")
    except Exception as e:
        print(f"❌ Failed to import SimilarityLogAnalyzer: {e}")
        return False
    
    # Test if log file exists
    log_file = "../log/68.log"
    abs_log_file = os.path.abspath(log_file)
    print(f"Log file path: {log_file}")
    print(f"Absolute log file path: {abs_log_file}")
    print(f"Log file exists: {os.path.exists(log_file)}")
    print(f"Absolute log file exists: {os.path.exists(abs_log_file)}")
    
    if not os.path.exists(log_file):
        print("❌ Log file not found from current directory")
        return False
    
    # Test analyzer creation
    try:
        analyzer = SimilarityLogAnalyzer(log_file)
        print("✅ Successfully created analyzer")
    except Exception as e:
        print(f"❌ Failed to create analyzer: {e}")
        return False
    
    # Test data parsing
    try:
        data = analyzer.parse_similarity_data()
        print(f"✅ Successfully parsed data: {len(data)} data points")
        if len(data) == 0:
            print("❌ No data points found")
            return False
    except Exception as e:
        print(f"❌ Failed to parse data: {e}")
        return False
    
    # Test plotting
    try:
        plot_path = analyzer.plot_similarity_metrics(data, "debug_plot.png")
        print(f"✅ Attempted to create plot: {plot_path}")
        
        # Check if plot was actually created
        if os.path.exists(plot_path):
            file_size = os.path.getsize(plot_path)
            print(f"✅ Plot file created successfully: {plot_path} (Size: {file_size} bytes)")
            return True
        else:
            print(f"❌ Plot file was not created: {plot_path}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to create plot: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_matplotlib():
    print("\n🔍 Testing matplotlib functionality...")
    try:
        import matplotlib
        print(f"Matplotlib version: {matplotlib.__version__}")
        print(f"Matplotlib backend: {matplotlib.get_backend()}")
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create a simple test plot
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        plt.figure(figsize=(8, 6))
        plt.plot(x, y)
        plt.title("Test Plot")
        plt.xlabel("X")
        plt.ylabel("Y")
        
        test_plot_path = "matplotlib_test.png"
        plt.savefig(test_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if os.path.exists(test_plot_path):
            file_size = os.path.getsize(test_plot_path)
            print(f"✅ Matplotlib test plot created: {test_plot_path} (Size: {file_size} bytes)")
            return True
        else:
            print(f"❌ Matplotlib test plot was not created")
            return False
            
    except Exception as e:
        print(f"❌ Matplotlib test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting debugging tests...\n")
    
    matplotlib_ok = test_matplotlib()
    basic_ok = test_basic_functionality()
    
    print(f"\n📊 Test Results:")
    print(f"  Matplotlib test: {'✅ PASS' if matplotlib_ok else '❌ FAIL'}")
    print(f"  Basic functionality test: {'✅ PASS' if basic_ok else '❌ FAIL'}")
    
    if matplotlib_ok and basic_ok:
        print("\n🎉 All tests passed! The script should work correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")