#!/usr/bin/env python3
"""
Similarity Log Analyzer for MMDetection Training Logs

This script analyzes and visualizes similarity metrics from MMDetection training logs.
It supports both single-log analysis and multi-log comparison.

Features:
- Single log analysis with detailed statistics and plots
- Multi-log comparison with side-by-side visualization  
- Four similarity metrics: Mean, Max, Standard Deviation, IoU-specific
- Automatic legend generation from file names
- CSV export for further analysis

Usage:
    Single log analysis:
        python analyze_similarity_logs.py --log_file path/to/logfile.log
        
    Multi-log comparison:
        python analyze_similarity_logs.py --log_files log1.log log2.log log3.log
        python analyze_similarity_logs.py --log_files *.log
    
    As a module:
        from analyze_similarity_logs import SimilarityLogAnalyzer
        analyzer = SimilarityLogAnalyzer("path/to/logfile.log")
        data = analyzer.parse_similarity_data()
        analyzer.plot_similarity_metrics(data)
        
        # For multi-log comparison:
        multi_data = [(name1, data1), (name2, data2), ...]
        analyzer.plot_multi_log_comparison(multi_data)
"""

import re
import argparse
import os
import sys
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import itertools

# Set matplotlib backend for non-interactive environments
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


class SimilarityLogAnalyzer:
    """Analyzer for similarity metrics in MMDetection training logs"""
    
    def __init__(self, log_file_path: str):
        """
        Initialize the analyzer with a log file path
        
        Args:
            log_file_path: Path to the log file to analyze
        """
        self.log_file_path = log_file_path
        self.similarity_pattern = re.compile(
            r'.*SIMILARITY_DATA: step: (\d+), mean_sim: ([\d.]+), max_sim: ([\d.]+), std_sim: ([\d.]+)(?:, high_iou_sim: ([\d.]+))?(?:, low_iou_sim: ([\d.]+))?'
        )
    
    def parse_similarity_data(self) -> List[Dict]:
        """
        Parse similarity data from the log file
        
        Returns:
            List of dictionaries containing similarity metrics
        """
        print(f"[INFO] Reading log file: {self.log_file_path}")
        
        # Smart path resolution for different working directories
        original_path = self.log_file_path
        normalized_path = os.path.normpath(original_path)
        
        # Try multiple path variations
        possible_paths = [
            normalized_path,  # Original path
            os.path.abspath(normalized_path),  # Absolute version
        ]
        
        # If we're in tools directory and path starts with .., try without ../
        if os.path.basename(os.getcwd()) == 'tools' and original_path.startswith('..'):
            # Remove ../ or ..\ and try from project root
            cleaned_path = original_path.replace('../', '').replace('..\\', '')
            project_root = os.path.dirname(os.getcwd())
            possible_paths.append(os.path.join(project_root, cleaned_path))
        
        # If we're in project root and path doesn't have .., try with log/ prefix
        if not original_path.startswith('..') and not os.path.isabs(original_path):
            if not original_path.startswith('log'):
                possible_paths.append(os.path.join('log', original_path))
        
        # Find the first existing path
        found_path = None
        for path in possible_paths:
            if os.path.exists(path):
                found_path = path
                break
        
        if not found_path:
            print(f"[ERROR] Log file not found! Tried these paths:")
            for i, path in enumerate(possible_paths, 1):
                print(f"  {i}. {path} {'✓' if os.path.exists(path) else '✗'}")
            print(f"[DEBUG] Current working directory: {os.getcwd()}")
            return []
        
        # Update the path to use the found version
        self.log_file_path = found_path
        print(f"[DEBUG] Using log file: {found_path}")
        
        similarity_data = []
        total_lines = 0
        matching_lines = 0
        
        try:
            print(f"[DEBUG] Opening file for reading...")
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                debug_similarity_lines = 0
                for line_num, line in enumerate(f, 1):
                    total_lines += 1
                    
                    # Debug: Show SIMILARITY_DATA lines found
                    if 'SIMILARITY_DATA' in line:
                        debug_similarity_lines += 1
                        if debug_similarity_lines <= 3:
                            print(f"[DEBUG] SIMILARITY_DATA line {debug_similarity_lines} at line {line_num}: {line.strip()}")
                    
                    match = self.similarity_pattern.search(line)
                    if match:
                        matching_lines += 1
                        try:
                            step = int(match.group(1))
                            mean_sim = float(match.group(2))
                            max_sim = float(match.group(3))
                            std_sim = float(match.group(4))
                            high_iou_sim = float(match.group(5)) if match.group(5) else None
                            low_iou_sim = float(match.group(6)) if match.group(6) else None
                            
                            data_point = {
                                'step': step,
                                'mean_sim': mean_sim,
                                'max_sim': max_sim,
                                'std_sim': std_sim,
                                'high_iou_sim': high_iou_sim,
                                'low_iou_sim': low_iou_sim,
                                'line_num': line_num
                            }
                            similarity_data.append(data_point)
                            
                            # Show first few matches for debugging
                            if matching_lines <= 5:
                                print(f"  [DATA] Found data at line {line_num}: step={step}, mean_sim={mean_sim:.4f}")
                                
                        except (ValueError, IndexError) as e:
                            print(f"[WARNING] Could not parse line {line_num}: {e}")
                            continue
        
        except FileNotFoundError:
            print(f"[ERROR] Log file '{self.log_file_path}' not found!")
            return []
        except Exception as e:
            print(f"[ERROR] Error reading log file: {e}")
            return []
        
        print(f"[INFO] Parsing complete:")
        print(f"  - Total lines processed: {total_lines}")
        print(f"  - Lines containing 'SIMILARITY_DATA': {debug_similarity_lines}")
        print(f"  - Similarity data entries parsed: {matching_lines}")
        
        if matching_lines == 0:
            print("[WARNING] No similarity data found in the log file!")
            print("[DEBUG] Let me check what's actually in the file...")
            
            # Look for any lines containing similarity-related keywords
            try:
                with open(self.log_file_path, 'r', encoding='utf-8') as f:
                    sample_lines = []
                    for line_num, line in enumerate(f, 1):
                        lower_line = line.lower()
                        if 'similarity' in lower_line or 'sim' in lower_line or 'step:' in line:
                            sample_lines.append(f"Line {line_num}: {line.strip()}")
                            if len(sample_lines) >= 5:
                                break
                    
                    if sample_lines:
                        print("Found these potentially relevant lines:")
                        for sample in sample_lines:
                            print(f"  {sample}")
                    else:
                        print("No lines containing 'similarity', 'sim', or 'step:' found")
                        
                        # Show first 10 lines of the file for context
                        print("\nFirst 10 lines of the file:")
                        with open(self.log_file_path, 'r', encoding='utf-8') as f:
                            for i, line in enumerate(f, 1):
                                if i <= 10:
                                    print(f"  Line {i}: {line.strip()}")
                                else:
                                    break
                        
            except Exception as e:
                print(f"[ERROR] Error analyzing file content: {e}")
        
        # Sort by step number
        similarity_data.sort(key=lambda x: x['step'])
        return similarity_data
    
    def plot_similarity_metrics(self, data: List[Dict], save_path: Optional[str] = None) -> str:
        """
        Create plots for similarity metrics
        
        Args:
            data: List of similarity data points
            save_path: Optional path to save the plot
            
        Returns:
            Path where the plot was saved
        """
        if not data:
            print("[ERROR] No data to plot!")
            return ""
        
        # Extract data for plotting
        steps = [d['step'] for d in data]
        mean_sims = [d['mean_sim'] for d in data]
        max_sims = [d['max_sim'] for d in data]
        std_sims = [d['std_sim'] for d in data]
        
        # Extract IoU-specific similarities (may have None values)
        high_iou_sims = [d['high_iou_sim'] for d in data if d['high_iou_sim'] is not None]
        low_iou_sims = [d['low_iou_sim'] for d in data if d['low_iou_sim'] is not None]
        high_iou_steps = [d['step'] for d in data if d['high_iou_sim'] is not None]
        low_iou_steps = [d['step'] for d in data if d['low_iou_sim'] is not None]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Similarity Metrics Analysis During Training', fontsize=16, fontweight='bold')
        
        # Plot 1: Mean Similarity
        ax1.plot(steps, mean_sims, 'b-', linewidth=2, marker='o', markersize=3, alpha=0.7)
        ax1.set_title('Mean Similarity vs Training Steps', fontweight='bold')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Mean Similarity')
        ax1.grid(True, alpha=0.3)
        # Set y-axis range from minimum value to 1 for better visualization
        min_mean_sim = min(mean_sims) if mean_sims else 0
        ax1.set_ylim(min_mean_sim, 1)
        
        # Plot 2: Max Similarity
        ax2.plot(steps, max_sims, 'r-', linewidth=2, marker='o', markersize=3, alpha=0.7)
        ax2.set_title('Max Similarity vs Training Steps', fontweight='bold')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Max Similarity')
        ax2.grid(True, alpha=0.3)
        # Set y-axis range from minimum value to 1 for better visualization
        min_max_sim = min(max_sims) if max_sims else 0
        ax2.set_ylim(min_max_sim, 1)
        
        # Plot 3: Standard Deviation of Similarity
        ax3.plot(steps, std_sims, 'g-', linewidth=2, marker='o', markersize=3, alpha=0.7)
        ax3.set_title('Similarity Standard Deviation vs Training Steps', fontweight='bold')
        ax3.set_xlabel('Training Steps')
        ax3.set_ylabel('Similarity Std Dev')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, max(std_sims) * 1.1 if std_sims else 0.1)
        
        # Plot 4: IoU-specific Similarities
        if high_iou_sims:
            ax4.plot(high_iou_steps, high_iou_sims, 'purple', linewidth=2, marker='o', 
                    markersize=3, alpha=0.7, label='High IoU Similarity')
        if low_iou_sims:
            ax4.plot(low_iou_steps, low_iou_sims, 'orange', linewidth=2, marker='s', 
                    markersize=3, alpha=0.7, label='Low IoU Similarity')
        
        ax4.set_title('IoU-specific Similarities vs Training Steps', fontweight='bold')
        ax4.set_xlabel('Training Steps')
        ax4.set_ylabel('IoU Similarity')
        ax4.grid(True, alpha=0.3)
        # Set y-axis range from minimum value to 1 for better visualization
        all_iou_sims = high_iou_sims + low_iou_sims
        min_iou_sim = min(all_iou_sims) if all_iou_sims else 0
        ax4.set_ylim(min_iou_sim, 1)
        if high_iou_sims or low_iou_sims:
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No IoU-specific\nsimilarity data', 
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=12, alpha=0.6)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"similarity_analysis_{timestamp}.png"
        
        try:
            print(f"[PLOT] Saving plot to: {save_path}")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            # Check if file was actually created
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                print(f"[SUCCESS] Plot saved successfully to: {save_path} (Size: {file_size} bytes)")
            else:
                print(f"[WARNING] Plot file was not created at: {save_path}")
                
        except Exception as e:
            print(f"[ERROR] Error saving plot: {e}")
            return ""
        
        # Don't show the plot in non-interactive environments
        # plt.show()
        
        return save_path
    
    def plot_multi_log_comparison(self, multi_data: List[Tuple[str, List[Dict]]], save_path: Optional[str] = None) -> str:
        """
        Create comparison plots for multiple log files
        
        Args:
            multi_data: List of tuples (log_name, similarity_data)
            save_path: Optional path to save the plot
            
        Returns:
            Path where the plot was saved
        """
        if not multi_data:
            print("[ERROR] No data to plot!")
            return ""
        
        # Color palette for different logs
        colors = plt.cm.tab10(np.linspace(0, 1, len(multi_data)))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multi-Log Similarity Metrics Comparison', fontsize=16, fontweight='bold')
        
        # Track if any IoU data exists
        has_high_iou = False
        has_low_iou = False
        
        for idx, (log_name, data) in enumerate(multi_data):
            if not data:
                print(f"[WARNING] No data for log: {log_name}")
                continue
                
            # Extract data for plotting
            steps = [d['step'] for d in data]
            mean_sims = [d['mean_sim'] for d in data]
            max_sims = [d['max_sim'] for d in data]
            std_sims = [d['std_sim'] for d in data]
            
            # Extract IoU-specific similarities
            high_iou_sims = [d['high_iou_sim'] for d in data if d['high_iou_sim'] is not None]
            low_iou_sims = [d['low_iou_sim'] for d in data if d['low_iou_sim'] is not None]
            high_iou_steps = [d['step'] for d in data if d['high_iou_sim'] is not None]
            low_iou_steps = [d['step'] for d in data if d['low_iou_sim'] is not None]
            
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            
            # Clean log name for legend (remove path and extension)
            clean_name = os.path.splitext(os.path.basename(log_name))[0]
            
            # Plot 1: Mean Similarity
            ax1.plot(steps, mean_sims, color=color, linewidth=2, marker=marker, 
                    markersize=4, alpha=0.7, label=clean_name)
            
            # Plot 2: Max Similarity
            ax2.plot(steps, max_sims, color=color, linewidth=2, marker=marker, 
                    markersize=4, alpha=0.7, label=clean_name)
            
            # Plot 3: Standard Deviation
            ax3.plot(steps, std_sims, color=color, linewidth=2, marker=marker, 
                    markersize=4, alpha=0.7, label=clean_name)
            
            # Plot 4: IoU-specific Similarities
            if high_iou_sims:
                has_high_iou = True
                ax4.plot(high_iou_steps, high_iou_sims, color=color, linewidth=2, 
                        marker=marker, markersize=4, alpha=0.7, linestyle='-',
                        label=f'{clean_name} (High IoU)')
            if low_iou_sims:
                has_low_iou = True
                ax4.plot(low_iou_steps, low_iou_sims, color=color, linewidth=2, 
                        marker=marker, markersize=4, alpha=0.5, linestyle='--',
                        label=f'{clean_name} (Low IoU)')
        
        # Configure axes
        ax1.set_title('Mean Similarity Comparison', fontweight='bold')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Mean Similarity')
        ax1.grid(True, alpha=0.3)
        # Set y-axis range from minimum value to 1 for better visualization
        all_mean_sims = []
        for _, data in multi_data:
            if data:
                all_mean_sims.extend([d['mean_sim'] for d in data])
        min_mean_sim = min(all_mean_sims) if all_mean_sims else 0
        ax1.set_ylim(min_mean_sim, 1)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax2.set_title('Max Similarity Comparison', fontweight='bold')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Max Similarity')
        ax2.grid(True, alpha=0.3)
        # Set y-axis range from minimum value to 1 for better visualization
        all_max_sims = []
        for _, data in multi_data:
            if data:
                all_max_sims.extend([d['max_sim'] for d in data])
        min_max_sim = min(all_max_sims) if all_max_sims else 0
        ax2.set_ylim(min_max_sim, 1)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax3.set_title('Similarity Std Dev Comparison', fontweight='bold')
        ax3.set_xlabel('Training Steps')
        ax3.set_ylabel('Similarity Std Dev')
        ax3.grid(True, alpha=0.3)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax4.set_title('IoU-specific Similarities Comparison', fontweight='bold')
        ax4.set_xlabel('Training Steps')
        ax4.set_ylabel('IoU Similarity')
        ax4.grid(True, alpha=0.3)
        # Set y-axis range from minimum value to 1 for better visualization
        all_iou_sims = []
        for _, data in multi_data:
            if data:
                all_iou_sims.extend([d['high_iou_sim'] for d in data if d['high_iou_sim'] is not None])
                all_iou_sims.extend([d['low_iou_sim'] for d in data if d['low_iou_sim'] is not None])
        min_iou_sim = min(all_iou_sims) if all_iou_sims else 0
        ax4.set_ylim(min_iou_sim, 1)
        
        if has_high_iou or has_low_iou:
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax4.text(0.5, 0.5, 'No IoU-specific\nsimilarity data', 
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=12, alpha=0.6)
        
        # Adjust layout to accommodate legends
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)
        
        # Save the plot
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"multi_log_similarity_comparison_{timestamp}.png"
        
        try:
            print(f"[PLOT] Saving comparison plot to: {save_path}")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            # Check if file was actually created
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                print(f"[SUCCESS] Comparison plot saved successfully to: {save_path} (Size: {file_size} bytes)")
            else:
                print(f"[WARNING] Plot file was not created at: {save_path}")
                
        except Exception as e:
            print(f"[ERROR] Error saving comparison plot: {e}")
            return ""
        
        return save_path
    
    def print_statistics(self, data: List[Dict]) -> None:
        """
        Print statistical summary of the similarity data
        
        Args:
            data: List of similarity data points
        """
        if not data:
            print("[ERROR] No data for statistics!")
            return
        
        print("\n" + "="*60)
        print("[STATS] SIMILARITY METRICS STATISTICS")
        print("="*60)
        
        # Extract metrics
        mean_sims = [d['mean_sim'] for d in data]
        max_sims = [d['max_sim'] for d in data]
        std_sims = [d['std_sim'] for d in data]
        
        high_iou_sims = [d['high_iou_sim'] for d in data if d['high_iou_sim'] is not None]
        low_iou_sims = [d['low_iou_sim'] for d in data if d['low_iou_sim'] is not None]
        
        def print_metric_stats(name: str, values: List[float]):
            if values:
                print(f"\n{name}:")
                print(f"  • Count: {len(values)}")
                print(f"  • Mean: {np.mean(values):.4f}")
                print(f"  • Std Dev: {np.std(values):.4f}")
                print(f"  • Min: {np.min(values):.4f}")
                print(f"  • Max: {np.max(values):.4f}")
                print(f"  • Median: {np.median(values):.4f}")
        
        print(f"\nTraining Steps Range: {data[0]['step']} - {data[-1]['step']}")
        print(f"Total Data Points: {len(data)}")
        
        print_metric_stats("Mean Similarity", mean_sims)
        print_metric_stats("Max Similarity", max_sims)
        print_metric_stats("Similarity Standard Deviation", std_sims)
        
        if high_iou_sims:
            print_metric_stats("High IoU Similarity", high_iou_sims)
        else:
            print("\nHigh IoU Similarity: No data available")
            
        if low_iou_sims:
            print_metric_stats("Low IoU Similarity", low_iou_sims)
        else:
            print("\nLow IoU Similarity: No data available")
        
        print("\n" + "="*60)
    
    def print_multi_log_statistics(self, multi_data: List[Tuple[str, List[Dict]]]) -> None:
        """
        Print statistical comparison for multiple log files
        
        Args:
            multi_data: List of tuples (log_name, similarity_data)
        """
        if not multi_data:
            print("[ERROR] No data for statistics!")
            return
        
        print("\n" + "="*80)
        print("[STATS] MULTI-LOG SIMILARITY METRICS COMPARISON")
        print("="*80)
        
        # Create comparison table
        print(f"\n{'Log File':<30} {'Data Points':<12} {'Mean Sim':<12} {'Max Sim':<12} {'Std Dev':<12}")
        print("-" * 80)
        
        for log_name, data in multi_data:
            if not data:
                print(f"{os.path.basename(log_name):<30} {'No data':<12} {'-':<12} {'-':<12} {'-':<12}")
                continue
                
            clean_name = os.path.splitext(os.path.basename(log_name))[0]
            mean_sims = [d['mean_sim'] for d in data]
            max_sims = [d['max_sim'] for d in data]
            std_sims = [d['std_sim'] for d in data]
            
            avg_mean = np.mean(mean_sims)
            avg_max = np.mean(max_sims)
            avg_std = np.mean(std_sims)
            
            print(f"{clean_name:<30} {len(data):<12} {avg_mean:<12.4f} {avg_max:<12.4f} {avg_std:<12.4f}")
        
        print("\n" + "="*80)
        
        # Detailed statistics for each log
        for log_name, data in multi_data:
            if not data:
                continue
                
            clean_name = os.path.splitext(os.path.basename(log_name))[0]
            print(f"\n[CHART] Detailed Statistics for: {clean_name}")
            print("-" * 60)
            
            mean_sims = [d['mean_sim'] for d in data]
            max_sims = [d['max_sim'] for d in data]
            std_sims = [d['std_sim'] for d in data]
            high_iou_sims = [d['high_iou_sim'] for d in data if d['high_iou_sim'] is not None]
            low_iou_sims = [d['low_iou_sim'] for d in data if d['low_iou_sim'] is not None]
            
            def print_metric_stats_compact(name: str, values: List[float]):
                if values:
                    print(f"  {name:<20}: Count={len(values):<4} Mean={np.mean(values):<8.4f} "
                          f"Std={np.std(values):<8.4f} Range=[{np.min(values):.4f}, {np.max(values):.4f}]")
            
            print(f"  Training Steps: {data[0]['step']} - {data[-1]['step']}")
            print_metric_stats_compact("Mean Similarity", mean_sims)
            print_metric_stats_compact("Max Similarity", max_sims)
            print_metric_stats_compact("Std Similarity", std_sims)
            
            if high_iou_sims:
                print_metric_stats_compact("High IoU Sim", high_iou_sims)
            if low_iou_sims:
                print_metric_stats_compact("Low IoU Sim", low_iou_sims)
        
        print("\n" + "="*80)
    
    def export_to_csv(self, data: List[Dict], csv_path: Optional[str] = None) -> str:
        """
        Export similarity data to CSV file
        
        Args:
            data: List of similarity data points
            csv_path: Optional path for CSV file
            
        Returns:
            Path where CSV was saved
        """
        if not data:
            print("[ERROR] No data to export!")
            return ""
        
        if csv_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"similarity_data_{timestamp}.csv"
        
        try:
            import pandas as pd
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)
            print(f"📄 Data exported to CSV: {csv_path}")
            return csv_path
        except ImportError:
            # Fallback to manual CSV writing if pandas not available
            with open(csv_path, 'w') as f:
                # Write header
                f.write("step,mean_sim,max_sim,std_sim,high_iou_sim,low_iou_sim,line_num\n")
                
                # Write data
                for d in data:
                    high_iou = d['high_iou_sim'] if d['high_iou_sim'] is not None else ""
                    low_iou = d['low_iou_sim'] if d['low_iou_sim'] is not None else ""
                    f.write(f"{d['step']},{d['mean_sim']},{d['max_sim']},{d['std_sim']},{high_iou},{low_iou},{d['line_num']}\n")
            
            print(f"📄 Data exported to CSV: {csv_path}")
            return csv_path
    
    def export_multi_log_csv(self, combined_data: List[Dict], csv_path: Optional[str] = None) -> str:
        """
        Export combined multi-log similarity data to CSV file
        
        Args:
            combined_data: List of similarity data points with log_file field
            csv_path: Optional path for CSV file
            
        Returns:
            Path where CSV was saved
        """
        if not combined_data:
            print("[ERROR] No data to export!")
            return ""
        
        if csv_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"multi_log_similarity_data_{timestamp}.csv"
        
        try:
            import pandas as pd
            df = pd.DataFrame(combined_data)
            # Reorder columns to put log_file first
            cols = ['log_file'] + [col for col in df.columns if col != 'log_file']
            df = df[cols]
            df.to_csv(csv_path, index=False)
            print(f"📄 Multi-log data exported to CSV: {csv_path}")
            return csv_path
        except ImportError:
            # Fallback to manual CSV writing if pandas not available
            with open(csv_path, 'w') as f:
                # Write header
                f.write("log_file,step,mean_sim,max_sim,std_sim,high_iou_sim,low_iou_sim,line_num\n")
                
                # Write data
                for d in combined_data:
                    high_iou = d['high_iou_sim'] if d['high_iou_sim'] is not None else ""
                    low_iou = d['low_iou_sim'] if d['low_iou_sim'] is not None else ""
                    log_file = d.get('log_file', 'unknown')
                    f.write(f"{log_file},{d['step']},{d['mean_sim']},{d['max_sim']},{d['std_sim']},{high_iou},{low_iou},{d['line_num']}\n")
            
            print(f"📄 Multi-log data exported to CSV: {csv_path}")
            return csv_path


def main():
    """Main function for command line usage"""
    print("[INFO] Starting similarity log analysis...")
    
    parser = argparse.ArgumentParser(
        description="Analyze similarity metrics from MMDetection training logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single log analysis:
    python analyze_similarity_logs.py --log_file training.log
    python analyze_similarity_logs.py --log_file training.log --output_plot similarity.png
    
  Multi-log comparison:
    python analyze_similarity_logs.py --log_files log1.log log2.log log3.log
    python analyze_similarity_logs.py --log_files log1.log log2.log --output_plot comparison.png
    python analyze_similarity_logs.py --log_files *.log --comparison_mode
        """
    )
    
    # Create mutually exclusive group for single vs multiple files
    log_group = parser.add_mutually_exclusive_group(required=True)
    log_group.add_argument(
        '--log_file', 
        type=str,
        help='Path to a single log file to analyze'
    )
    log_group.add_argument(
        '--log_files', 
        type=str,
        nargs='+',
        help='Paths to multiple log files for comparison analysis'
    )
    
    parser.add_argument(
        '--output_plot', 
        type=str, 
        default=None,
        help='Path to save the output plot (optional)'
    )
    parser.add_argument(
        '--export_csv', 
        type=str, 
        default=None,
        help='Path to export data as CSV (optional)'
    )
    parser.add_argument(
        '--no_plot', 
        action='store_true',
        help='Skip plotting (only show statistics)'
    )
    parser.add_argument(
        '--comparison_mode', 
        action='store_true',
        help='Force comparison mode even for single log (useful with wildcards)'
    )
    
    args = parser.parse_args()
    
    # Determine if we're in multi-log mode
    if args.log_files or args.comparison_mode:
        # Multi-log analysis mode
        log_files = args.log_files if args.log_files else [args.log_file]
        print(f"[INFO] Multi-log comparison mode: analyzing {len(log_files)} log files")
        
        # Validate all log files exist
        for log_file in log_files:
            if not os.path.exists(log_file):
                print(f"[ERROR] Log file '{log_file}' does not exist!")
                return 1
        
        print(f"[SUCCESS] All {len(log_files)} log files found")
        
        # Parse data from all log files
        multi_data = []
        total_data_points = 0
        
        for log_file in log_files:
            print(f"\n[INFO] Analyzing log file: {log_file}")
            analyzer = SimilarityLogAnalyzer(log_file)
            data = analyzer.parse_similarity_data()
            
            if data:
                multi_data.append((log_file, data))
                total_data_points += len(data)
                print(f"[SUCCESS] Found {len(data)} similarity data points in {os.path.basename(log_file)}")
            else:
                print(f"[WARNING] No similarity data found in {os.path.basename(log_file)}")
                multi_data.append((log_file, []))
        
        if total_data_points == 0:
            print("[ERROR] No similarity data found in any of the log files!")
            return 1
        
        print(f"\n[SUCCESS] Total: {total_data_points} similarity data points across {len(multi_data)} files")
        
        # Generate comparison plot (unless disabled)
        if not args.no_plot:
            print("[INFO] Generating multi-log comparison plots...")
            # Use the first analyzer for plotting (they all have the same methods)
            plot_path = analyzer.plot_multi_log_comparison(multi_data, save_path=args.output_plot)
            if plot_path:
                print(f"[SUCCESS] Comparison plot saved to: {plot_path}")
        
        # Print multi-log statistics
        analyzer.print_multi_log_statistics(multi_data)
        
        # Export to CSV if requested (export all data combined)
        if args.export_csv:
            # Combine all data with log file info
            combined_data = []
            for log_file, data in multi_data:
                for item in data:
                    item_copy = item.copy()
                    item_copy['log_file'] = os.path.basename(log_file)
                    combined_data.append(item_copy)
            
            if combined_data:
                csv_path = analyzer.export_multi_log_csv(combined_data, args.export_csv)
                print(f"[SUCCESS] Combined data exported to: {csv_path}")
    
    else:
        # Single log analysis mode
        log_file = args.log_file
        print(f"[INFO] Single log analysis mode: {log_file}")
        
        # Validate log file exists
        if not os.path.exists(log_file):
            print(f"[ERROR] Log file '{log_file}' does not exist!")
            print(f"   Please check the file path and ensure the file exists.")
            return 1
        
        print(f"[SUCCESS] Log file found: {log_file}")
        
        # Create analyzer
        analyzer = SimilarityLogAnalyzer(log_file)
        
        # Parse data
        print(f"[INFO] Analyzing log file: {log_file}")
        data = analyzer.parse_similarity_data()
        
        if not data:
            print("[ERROR] No similarity data found in the log file!")
            print("   Make sure the log file contains lines with 'SIMILARITY_DATA:' entries.")
            return 1
        
        print(f"[SUCCESS] Found {len(data)} similarity data points")
        
        # Generate plot (unless disabled)
        if not args.no_plot:
            print("[INFO] Generating plots...")
            plot_path = analyzer.plot_similarity_metrics(data, save_path=args.output_plot)
            print(f"[SUCCESS] Plot saved to: {plot_path}")
        
        # Print statistics
        analyzer.print_statistics(data)
        
        # Export to CSV if requested
        if args.export_csv:
            csv_path = analyzer.export_to_csv(data, args.export_csv)
            print(f"[SUCCESS] Data exported to: {csv_path}")
    
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        print(f"[ERROR] Error occurred: {e}")
        import traceback
        print("Full error traceback:")
        traceback.print_exc()
        exit(1)