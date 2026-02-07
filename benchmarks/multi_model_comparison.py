#!/usr/bin/env python3
"""
Multi-Model Performance Comparison Script (Updated)
===================================================

Compares TTFT, ITL and throughput across all available models for both TensorRT-LLM and vLLM:
- Gemma 3-1B (TensorRT-LLM & vLLM)
- Llama 3.2-3B (TensorRT-LLM & vLLM) 
- Phi-2 2.7B (TensorRT-LLM & vLLM)
- Qwen 2.5-3B (TensorRT-LLM only)

Generates clean dashboard visualizations with outlier exclusion.
"""

import json
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set up styling
plt.style.use('default')
sns.set_palette("husl")

class ModelDataLoader:
    """Loads performance data from various file formats"""
    
    def __init__(self, analysis_dir: str = "/home/ec2-user/llm_host/LLM-serve/analysis"):
        self.analysis_dir = Path(analysis_dir)
    
    def load_json_data(self, model_name: str, json_path: Path) -> Optional[List[Dict]]:
        """Load data from JSON results file"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data.get('results', [])
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading {json_path}: {e}")
            return None
    
    def parse_markdown_table(self, md_content: str, table_name: str) -> Dict[int, float]:
        """Extract table data from markdown content"""
        pattern = rf"### {table_name}.*?\n\n(.*?)(?=\n###|\n##|\Z)"
        match = re.search(pattern, md_content, re.DOTALL | re.IGNORECASE)
        
        if not match:
            return {}
        
        table_content = match.group(1)
        data = {}
        
        lines = [line.strip() for line in table_content.split('\n') if line.strip() and '|' in line]
        for line in lines[2:]:  # Skip header and separator
            parts = [part.strip() for part in line.split('|') if part.strip()]
            if len(parts) >= 3:
                try:
                    concurrency = int(parts[0])
                    triton_value = float(parts[1])
                    data[concurrency] = triton_value
                except (ValueError, IndexError):
                    continue
        
        return data
    
    def load_qwen_data(self, model_path: Path) -> Optional[List[Dict]]:
        """Load Qwen data from markdown file"""
        md_path = model_path / "BENCHMARK_RESULTS.md"
        if not md_path.exists():
            return None
        
        with open(md_path, 'r') as f:
            content = f.read()
        
        throughput_data = self.parse_markdown_table(content, "Throughput \\(tokens/sec\\)")
        ttft_data = self.parse_markdown_table(content, "Time to First Token \\(ms\\)")
        
        results = []
        for concurrency in sorted(throughput_data.keys()):
            if concurrency in ttft_data:
                results.append({
                    "engine": "Triton-TRT-LLM",
                    "concurrency": concurrency,
                    "throughput_tokens_per_sec": throughput_data[concurrency],
                    "ttft_ms": ttft_data[concurrency],
                    "itl_ms": 0  # Not available in markdown
                })
        
        return results
    
    def load_all_models(self) -> Dict[str, Dict[str, List[Dict]]]:
        """Load data for all available models for both engines from individual folders"""
        models_config = {
            "Gemma 3-1B": {"path": "gemma", "triton_file": "triton_server_results.json", "vllm_file": "vllm_server_results.json"},
            "Llama 3.2-3B": {"path": "llama", "triton_file": "triton_server_results.json", "vllm_file": "vllm_server_results.json"},
            "Phi-2 2.7B": {"path": "phi", "triton_file": "triton_server_results.json", "vllm_file": "vllm_server_results.json"},
            "Qwen 2.5-3B": {"path": "qwen", "triton_file": "triton_server_results.json", "vllm_file": "vllm_server_results.json"}
        }
        
        loaded_data = {"tensorrt": {}, "vllm": {}}
        
        for model_name, config in models_config.items():
            model_path = self.analysis_dir / config["path"]
            
            # Load TensorRT-LLM data - try JSON first, then markdown
            triton_data = None
            json_path = model_path / config["triton_file"]
            if json_path.exists():
                triton_data = self.load_json_data(model_name, json_path)
            else:
                # Try markdown format (for qwen-triton compatibility)
                triton_data = self.load_qwen_data(model_path)
            
            if triton_data:
                loaded_data["tensorrt"][model_name] = triton_data
                print(f"✅ Loaded TensorRT {model_name}: {len(triton_data)} data points")
            
            # Load vLLM data
            if config["vllm_file"]:
                vllm_json_path = model_path / config["vllm_file"]
                vllm_data = self.load_json_data(model_name, vllm_json_path)
                if vllm_data:
                    loaded_data["vllm"][model_name] = vllm_data
                    print(f"✅ Loaded vLLM {model_name}: {len(vllm_data)} data points")
        
        return loaded_data

class MultiModelVisualizer:
    """Creates clean comparison visualizations across multiple models"""
    
    def __init__(self, models_data: Dict[str, Dict[str, List[Dict]]], output_dir: str):
        self.models_data = models_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color scheme for models
        self.colors = {
            "Gemma 3-1B": "#1f77b4",      # Blue
            "Llama 3.2-3B": "#ff7f0e",    # Orange  
            "Phi-2 2.7B": "#2ca02c",      # Green
            "Qwen 2.5-3B": "#d62728"      # Red
        }
    
    def prepare_engine_data(self, engine_data: Dict[str, List[Dict]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare data for engine-specific charts"""
        throughput_data = []
        ttft_data = []
        itl_data = []
        
        for model_name, results in engine_data.items():
            for result in results:
                concurrency = result.get("concurrency", 0)
                
                throughput_data.append({
                    "Model": model_name,
                    "Concurrency": concurrency,
                    "Throughput (tok/s)": result.get("throughput_tokens_per_sec", 0)
                })
                
                ttft_data.append({
                    "Model": model_name,
                    "Concurrency": concurrency,
                    "TTFT (ms)": result.get("ttft_ms", 0)
                })
                
                itl_data.append({
                    "Model": model_name,
                    "Concurrency": concurrency,
                    "ITL (ms)": result.get("itl_ms", 0)
                })
        
        return pd.DataFrame(throughput_data), pd.DataFrame(ttft_data), pd.DataFrame(itl_data)
    
    def _create_clean_subplot(self, df: pd.DataFrame, metric_col: str, ylabel: str, ax, title: str):
        """Create a clean subplot with outlier exclusion (no overlapping labels)"""
        # Identify and exclude outliers
        all_values = df[metric_col].values
        all_values = all_values[~pd.isna(all_values)]
        if len(all_values) == 0:
            return
        
        median_val = np.median(all_values)
        outlier_threshold = median_val * 10
        
        # Filter out outliers
        df_clean = df[df[metric_col] <= outlier_threshold].copy()
        if df_clean.empty:
            return
        
        pivot_clean = df_clean.pivot(index="Concurrency", columns="Model", values=metric_col)
        
        # Create grouped bar chart
        x = np.arange(len(pivot_clean.index))
        bar_width = 0.2
        
        for i, (model, color) in enumerate(self.colors.items()):
            if model in pivot_clean.columns:
                values = pivot_clean[model]
                ax.bar(x + i * bar_width, values, bar_width, 
                      label=model, color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Customize chart
        ax.set_xlabel('Concurrency Level', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(f'{title} Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x + bar_width * (len(self.colors) - 1) / 2)
        ax.set_xticklabels(pivot_clean.index)
        ax.legend(loc='upper left', frameon=True, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add exclusion note if we excluded data
        if len(df) > len(df_clean):
            excluded_count = len(df) - len(df_clean)
            note = f"Note: {excluded_count} outlier(s) excluded for scale clarity"
            ax.text(0.98, 0.02, note, transform=ax.transAxes, fontsize=8,
                    horizontalalignment='right', verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    def create_engine_dashboard(self, engine: str):
        """Create clean dashboard for specific engine"""
        if engine not in self.models_data or not self.models_data[engine]:
            print(f"❌ No data available for {engine}")
            return
        
        engine_data = self.models_data[engine]
        df_throughput, df_ttft, df_itl = self.prepare_engine_data(engine_data)
        
        if df_throughput.empty:
            print(f"❌ No valid data for {engine} dashboard")
            return
        
        # Create 4-panel dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # 1. Throughput (top left)
        self._create_clean_subplot(df_throughput, "Throughput (tok/s)", "Throughput (tokens/sec)", ax1, "Throughput")
        
        # 2. TTFT (top right) - clean version
        self._create_clean_subplot(df_ttft, "TTFT (ms)", "Time to First Token (ms)", ax2, "TTFT")
        
        # 3. ITL (bottom left) - clean version
        self._create_clean_subplot(df_itl, "ITL (ms)", "Inter-Token Latency (ms)", ax3, "ITL")
        
        # 4. Performance summary (bottom right)
        ax4.axis('off')
        
        # Calculate summary stats
        pivot_throughput = df_throughput.pivot(index="Concurrency", columns="Model", values="Throughput (tok/s)")
        pivot_ttft = df_ttft.pivot(index="Concurrency", columns="Model", values="TTFT (ms)")
        pivot_itl = df_itl.pivot(index="Concurrency", columns="Model", values="ITL (ms)")
        
        summary_text = "Performance Summary:\n\n"
        for model in self.colors.keys():
            if model in pivot_throughput.columns:
                max_throughput = pivot_throughput[model].max() if not pivot_throughput[model].isna().all() else 0
                min_ttft = pivot_ttft[model].min() if model in pivot_ttft.columns and not pivot_ttft[model].isna().all() else 0
                min_itl = pivot_itl[model].min() if model in pivot_itl.columns and not pivot_itl[model].isna().all() else 0
                
                summary_text += f"{model}:\n"
                summary_text += f"  Peak: {max_throughput:.0f} tok/s\n"
                if min_ttft > 0:
                    summary_text += f"  Min TTFT: {min_ttft:.1f} ms\n"
                if min_itl > 0:
                    summary_text += f"  Min ITL: {min_itl:.1f} ms\n"
                summary_text += "\n"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        ax4.set_title('Performance Summary', fontsize=12, fontweight='bold')
        
        # Engine-specific title
        engine_name = "TensorRT-LLM" if engine == "tensorrt" else "vLLM"
        fig.suptitle(f'{engine_name} Multi-Model Performance Dashboard\nGemma vs Llama vs Phi vs Qwen', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        
        # Use the main dashboard filename for tensorrt, specific for vllm
        if engine == "tensorrt":
            filename = "multi_model_dashboard.png"
        else:
            filename = f"multi_model_dashboard_{engine}.png"
        
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ {engine_name} dashboard saved to: {self.output_dir}/{filename}")
    
    def create_individual_charts(self, engine: str):
        """Create separate PNG files for each metric"""
        if engine not in self.models_data or not self.models_data[engine]:
            print(f"❌ No data available for {engine}")
            return
        
        engine_data = self.models_data[engine]
        df_throughput, df_ttft, df_itl = self.prepare_engine_data(engine_data)
        
        if df_throughput.empty:
            print(f"❌ No valid data for {engine} individual charts")
            return
        
        engine_name = "TensorRT-LLM" if engine == "tensorrt" else "vLLM"
        
        # 1. Throughput chart
        self._create_individual_chart(df_throughput, "Throughput (tok/s)", "Throughput (tokens/sec)", 
                                    f"{engine_name} Throughput Comparison", 
                                    f"throughput_comparison_{engine}.png")
        
        # 2. TTFT chart
        self._create_individual_chart(df_ttft, "TTFT (ms)", "Time to First Token (ms)", 
                                    f"{engine_name} TTFT Comparison", 
                                    f"ttft_comparison_{engine}.png")
        
        # 3. ITL chart
        if not df_itl.empty and df_itl["ITL (ms)"].sum() > 0:  # Only if we have ITL data
            self._create_individual_chart(df_itl, "ITL (ms)", "Inter-Token Latency (ms)", 
                                        f"{engine_name} ITL Comparison", 
                                        f"itl_comparison_{engine}.png")
    
    def _create_individual_chart(self, df: pd.DataFrame, metric_col: str, ylabel: str, title: str, filename: str):
        """Create an individual chart for a specific metric"""
        # Identify and exclude outliers
        all_values = df[metric_col].values
        all_values = all_values[~pd.isna(all_values)]
        if len(all_values) == 0:
            return
        
        median_val = np.median(all_values)
        outlier_threshold = median_val * 10
        
        # Filter out outliers
        df_clean = df[df[metric_col] <= outlier_threshold].copy()
        if df_clean.empty:
            return
        
        pivot_clean = df_clean.pivot(index="Concurrency", columns="Model", values=metric_col)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create grouped bar chart
        x = np.arange(len(pivot_clean.index))
        bar_width = 0.2
        
        for i, (model, color) in enumerate(self.colors.items()):
            if model in pivot_clean.columns:
                values = pivot_clean[model]
                ax.bar(x + i * bar_width, values, bar_width, 
                      label=model, color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Customize chart
        ax.set_xlabel('Concurrency Level', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x + bar_width * (len(self.colors) - 1) / 2)
        ax.set_xticklabels(pivot_clean.index)
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add exclusion note if we excluded data
        if len(df) > len(df_clean):
            excluded_count = len(df) - len(df_clean)
            note = f"Note: {excluded_count} outlier(s) excluded for scale clarity"
            ax.text(0.98, 0.02, note, transform=ax.transAxes, fontsize=9,
                    horizontalalignment='right', verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Individual chart saved: {self.output_dir}/{filename}")
    
    def create_performance_summary_table(self):
        """Create a summary performance table for both engines"""
        summary_data = []
        
        for engine, engine_data in self.models_data.items():
            if not engine_data:
                continue
                
            engine_name = "TensorRT-LLM" if engine == "tensorrt" else "vLLM"
            df_throughput, df_ttft, df_itl = self.prepare_engine_data(engine_data)
            
            for model in engine_data.keys():
                model_throughput = df_throughput[df_throughput['Model'] == model]
                model_ttft = df_ttft[df_ttft['Model'] == model]
                
                if not model_throughput.empty:
                    max_throughput = model_throughput['Throughput (tok/s)'].max()
                    min_ttft = model_ttft['TTFT (ms)'].min() if not model_ttft.empty else 0
                    
                    summary_data.append({
                        'Engine': engine_name,
                        'Model': model,
                        'Peak Throughput': f"{max_throughput:.0f} tok/s",
                        'Min TTFT': f"{min_ttft:.1f} ms" if min_ttft > 0 else "N/A",
                    })
        
        if not summary_data:
            print("❌ No data for performance summary")
            return
        
        # Create table visualization
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        df_summary = pd.DataFrame(summary_data)
        table = ax.table(cellText=df_summary.values, 
                        colLabels=df_summary.columns,
                        cellLoc='center',
                        loc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Color the header
        for i in range(len(df_summary.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows alternately
        for i in range(1, len(df_summary) + 1):
            for j in range(len(df_summary.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title('Performance Summary: Multi-Engine Models Comparison\n', 
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_summary_table.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Performance summary table saved to: {self.output_dir}/performance_summary_table.png")
        
        # Save as CSV
        csv_path = self.output_dir / "performance_summary.csv"
        df_summary.to_csv(csv_path, index=False)
        print(f"✅ Performance summary CSV saved to: {csv_path}")
        
        return df_summary

def main():
    """Main execution function"""
    print("🚀 Multi-Model Performance Comparison (Updated)")
    print("=" * 60)
    
    # Initialize data loader
    loader = ModelDataLoader()
    models_data = loader.load_all_models()
    
    if not models_data or (not models_data.get("tensorrt") and not models_data.get("vllm")):
        print("❌ No model data found!")
        return
    
    print(f"\n📊 TensorRT Models: {len(models_data.get('tensorrt', {}))}")
    print(f"📊 vLLM Models: {len(models_data.get('vllm', {}))}")
    
    # Initialize visualizer
    output_dir = "/home/ec2-user/llm_host/LLM-serve/analysis/all_models"
    visualizer = MultiModelVisualizer(models_data, output_dir)
    
    print("\n🎨 Creating dashboards...")
    
    # Create TensorRT-LLM dashboard (main dashboard)
    if models_data.get("tensorrt"):
        visualizer.create_engine_dashboard("tensorrt")
        visualizer.create_individual_charts("tensorrt")
    
    # Create vLLM dashboard
    if models_data.get("vllm"):
        visualizer.create_engine_dashboard("vllm")
        visualizer.create_individual_charts("vllm")
    
    # Create summary table
    print("\n📋 Creating performance summary...")
    visualizer.create_performance_summary_table()
    
    print(f"\n🎉 All visualizations completed!")
    print(f"📁 Results saved in: {output_dir}")

if __name__ == "__main__":
    main()