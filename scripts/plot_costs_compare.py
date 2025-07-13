# SPDX-FileCopyrightText: : 2022 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT

"""
Compares costs between a base scenario and two alternative scenarios,
generating a combined, publication-quality plot entirely in English.
"""

import logging
import math
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import yaml

logger = logging.getLogger(__name__)

# --- Configuration ---
# Set the version numbers to compare
VERSION_BASE = "0711.4H.10.1"
VERSION_COMP1 = "0711.4H.10.2"
VERSION_COMP2 = "0711.4H.10.3"

# --- Matplotlib settings for English plot ---
plt.rcParams['font.family'] = 'Arial' # Use a standard English font
plt.rcParams['axes.unicode_minus'] = False


def format_legend_label(label):
    """
    Shortens long legend labels for better readability and removes 'marginal'.
    Example: 'variable cost-non-renewable' -> 'Var. cost non-renew.'
             'marginal-solar' -> 'solar'
    """
    label = str(label)
    # <<< MODIFIED SECTION START >>>
    replacements = {
        'marginal': '',  # Remove 'marginal' from labels
        'variable cost': 'Var. cost',
        'capital': 'Cap.',
        'non-renewable': 'non-renew.',
        'renewable': 'renew.',
        'demand side': 'demand-side',
        'transmission lines': 'Transmission',
        'long-duration storages': 'LD Storage',
        'batteries': 'Batteries',
        'carbon capture': 'Carbon Capture',
        'carbon management': 'Carbon Mgmt.',
        ' products': '' # Remove for industry types if they exist
    }
    # <<< MODIFIED SECTION END >>>
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label.replace('-', ' ').strip()


def generate_yearly_comparison_plots(name_base, name_comp1, name_comp2, output_dir):
    """
    Generates a side-by-side comparison plot with a right-side legend and tighter bars.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Collecting data for comparison: {name_base} vs {name_comp1}")
    years_data1 = collect_all_years_data(name_base, name_comp1, 'costs')
    
    logger.info(f"Collecting data for comparison: {name_base} vs {name_comp2}")
    years_data2 = collect_all_years_data(name_base, name_comp2, 'costs')

    if not years_data1 and not years_data2:
        logger.error("No data found for either comparison. Skipping plot generation.")
        return

    # 压缩图片宽度，主体偏左，右侧留出图例空间
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8), sharey=True, gridspec_kw={'width_ratios': [1, 1]})

    all_handles, all_labels = [], []

    # Prepare names for the plot titles
    name_base_plot = "Case 1"
    name_comp1_plot = "Case 2"
    name_comp2_plot = "Case 3"

    # 保存 Case 1 → Case 2 的数据到 CSV
    if years_data1:
        processed_data1, net_changes1, years1 = process_cost_change_data(years_data1, name_base, name_comp1)
        if processed_data1:
            df_rows = []
            for i, year in enumerate(years1):
                for cat, data in processed_data1.items():
                    if i < len(data['changes']):
                        df_rows.append({
                            'case': f'{name_base_plot} → {name_comp1_plot}',
                            'year': year,
                            'category': cat,
                            'cost_saving_Billion_CNY': round(data['changes'][i]/1e9, 4)
                        })
            df_save = pd.DataFrame(df_rows)
            save_path = output_dir / 'data' / f'cost_savings_case1_vs_case2.csv'
            df_save.to_csv(save_path, index=False)
            draw_single_cost_change_plot(ax1, processed_data1, net_changes1, years1, name_base_plot, name_comp1_plot)
            handles, labels = ax1.get_legend_handles_labels()
            all_handles.extend(handles)
            all_labels.extend(labels)

    # 保存 Case 1 → Case 3 的数据到 CSV
    if years_data2:
        processed_data2, net_changes2, years2 = process_cost_change_data(years_data2, name_base, name_comp2)
        if processed_data2:
            df_rows = []
            for i, year in enumerate(years2):
                for cat, data in processed_data2.items():
                    if i < len(data['changes']):
                        df_rows.append({
                            'case': f'{name_base_plot} → {name_comp2_plot}',
                            'year': year,
                            'category': cat,
                            'cost_saving_Billion_CNY': round(data['changes'][i]/1e9, 4)
                        })
            df_save = pd.DataFrame(df_rows)
            save_path = output_dir / 'data' / f'cost_savings_case1_vs_case3.csv'
            df_save.to_csv(save_path, index=False)
            draw_single_cost_change_plot(ax2, processed_data2, net_changes2, years2, name_base_plot, name_comp2_plot)
            handles, labels = ax2.get_legend_handles_labels()
            all_handles.extend(handles)
            all_labels.extend(labels)

    # 创建图例，放在右侧，字体略微缩小，贴近图像
    if all_handles:
        by_label = dict(zip(all_labels, all_handles))
        formatted_labels = [format_legend_label(l) for l in by_label.keys()]
        # 右侧图例，字体略微缩小，贴近图像
        fig.legend(by_label.values(), formatted_labels,
                  loc='center left',
                  bbox_to_anchor=(0.89, 0.5),  # 更贴近图像主体
                  ncol=1,
                  fontsize=12,
                  frameon=True)
        # 调整子图布局，主体更偏左，右侧空间更紧凑
        fig.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.1, wspace=0.13)

    # Save the plot
    plot_file = plots_dir / f"cost_change_comparison_EN_{name_base}_vs_{name_comp1}_and_{name_comp2}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"English, publication-style plot saved to: {plot_file}")
    plt.close()


def draw_single_cost_change_plot(ax, filtered_categories, net_changes_cny, years, name1, name2):
    """
    Draws a single cost change plot on a given axes object (in English).
    """
    # 减小柱宽度和间距，使图更紧凑
    bar_width = 0.35
    x = np.arange(len(years))
    
    # 自定义颜色映射，为不同类别指定更加区分的颜色
    custom_colors = {
        'variable cost-non-renewable': '#e41a1c',  # 红色
        'capital-non-renewable': '#ff7f00',        # 橙色
        'capital–renewable': '#4daf4a',            # 绿色
        'coal cc': '#a65628',                      # 棕色
        'capital–demand side': '#984ea3',          # 紫色
        'transmission lines': '#377eb8',           # 蓝色
        'batteries': '#f781bf',                    # 粉色
        'long-duration storages': '#a65628',       # 棕色
        'carbon capture': '#808080',               # 灰色
        'carbon management': "#7602C4",            # 深紫色
        'synthetic fuels': '#ffff33',              # 黄色
        'Net Change': '#000000',                   # 黑色，用于净变化线
    }
    
    # 为未定义的类别创建区分度更高的颜色
    all_possible_carriers = sorted(filtered_categories.keys())

    # 创建高对比度颜色列表
    distinct_colors = [
        '#42d4f4', '#bfef45', '#fabed4', 
        '#469990', '#dcbeff',  
        '#aaffc3', '#808000', '#ffd8b1',
    ]

    # 如果类别多于预定义颜色，则添加一组使用tab20和Set3组合的颜色
    if len(all_possible_carriers) > len(distinct_colors):
        # 使用tab20和Set3颜色映射组合
        extra_colors_tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
        extra_colors_set3 = plt.cm.Set3(np.linspace(0, 1, 12))
        # 交替添加以增加差异
        for i in range(min(20, 12)):
            if i < 20:
                distinct_colors.append(tuple(extra_colors_tab20[i]))
            if i < 12:
                distinct_colors.append(tuple(extra_colors_set3[i]))

    # 如果仍然不够，使用hsv颜色空间以最大化色相差异
    if len(all_possible_carriers) > len(distinct_colors):
        remaining_needed = len(all_possible_carriers) - len(distinct_colors)
        hsv_colors = plt.cm.hsv(np.linspace(0, 1, remaining_needed))
        distinct_colors.extend([tuple(c) for c in hsv_colors])

    # 将自定义颜色和高对比度颜色组合
    carrier_colors = {}
    undefined_carriers = []
    for i, carrier in enumerate(all_possible_carriers):
        if carrier in custom_colors:
            carrier_colors[carrier] = custom_colors[carrier]
        else:
            undefined_carriers.append(carrier)

    # 为未定义类别分配高对比度颜色
    for i, carrier in enumerate(undefined_carriers):
        color_index = i % len(distinct_colors)
        carrier_colors[carrier] = distinct_colors[color_index]

    bottom_positive = np.zeros(len(x))
    bottom_negative = np.zeros(len(x))

    for carrier in all_possible_carriers:
        data = filtered_categories.get(carrier)
        if not data: continue

        # 只有绝对值超过1B的类别才显示图例
        show_legend = any(abs(c) > 1e9 for c in data['changes'])
        legend_label = carrier if show_legend else None

        full_changes = np.zeros(len(years))
        for i, year_val in enumerate(data['years']):
            if year_val in years:
                full_changes[years.index(year_val)] = data['changes'][i]

        positive_changes = np.array([max(0, c) for c in full_changes])
        negative_changes = np.array([min(0, c) for c in full_changes])

        if np.any(positive_changes > 0):
            ax.bar(x, positive_changes, bottom=bottom_positive, label=legend_label,
                   color=carrier_colors[carrier], alpha=0.85, width=bar_width)
            bottom_positive += positive_changes

        if np.any(negative_changes < 0):
            ax.bar(x, negative_changes, bottom=bottom_negative, label=legend_label,
                   color=carrier_colors[carrier], alpha=0.85, width=bar_width)
            bottom_negative += negative_changes

    ax.plot(x, net_changes_cny, 'k-', linewidth=2.5, label='Net Change', marker='o', markersize=8, zorder=20)
    
    # 放大标注字体
    for i, net_change in enumerate(net_changes_cny):
        if abs(net_change) > 1e3:
            ax.annotate(f'{net_change/1e9:.1f}B', xy=(i, net_change),
                        xytext=(0, 10 if net_change >= 0 else -20),
                        textcoords="offset points", ha='center',
                        va='bottom' if net_change >= 0 else 'top',
                        fontsize=12,  # 放大标注字体
                        weight='bold', color='black',
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec='none'))

    # 放大坐标轴标签和标题
    ax.set_xlabel('Year', fontsize=13)  # 增大字体
    ax.set_ylabel('Cost Savings (Billion CNY)', fontsize=13)  # 增大字体
    ax.set_title(f'{name1} → {name2}', fontsize=20)  # 放大标题字体
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=11)  # 增大x轴刻度标签
    ax.tick_params(axis='both', which='major', labelsize=11)  # 增大y轴刻度标签
    
    y_ticks = ax.get_yticks()
    ax.set_yticklabels([f'{tick/1e9:.1f}B' for tick in y_ticks])
    
    ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def find_summary_files(summary_dir):
    """
    Recursively finds all CSV files in a summary directory.
    """
    summary_path = Path(summary_dir)
    csv_files = []
    if summary_path.exists():
        csv_files = list(summary_path.rglob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files in {summary_dir}")
    else:
        logger.warning(f"Summary directory does not exist: {summary_dir}")
    return csv_files


# <<< FIX: Restored the missing load_single_csv_file function >>>
def load_single_csv_file(file_path):
    """
    Loads a single CSV file, handling potential multi-index structures.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_lines = [f.readline().strip() for _ in range(5)]
        
        has_multiindex = any(',' in line and line.split(',')[1] == '' for line in first_lines)
        
        if has_multiindex:
            df = pd.read_csv(file_path, header=None)
            if len(df.columns) >= 4:
                df.set_index([0, 1, 2], inplace=True)
                df.columns = [df.columns[0]]
                df[df.columns[0]] = pd.to_numeric(df[df.columns[0]], errors='coerce')
            else:
                df = pd.read_csv(file_path, index_col=[0, 1])
                df[df.columns[0]] = pd.to_numeric(df[df.columns[0]], errors='coerce')
        else:
            df = pd.read_csv(file_path, index_col=0)
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        logger.warning(f"Error loading {file_path}: {e}")
        return None


def load_summary_data(summary_dir):
    """
    Loads all summary CSV files from a directory into a dictionary.
    """
    summary_data = {}
    csv_files = find_summary_files(summary_dir)
    if not csv_files:
        return summary_data
    for csv_file in csv_files:
        df = load_single_csv_file(csv_file)
        if df is not None:
            summary_data[csv_file.stem] = df
    return summary_data


def process_cost_change_data(years_data, name1, name2):
    """
    Processes cost data to calculate changes by category.
    """
    years = sorted(years_data.keys())
    # This mapping defines how raw cost items are grouped into plot categories.
    cost_category_mapping = {
        ('marginal', 'coal'): 'variable cost-non-renewable', ('marginal', 'coal power plant'): 'variable cost-non-renewable',
        ('marginal', 'coal cc'): 'coal cc', ('marginal', 'gas'): 'variable cost-non-renewable',
        ('marginal', 'nuclear'): 'variable cost-non-renewable', ('marginal', 'CHP coal'): 'variable cost-non-renewable',
        ('marginal', 'CHP gas'): 'variable cost-non-renewable', ('marginal', 'OCGT gas'): 'variable cost-non-renewable',
        ('marginal', 'coal boiler'): 'variable cost-non-renewable', ('marginal', 'gas boiler'): 'variable cost-non-renewable',
        ('capital', 'coal'): 'capital-non-renewable', ('capital', 'coal power plant'): 'capital-non-renewable',
        ('capital', 'coal cc'): 'coal cc', ('capital', 'gas'): 'capital-non-renewable',
        ('capital', 'nuclear'): 'capital-non-renewable', ('capital', 'CHP coal'): 'capital-non-renewable',
        ('capital', 'CHP gas'): 'capital-non-renewable', ('capital', 'OCGT gas'): 'capital-non-renewable',
        ('capital', 'coal boiler'): 'capital-non-renewable', ('capital', 'gas boiler'): 'capital-non-renewable',
        ('capital', 'heat pump'): 'capital–demand side', ('capital', 'resistive heater'): 'capital–demand side',
        ('capital', 'hydro_inflow'): 'capital–renewable', ('capital', 'hydroelectricity'): 'capital–renewable',
        ('capital', 'offwind'): 'capital–renewable', ('capital', 'onwind'): 'capital–renewable',
        ('capital', 'solar'): 'capital–renewable', ('capital', 'solar thermal'): 'capital–renewable',
        ('capital', 'biomass'): 'capital–renewable', ('capital', 'biogas'): 'capital–renewable',
        ('capital', 'AC'): 'transmission lines', ('capital', 'stations'): 'transmission lines',
        ('capital', 'battery'): 'batteries', ('capital', 'battery discharger'): 'batteries',
        ('marginal', 'battery'): 'batteries', ('marginal', 'battery discharger'): 'batteries',
        ('capital', 'PHS'): 'long-duration storages', ('capital', 'water tanks'): 'long-duration storages',
        ('capital', 'H2'): 'long-duration storages', ('capital', 'H2 CHP'): 'long-duration storages',
        ('marginal', 'PHS'): 'long-duration storages', ('marginal', 'water tanks'): 'long-duration storages',
        ('marginal', 'H2'): 'long-duration storages', ('marginal', 'H2 CHP'): 'long-duration storages',
        ('capital', 'CO2 capture'): 'carbon capture', ('marginal', 'CO2 capture'): 'carbon capture',
        ('capital', 'Sabatier'): 'synthetic fuels', ('marginal', 'Sabatier'): 'synthetic fuels',
        ('capital', 'CO2'): 'carbon management', ('marginal', 'CO2'): 'carbon management',
    }
    category_changes, net_changes = {}, []
    for year in years:
        df1, df2 = years_data[year][name1], years_data[year][name2]
        year_net_change = 0
        all_indices = df1.index.union(df2.index)
        for idx in all_indices:
            if len(idx) < 3: continue
            _, cost_type, carrier = idx[0], idx[1], idx[2]
            category_name = cost_category_mapping.get((cost_type, carrier), f"{cost_type}-{carrier}")
            if category_name not in category_changes: category_changes[category_name] = {'years': [], 'changes': []}
            v1 = df1.loc[idx].iloc[0] if idx in df1.index else 0
            v2 = df2.loc[idx].iloc[0] if idx in df2.index else 0
            change = (0 if pd.isna(v1) else v1) - (0 if pd.isna(v2) else v2) # Positive change = cost saving
            if year not in category_changes[category_name]['years']:
                category_changes[category_name]['years'].append(year)
                category_changes[category_name]['changes'].append(0)
            category_changes[category_name]['changes'][category_changes[category_name]['years'].index(year)] += change
            year_net_change += change
        net_changes.append(year_net_change)
    
    # Exclude marginal categories that are often zero or negligible
    exclude_categories = {
        'nan', 'nan-nan', 'synthetic fuels', 'marginal-renewable', 'marginal-heat pump', 
        'marginal-resistive heater', 'marginal-onwind', 'marginal-offwind', 'marginal-solar', 
        'marginal-solar thermal', 'marginal-biomass', 'marginal-biogas', 'marginal-H2', 'marginal-H2 CHP',
    }
    EUR_TO_CNY = 7.55
    filtered_categories = {}
    for cat, data in category_changes.items():
        if cat in exclude_categories or pd.isna(cat): continue
        if any(abs(c) > 1e-6 for c in data['changes']):
            data['changes'] = [c * EUR_TO_CNY for c in data['changes']]
            filtered_categories[cat] = data
            
    return filtered_categories, [nc * EUR_TO_CNY for nc in net_changes], years


def collect_all_years_data(name1, name2, file_type):
    """
    Collects data for two versions across all specified years.
    """
    years_data = {}
    years = [2025, 2030, 2035, 2040, 2045, 2050]
    pathway = "linear2050" # Default pathway
    try:
        with open("config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        pathway = config.get('scenario', {}).get('pathway', 'linear2050')
        if isinstance(pathway, list): pathway = pathway[0]
    except FileNotFoundError:
        logger.warning("config.yaml not found, using default pathway 'linear2050'.")
    except Exception as e:
        logger.warning(f"Error reading config.yaml, using default pathway. Error: {e}")

    for year in years:
        dir_pattern = f"postnetwork-ll-current+Neighbor-{pathway}-{year}"
        v1_dir = Path(f"results/version-{name1}/summary/postnetworks/positive/{dir_pattern}")
        v2_dir = Path(f"results/version-{name2}/summary/postnetworks/positive/{dir_pattern}")
        if v1_dir.exists() and v2_dir.exists():
            f1, f2 = v1_dir / f"{file_type}.csv", v2_dir / f"{file_type}.csv"
            if f1.exists() and f2.exists():
                df1, df2 = load_single_csv_file(f1), load_single_csv_file(f2)
                if df1 is not None and df2 is not None:
                    years_data[year] = {name1: df1, name2: df2}
    if not years_data:
        logger.warning(f"No matching year data found between {name1} and {name2}.")
    return years_data


def main():
    """
    Main function to run the comparison script.
    """
    parser = argparse.ArgumentParser(description='Compare a base version with two alternative versions.')
    parser.add_argument('--base', help=f'Base version ID (default: {VERSION_BASE})')
    parser.add_argument('--comp1', help=f'First comparison version ID (default: {VERSION_COMP1})')
    parser.add_argument('--comp2', help=f'Second comparison version ID (default: {VERSION_COMP2})')
    parser.add_argument('--output', default='results/comparison_results', help='Output directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--results-dir', default='results', help='Main results directory')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='%(levelname)s: %(message)s')
    
    version_base = args.base or VERSION_BASE
    version_comp1 = args.comp1 or VERSION_COMP1
    version_comp2 = args.comp2 or VERSION_COMP2

    logger.info(f"Starting comparison:\n- Base: {version_base}\n- Comp 1: {version_comp1}\n- Comp 2: {version_comp2}")

    data_base = load_summary_data(f"{args.results_dir}/version-{version_base}/summary")
    data_comp1 = load_summary_data(f"{args.results_dir}/version-{version_comp1}/summary")
    data_comp2 = load_summary_data(f"{args.results_dir}/version-{version_comp2}/summary")

    if not data_base:
        logger.error(f"Failed to load data for base version {version_base}. Aborting.")
        return

    if data_comp1 or data_comp2:
        generate_yearly_comparison_plots(version_base, version_comp1, version_comp2, Path(args.output))
    else:
        logger.warning("Could not load data for any comparison versions. Skipping plot generation.")

    logger.info("Comparison complete.")

if __name__ == "__main__":
    main()