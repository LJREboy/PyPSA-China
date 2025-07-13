import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import matplotlib.colors as mcolors

def plot_industry_energy_use(version_list, year="2050", output_file=None):
    """
    绘制工业负荷实际用能的月度柱状图
    
    参数:
    version_list: 版本号列表
    year: 年份，默认2050年
    output_file: 输出文件路径
    """
    # 设置图表风格
    plt.style.use('ggplot')
    plt.rcParams['font.family'] = 'Arial'
    
    # 定义月份标签
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # 读取产品信息csv，筛选出产品材料
    prod_param_path = Path("data/loadshedding/production_parameters_forLS_modified.csv")
    product_materials = set()
    if prod_param_path.exists():
        prod_df = pd.read_csv(prod_param_path)
        # 只选type不是raw_material的材料
        product_materials = set(prod_df.loc[prod_df['type'] != 'raw_material', 'materials'].astype(str))
    else:
        print(f"警告: 未找到产品参数文件 {prod_param_path}")

    # 首先收集所有版本中的材料类型（只收集产品材料）
    all_materials = set()
    material_data_by_version = {}

    for version in version_list:
        print(f"预处理版本 {version}...")
        # 参考 make_summary.py 的路径拼接方式
        # 默认只处理电网主文件（elec_s_{year}.nc），如需 postnetwork 可调整
        net_path = Path(f"results/version-{version}/networks/elec_s_{year}.nc")
        # 如果主文件不存在，尝试查找 postnetwork 文件（如 make_summary）
        if not net_path.exists():
            # 兼容 postnetwork 路径（如 make_summary）
            postnet_dir = Path(f"results/version-{version}/postnetworks")
            # 搜索所有 postnetwork 文件，优先选第一个
            postnet_files = list(postnet_dir.glob(f"**/postnetwork-*-*-*-{year}.nc"))
            if postnet_files:
                net_path = postnet_files[0]
                print(f"自动选用 postnetwork 文件: {net_path}")
            else:
                print(f"警告: 未找到网络文件 {net_path} 或 postnetwork 文件")
                continue
        n = pypsa.Network(str(net_path))
        # 识别工业负荷切除发电机
        industrial_gens = [name for name in n.generators.index if "industrial load shedding" in name]
        if not industrial_gens:
            print(f"版本 {version} 中未找到工业负荷切除发电机")
            continue
        material_pattern = re.compile(r'([A-Za-z_]+) industrial load shedding')
        version_materials = set()
        for gen_name in industrial_gens:
            match = material_pattern.search(gen_name)
            if match:
                material = match.group(1).split()[-1]
                # 只收集产品材料
                if material in product_materials:
                    version_materials.add(material)
                    all_materials.add(material)
        material_data_by_version[version] = {
            'materials': list(version_materials),
            'network': n,
            'generators': industrial_gens
        }

    # 将所有产品材料转为列表并排序，以确保一致的顺序
    all_materials = sorted(list(all_materials))
    print(f"所有识别出的产品材料类型: {all_materials}")

    # 定义高对比度的颜色映射 - 使用具有强对比度的颜色
    distinct_colors = [
        '#1f77b4',  # 蓝色
        '#d62728',  # 红色
        '#2ca02c',  # 绿色
        '#9467bd',  # 紫色
        '#ff7f0e',  # 橙色
        '#8c564b',  # 棕色
        '#e377c2',  # 粉色
        '#7f7f7f',  # 灰色
        '#bcbd22',  # 黄绿色
        '#17becf',  # 青色
        '#f0027f',  # 深粉色
        '#bf5b17',  # 褐色
        '#386cb0',  # 深蓝色
        '#66a61e',  # 草绿色
        '#666666'   # 深灰色
    ]
    while len(distinct_colors) < len(all_materials):
        distinct_colors.extend(distinct_colors)
    color_map = {material: distinct_colors[i] for i, material in enumerate(all_materials)}
    
    # 创建图表和子图，支持3个版本，缩减横向宽度
    fig_width = 2.6 * len(version_list)  # 更紧凑
    fig_height = 5.2  # 纵向略高
    fig, axes = plt.subplots(1, len(version_list), figsize=(fig_width, fig_height), sharey=True)
    if len(version_list) == 1:
        axes = [axes]
    
    # 创建一个空列表来收集图例句柄和标签
    legend_handles = []
    legend_labels = []

    bar_width = 0.7  # 适中宽度

    legend_threshold = 1e3  # 只显示年总能量超过1TWh的产品图例
    product_total_energy = {}
    for i, version in enumerate(version_list):
        if version not in material_data_by_version:
            continue

        data = material_data_by_version[version]
        n = data['network']
        industrial_gens = data['generators']
        materials = data['materials']

        freq_hours = (n.snapshot_weightings.sum().iloc[0]) / len(n.snapshots)
        material_data = {}
        baseline_data = {}

        if version == "0711.4H.10.3":
            # 用 link 的 p0 作为实际用能
            for material in all_materials:
                link_name = f"Anhui {material} production"
                if link_name in n.links.index:
                    p0 = n.links_t.p0[link_name]
                    energy = p0 * freq_hours
                    material_data[material] = energy
                    # 基准线用 link 的 p_nom
                    p_nom = n.links.at[link_name, 'p_nom'] if 'p_nom' in n.links.columns else None
                    if p_nom is not None:
                        baseline_data[material] = np.full(len(n.snapshots), p_nom * freq_hours)
                    print(f"0711.4H.10.3: 产品 {material} link输入功率已提取，freq_hours={freq_hours}")
        else:
            # 用 generator 的 (p_nom-load_shed) 作为实际用能
            for gen_name in industrial_gens:
                match = material_pattern.search(gen_name)
                if match:
                    material = match.group(1).split()[-1]
                    p_nom = n.generators.at[gen_name, 'p_nom']
                    load_shed = n.generators_t.p[gen_name]
                    actual_load = (p_nom - load_shed) * freq_hours
                    material_data[material] = actual_load
                    baseline_data[material] = np.full(len(n.snapshots), p_nom * freq_hours)
                    print(f"成功处理产品 {material}, p_nom = {p_nom:.2f} MW, freq_hours={freq_hours}")

        all_load_data = pd.DataFrame(index=n.snapshots)
        baseline_load_data = pd.DataFrame(index=n.snapshots)
        for material, data_arr in material_data.items():
            all_load_data[material] = data_arr
        for material, data_arr in baseline_data.items():
            baseline_load_data[material] = data_arr
        all_load_data['month'] = all_load_data.index.month
        monthly_load = all_load_data.groupby('month').sum() / 1000
        baseline_load_data['month'] = baseline_load_data.index.month
        monthly_baseline = baseline_load_data.groupby('month').sum().sum(axis=1) / 1000

        # 统计年总能量
        product_total_energy[version] = monthly_load.sum()

        # 绘制柱状图
        ax = axes[i]
        bottom = np.zeros(12)
        x_pos = np.arange(12)
        for material in all_materials:
            if material in monthly_load.columns:
                bar = ax.bar(x_pos, monthly_load[material], bottom=bottom,
                             color=color_map[material], alpha=0.8, width=bar_width, align='center')
                if i == 0 and product_total_energy[version][material] > legend_threshold:
                    legend_handles.append(bar)
                    legend_labels.append(material)
                bottom += monthly_load[material]
        ax.set_xlim(-0.5, 11.5)
        # 绘制基准线（每月总p_nom能量）
        if monthly_baseline is not None and len(monthly_baseline) == 12:
            ax.plot(np.arange(12), monthly_baseline.values, '--', color='black', label='Baseline (p_nom)', linewidth=2)
            if i == 0:
                legend_handles.append(ax.lines[-1])
                legend_labels.append('E_max')
        ax.set_title(f'Case {i+1}', fontsize=14)
        xtick_pos = np.arange(12)
        xtick_labels = [m if (idx % 4 == 0) else '' for idx, m in enumerate(month_labels)]
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_labels, rotation=0)
        if i == 0:
            ax.set_ylabel('Energy Usage (GWh)', fontsize=12)
        ax.set_xlabel('Month', fontsize=12)
    
    # 添加单独的图例 - 不使用子图的图例
    fig.legend(legend_handles, legend_labels, 
              loc='upper center', 
              bbox_to_anchor=(0.5, 0.05), 
              ncol=min(len(all_materials), 4), 
              fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # 为图例留出更多空间
    
    # 保存图表
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {output_file}")
    else:
        plt.show()
        
    return fig

if __name__ == "__main__":
    # 指定要比较的版本
    versions = ["0711.4H.10.1", "0711.4H.10.2", "0711.4H.10.3"]
    
    # 设置输出路径
    output_dir = Path("results/comparison_results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "industry_energy_use_monthly_2050.png"
    
    # 绘制图表
    fig = plot_industry_energy_use(versions, "2050", output_file)