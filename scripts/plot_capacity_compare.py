import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime

def plot_capacity_compare(
    version_list,
    config_path="config.yaml",
    pathway=None,
    topology=None,
    opts=None,
    years=None,
    output_file=None
):
    """
    绘制不同版本的装机容量对比图
    参数:
    version_list: [str], 版本号列表（如 ["0711.4H.10.1", "0711.4H.10.2", "0711.4H.10.3"]）
    config_path: str, 配置文件路径
    pathway, topology, opts: str, 若为None则自动从config.yaml读取
    years: [int/str], 若为None则使用固定的年份序列2025-2050每隔5年
    output_file: str, 输出图片路径
    """
    # 1. 读取配置
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if pathway is None:
        pathway = config["scenario"]["pathway"]
        if isinstance(pathway, list): pathway = pathway[0]
    if topology is None:
        topology = config["scenario"]["topology"]
        if isinstance(topology, list): topology = topology[0]
    if opts is None:
        opts = config["scenario"]["opts"]
        if isinstance(opts, list): opts = opts[0]
    
    # 使用固定的年份序列：2025-2050每隔5年
    if years is None:
        years = ["2025", "2030", "2035", "2040", "2045", "2050"]
        print(f"使用固定年份序列: {', '.join(years)}")

    # 2. carrier分组映射（可根据实际情况调整）
    tech_map = {
        # 'coal': ['coal', 'coal power plant'],  # 合并煤电厂
        # 'gas': ['gas'],
        'coal w/CCS': ['coal CCS', 'coal cc'],  # 添加'coal cc'作为带碳捕捉的煤电
        'solar PV': ['solar'],
        'solar thermal': ['solar thermal'],  # 新增太阳能热发电
        'onshore wind': ['onwind'],
        'offshore wind': ['offwind'],
        'hydro': ['hydroelectricity'],  # 新增水电
        'H2': ['H2'],  # 新增氢能
        'li-ion battery': ['battery'],
        'natural gas': ['OCGT', 'CCGT', 'OCGT gas'],  # 添加'OCGT gas'
        'natural gas w/CCS': ['CCGT CCS'],
        'geothermal': ['geothermal'],
        'CHP': ['CHP coal', 'CHP gas'],  # 新增热电联产
        # 'heat storage': ['water tanks'],  # 新增热储能
        'pumped hydro': ['PHS'],  # 新增抽水蓄能
        # 'biomass': ['biomass'],  # 新增生物质能
        'heat pump': ['heat pump'],  # 新增热泵
        'boilers': ['coal boiler', 'gas boiler']  # 新增锅炉
        # 可补充其它carrier
    }
    techs = list(tech_map.keys())
    colors = plt.get_cmap('tab20').colors[:len(techs)]

    # 3. 读取每个版本每个年份的装机
    all_data = {}
    for version in version_list:
        version_data = {}
        for year in years:
            # 路径拼接，和 plot_costs_all_new.py 一致
            dir_pattern = f"postnetwork-{opts}-{topology}-{pathway}-{year}"
            summary_dir = Path(f"results/version-{version}/summary/postnetworks/positive/{dir_pattern}")
            cap_file = summary_dir / "capacities.csv"
            if not cap_file.exists():
                print(f"警告: 找不到文件 {cap_file}")
                continue
            # 读取
            try:
                df = pd.read_csv(cap_file, header=[0,1,2], index_col=[0,1])
                # 只取第一个（或唯一）列
                if isinstance(df.columns, pd.MultiIndex):
                    df = df[df.columns[0]]
                else:
                    df = df.iloc[:, 0]
                # 统计各技术总装机
                cap_by_tech = {}
                for tech, carriers in tech_map.items():
                    cap = 0
                    for carrier in carriers:
                        idx = [i for i in df.index if i[1] == carrier]
                        cap += df.loc[idx].sum() if idx else 0
                    cap_by_tech[tech] = cap / 1e3  # MW转GW
                version_data[year] = cap_by_tech
            except Exception as e:
                print(f"读取{cap_file}失败: {e}")
        all_data[version] = version_data

    # 绘图部分修改
        # 4. 绘图 - 修改图表宽度和字体大小
    fig, axes = plt.subplots(1, len(version_list), figsize=(2.5*len(version_list), 6), sharey=True)
    if len(version_list) == 1:
        axes = [axes]
        
    # 用于收集图例句柄和标签
    handles, labels = None, None
    
    # 设置图片的名称
    name = ['Case1', 'Case2', 'Case3']

    for i, (ax, version) in enumerate(zip(axes, version_list)):
        if version not in all_data or not all_data[version]:
            ax.text(0.5, 0.5, f"No data for version {version}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            continue
        
        df = pd.DataFrame.from_dict(all_data[version], orient='index').fillna(0)
        
        # 设置x轴位置
        x = np.arange(len(df.index))
        bar_width = 0.6  # 增加柱子宽度以填充更窄的图表
        
        # 绘制所有技术（单一y轴，单位GW）
        bottom = None
        for j, tech in enumerate(techs):
            if tech in df.columns:
                bars = ax.bar(x, df[tech], width=bar_width, 
                            bottom=bottom, label=tech if handles is None else "", 
                            color=colors[j])
                bottom = df[tech] if bottom is None else bottom + df[tech]
        
        # 设置x轴
        ax.set_xticks(x)
        ax.set_xticklabels(df.index, rotation=45, ha='right', fontsize=9)
        ax.set_xlim(-0.5, len(x)-0.5)
        
        # 设置y轴标签 - 放大字体
        if i == 0:  # 只在第一个子图显示y轴标签
            ax.set_ylabel('Installed Capacity (GW)', fontsize=12)
        
        # 收集图例信息
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
            
        # 放大标题字体
        ax.set_title(name[i], fontsize=14, fontweight='bold')
        ax.set_xlabel('Year', fontsize=10)
        
        # 移除各自的图例
        ax.get_legend().remove() if ax.get_legend() else None
    
    # 放大图例字体并调整位置
    legend_cols = min(len(handles), 4)  # 减少列数，使每列宽度更大
    fig.legend(handles, labels, loc='lower center', 
              bbox_to_anchor=(0.5, -0.1), ncol=legend_cols, fontsize=10)
    
    # 调整布局，为底部的图例留出空间，减小子图间距
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22, wspace=0.05)  # 减小子图间距，增大底部边距
    
    # 确保输出目录存在并保存图片
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {output_file}")
    else:
        plt.show()

def find_available_versions():
    """自动查找results目录下的可用版本"""
    versions = []
    results_dir = Path("results")
    if not results_dir.exists():
        print("警告: results目录不存在")
        return versions
    
    for dir_path in results_dir.iterdir():
        if dir_path.is_dir() and dir_path.name.startswith("version-"):
            version = dir_path.name.replace("version-", "")
            versions.append(version)
    
    return sorted(versions)

if __name__ == "__main__":
    import argparse
    import sys
    
    # 如果提供了命令行参数，则使用argparse解析
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="对比不同版本的装机容量")
        parser.add_argument("--versions", type=str, required=True, help="用逗号分隔的版本号，如 '0711.4H.8.1,0711.4H.8.2,0711.4H.8.3'")
        parser.add_argument("--config", default="config.yaml", help="配置文件路径")
        parser.add_argument("--output", default=None, help="输出图片路径")
        args = parser.parse_args()

        version_list = args.versions.split(",")
        output_file = args.output
    else:
        # 直接点击运行时，使用固定版本号
        version_list = ["0711.4H.10.1", "0711.4H.10.2", "0711.4H.10.3"]
        print(f"使用固定版本号: {', '.join(version_list)}")
        
        # 设置默认输出路径
        output_dir = Path("results/comparison_results/plots")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = str(output_dir / f"capacity_comparison_{timestamp}.png")
    
    try:
        plot_capacity_compare(
            version_list=version_list,
            config_path="config.yaml",
            output_file=output_file
        )
    except Exception as e:
        print(f"执行过程中出错: {e}")
        import traceback
        traceback.print_exc()