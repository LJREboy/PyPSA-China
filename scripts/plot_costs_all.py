#!/usr/bin/env python3
# SPDX-FileCopyrightText: : 2024 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT

"""
绘制多个规划年份的系统成本对比图。

独立脚本，读取results目录下的成本数据，将多个规划年份的成本以柱状图形式在同一张图上展示。
"""

import os
import sys
import yaml
import logging
import argparse
import traceback
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# 导入辅助函数
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from plot_summary import rename_techs, preferred_order, get_colors_safe

logger = logging.getLogger(__name__)

def configure_logging(level=logging.INFO):
    """设置日志配置"""
    logging.basicConfig(
        level=level,
        format='%(levelname)s:%(name)s:%(message)s'
    )

def load_config(config_path="config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_cost_files(config, version, pathway, topology, opts, heating_demand="positive"):
    """获取所有年份的成本文件路径"""
    planning_horizons = sorted([str(year) for year in config['scenario']['planning_horizons']])
    
    cost_files = {}
    # 尝试两种可能的目录结构
    base_dirs = [
        os.path.join(
            config['results_dir'], 
            f"version-{version}", 
            "summaries", 
            heating_demand
        ),
        os.path.join(
            config['results_dir'], 
            f"version-{version}", 
            "summary", 
            "postnetworks",
            heating_demand
        )
    ]
    
    for base_dir in base_dirs:
        for year in planning_horizons:
            if base_dir.endswith("postnetworks/" + heating_demand):
                # 第二种目录结构
                path = os.path.join(
                    base_dir, 
                    f"postnetwork-{opts}-{topology}-{pathway}-{year}",
                    "costs.csv"
                )
            else:
                # 第一种目录结构
                path = os.path.join(
                    base_dir, 
                    f"{opts}-{topology}-{pathway}-{year}",
                    "costs.csv"
                )
            
            if os.path.exists(path):
                cost_files[year] = path
                logger.info(f"找到成本文件: {path}")
    
    if not cost_files:
        raise FileNotFoundError(f"在结果目录中未找到任何成本文件")
        
    return cost_files, planning_horizons

def plot_costs_all(cost_files, planning_horizons, config, output_file=None):
    """绘制多个年份的成本对比柱状图，格式与plot_costs保持一致"""
    
    # 读取并合并多个年份的成本数据
    dfs = {}
    for year in planning_horizons:
        if year in cost_files:
            df = pd.read_csv(cost_files[year], index_col=list(range(3)), header=[1])
            # 只取第一列数据
            if len(df.columns) > 0:
                dfs[year] = df.iloc[:, 0]
            else:
                logger.warning(f"年份 {year} 的数据为空")
    
    if not dfs:
        logger.error("未找到任何指定年份的数据")
        return
        
    # 合并所有年份的数据
    cost_df = pd.DataFrame(dfs)
    
    # 按技术类型聚合成本
    df = cost_df.groupby(cost_df.index.get_level_values(2)).sum()
    
    # 将含有"products"的技术汇总为一个类别
    df = df.groupby(df.index.map(lambda x: "Industrial products" if "products" in x else x)).sum()
    
    # 转换为十亿欧元
    df = df / 1e9
    
    # 重命名技术
    df = df.groupby(df.index.map(rename_techs)).sum()
    
    # 移除低于阈值的技术
    threshold = config['plotting'].get('costs_plots_threshold', 0.1)
    to_drop = df.index[df.max(axis=1) < threshold]
    logger.info(f"移除成本低于 {threshold} 十亿欧元的技术")
    df = df.drop(to_drop)
    
    # 按照首选顺序重排索引
    new_index = preferred_order.intersection(df.index).append(
        df.index.difference(preferred_order)
    )
    
    # 确保年份按照从小到大的顺序排列（不是按总成本排序）
    new_columns = sorted(df.columns)
    
    # 创建图形
    fig, ax = plt.subplots()
    fig.set_size_inches((12, 8))
    
    # 绘制堆叠柱状图 - 关键是使用.T转置，确保年份在x轴上
    df.loc[new_index, new_columns].T.plot(
        kind="bar",
        ax=ax,
        stacked=True,
        color=get_colors_safe(config, new_index),
        width=0.7
    )
    
    # 设置图例
    handles, labels = ax.get_legend_handles_labels()
    handles.reverse()
    labels.reverse()
    
    # 设置轴标签和网格
    y_max = config['plotting'].get('costs_max', 500)
    ax.set_ylim([0, y_max])
    ax.set_ylabel("系统成本 [十亿欧元/年]")
    ax.set_xlabel("")
    ax.grid(axis="y")
    ax.legend(
        handles, 
        labels, 
        ncol=4,
        bbox_to_anchor=[1, 1],
        loc="upper left"
    )
    
    # 添加总成本值标签
    for i, year in enumerate(new_columns):
        total = df[year].sum()
        ax.text(
            i, 
            total + y_max * 0.02,
            f'{total:.1f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        fig.savefig(output_file, dpi=300, bbox_inches='tight', transparent=True)
        logger.info(f"成本图已保存至 {output_file}")
    else:
        plt.show()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="绘制多个规划年份的系统成本对比图")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--version", default=None, help="结果版本")
    parser.add_argument("--pathway", default="allreduction", help="减排路径")
    parser.add_argument("--topology", default="current+Neighbor", help="网络拓扑")
    parser.add_argument("--opts", default="ll", help="优化选项")
    parser.add_argument("--output", default=None, help="输出文件路径")
    parser.add_argument("--years", default=None, help="要绘制的年份列表，用逗号分隔，如 '2020,2030,2050'")
    
    args = parser.parse_args()
    
    # 配置日志
    configure_logging()
    
    # 加载配置
    config = load_config(args.config)
    
    # 使用命令行参数或配置文件中的值
    version = args.version or config['version']
    pathway = args.pathway
    topology = args.topology
    opts = args.opts
    
    # 输出文件名
    if args.output is None:
        output_file = f"results/plots/costs_all_{pathway}_{topology}_{version}.pdf"
    else:
        output_file = args.output
    
    try:
        # 获取成本文件路径和计划年份
        cost_files, planning_horizons = get_cost_files(
            config, version, pathway, topology, opts
        )
        
        # 如果指定了年份，则只保留这些年份
        if args.years:
            selected_years = args.years.split(',')
            planning_horizons = [y for y in planning_horizons if y in selected_years]
            logger.info(f"选取指定年份: {planning_horizons}")
        
        # 绘制成本图
        plot_costs_all(cost_files, planning_horizons, config, output_file)
        
    except Exception as e:
        logger.error(f"错误: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()