import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
'''
 绘制考虑产业链协同的工业产品未满足量的月度柱状图
 为了使图片清晰已读，仅绘制年平均未满足量 < 0.99 的产品（含aluminum）
 该脚本假设产品材料信息存储在 data/loadshedding/production_parameters_forLS_modified.csv 中，
 并且网络文件存储在 results/version-{version}/networks/elec_s_{year}.nc 中。
 如果未找到网络文件，则尝试查找 postnetwork 文件。
 该脚本会自动推断网络文件路径，并绘制每月未满足量的标幺化柱状图。
'''
def plot_industry_unmet_demand(version="0711.4H.10.3", year="2050", output_file=None):
    """
    绘制工业产品未满足量的月度柱状图
    """
    # 读取产品信息csv，筛选出产品材料（非原材料）
    prod_param_path = Path("data/loadshedding/production_parameters_forLS_modified.csv")
    if prod_param_path.exists():
        prod_df = pd.read_csv(prod_param_path)
        target_materials = list(prod_df.loc[prod_df['type'] != 'raw_material', 'materials'].astype(str))
    else:
        print(f"警告: 未找到产品参数文件 {prod_param_path}")
        target_materials = []
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # 自动推断网络文件路径
    net_path = Path(f"results/version-{version}/networks/elec_s_{year}.nc")
    if not net_path.exists():
        postnet_dir = Path(f"results/version-{version}/postnetworks")
        postnet_files = list(postnet_dir.glob(f"**/postnetwork-*-*-*-{year}.nc"))
        if postnet_files:
            net_path = postnet_files[0]
            print(f"自动选用 postnetwork 文件: {net_path}")
        else:
            print(f"警告: 未找到网络文件 {net_path} 或 postnetwork 文件")
            return
    n = pypsa.Network(str(net_path))
    # 查找所有相关发电机
    single_province = "Anhui"  # 默认配置
    unmet_data = {}
    for material in target_materials:
        gen_name = f"{single_province} {material} industrial load shedding"
        if gen_name in n.generators.index:
            p_nom = n.generators.at[gen_name, 'p_nom']
            load_shed = n.generators_t.p[gen_name]
            unmet = (p_nom - load_shed)  # MW
            freq_hours = (n.snapshot_weightings.sum().iloc[0]) / len(n.snapshots)
            unmet_energy = unmet * freq_hours  # MWh
            unmet_data[material] = unmet_energy
            print(f"产品 {material}: p_nom={p_nom}, 未满足量已提取")
        else:
            print(f"未找到发电机: {gen_name}")
    # 构建DataFrame
    all_unmet = pd.DataFrame(index=n.snapshots)
    all_base = pd.DataFrame(index=n.snapshots)
    for material in target_materials:
        if material in unmet_data:
            all_unmet[material] = unmet_data[material]
            # 基值：p_nom * freq_hours
            gen_name = f"{single_province} {material} industrial load shedding"
            p_nom = n.generators.at[gen_name, 'p_nom'] if gen_name in n.generators.index else 0.0
            all_base[material] = np.full(len(n.snapshots), p_nom * freq_hours)
    # 按月聚合
    all_unmet['month'] = all_unmet.index.month
    all_base['month'] = all_base.index.month
    monthly_unmet = all_unmet.groupby('month').sum() / 1000  # GWh
    monthly_base = all_base.groupby('month').sum() / 1000  # GWh
    # 标幺化：未满足/基值
    monthly_pu = monthly_unmet / monthly_base.replace(0, np.nan)
    # 计算年平均未满足量（标幺值）
    annual_pu = monthly_unmet.sum() / monthly_base.sum()
    # 仅保留年平均未满足量 < 0.99 的产品，并确保aluminum一定包含
    filtered_materials = [m for m in target_materials if m in annual_pu.index and (annual_pu[m] < 0.9999 or m == "aluminum")]
    print("年平均未满足量 < 0.99 的产品（含aluminum）:", filtered_materials)
    # 生成足够的颜色
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab20', len(filtered_materials))
    colors = [cmap(i) for i in range(len(filtered_materials))]
    x = np.arange(1, 13)
    plt.figure(figsize=(10, 6))
    for i, material in enumerate(filtered_materials):
        if material in monthly_pu.columns:
            plt.plot(x, monthly_pu[material], color=colors[i], linestyle='-', marker='o', label=f"{material}")
    plt.xticks(x, month_labels, rotation=45)
    plt.ylabel('Product Demand Fulfillment (Per Unit)', fontsize=12)
    plt.xlabel('Month', fontsize=12)
    # 去掉图表标题
    plt.legend(ncol=4, fontsize=9)
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    output_dir = Path("results/comparison_results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "industry_unmet_demand_monthly_2050.png"
    plot_industry_unmet_demand("0711.4H.10.3", "2050", output_file)
