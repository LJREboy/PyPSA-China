# SPDX-FileCopyrightText: : 2022 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT

"""
This script prepares the base network for the PyPSA-China model for the year 2020.
It creates a PyPSA network with the following components:

1. Network Structure:
   - Sets up the network with specified snapshots and time resolution
   - Configures bus locations based on province centroids
   - Adds carriers for different energy types (electricity, heat, gas, coal)

2. Energy Components:
   - Adds renewable generators (wind, solar, hydro)
   - Configures conventional power plants
   - Sets up heat pumps and resistive heaters
   - Adds storage components (batteries, pumped hydro storage)
   - Configures transmission lines between provinces

3. Demand and Constraints:
   - Loads electricity demand data
   - Configures heat demand profiles
   - Sets up CO2 emission constraints
   - Adds capacity constraints for different technologies

4. Cost Parameters:
   - Loads technology costs
   - Configures capital and marginal costs
   - Sets up cost parameters for transmission and storage

The script takes configuration parameters from the Snakefile and creates a network
that serves as the base for further optimization and analysis.
"""

# for pathway network

from vresutils.costdata import annuity
from _helpers import configure_logging,override_component_attrs
import pypsa
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from functools import partial
import pyproj
from shapely.ops import transform
import xarray as xr
from functions import pro_names, HVAC_cost_curve
from add_electricity import load_costs

def haversine(p1,p2):
    """Calculate the great circle distance in km between two points on
    the earth (specified in decimal degrees)
    """

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [p1[0], p1[1], p2[0], p2[1]])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def add_buses(network,nodes,suffix,carrier,pro_centroid_x,pro_centroid_y):

    network.madd('Bus',
                 nodes,
                 suffix=suffix,
                 x=pro_centroid_x,
                 y=pro_centroid_y,
                 carrier=carrier,
                 )

# LS相关函数
def process_products(days):
    """处理产品数据"""

    df_pp = pd.read_csv(snakemake.input.production_parameters)
    df_pm = pd.read_csv(snakemake.input.production_relationship_matrix)

    # 处理产品关系矩阵
    relationship_matrix = df_pm.iloc[:, 1:].values.tolist()
    
    # 将产品参数转换为字典
    products = {
        "materials": df_pp['materials'].tolist(),
        "hierarchy": df_pp['hierarchy'].tolist(),
        "yearly_demand": df_pp['yearly_demand'].tolist(),
        "net_demand_rate": df_pp['net_demand_rate'].tolist(),
        "price": df_pp['price'].tolist(),
        "electricity_consumption": df_pp['electricity_consumption'].tolist(),
        "initial_inventory": df_pp['initial_inventory'].tolist(),
        "max_inventory": df_pp['max_inventory'].tolist(),
        "excess_rate": df_pp['excess_rate'].tolist(),
        "type": df_pp['type'].tolist(),
        "relationship_matrix": relationship_matrix,
    }
    
    products["yearly_demand"] = [demand for demand in products["yearly_demand"]]  # 年需求量扩大为原来的2倍

    # 价格矩阵修正为1欧元,1欧元=7.55人民币
    products["price"] = [price * 10000 / 7.55 for price in products["price"]]
 
    # 计算所有产品的日总电能需求，平均到每个时段
    products["daily_total_load"] = sum(np.array(products["yearly_demand"]) * np.array(products["electricity_consumption"])) / days # 每天总电能需求（MWh）
    products["load_t"] = products["daily_total_load"] / 24  # 每小时平均功率（MW）

    # 初始化产品总增加值字典
    products["product_total_profit"] = {}
    
    # 计算每个产品的总增加值
    for i, material in enumerate(products["materials"]):
        products["product_total_profit"][material] = products["price"][i] * products["yearly_demand"][i] * products["net_demand_rate"][i] # 产品i一年的总增加值（欧元）

    # 计算所有产品的全年总增加值
    products["total_profit"] = np.sum(np.array(products["price"]) * np.array(products["yearly_demand"]) * np.array(products["net_demand_rate"])) # 每年总增加值（欧元）
    # products["fit_t"] = products["daily_total_fit"] / 24  # 每小时增加值（欧元）
    products["total_profit"] = products["total_profit"] * 1e-9 # 每年总增加值（十亿欧元）

    products["demand"] = [yearly_demand / days for yearly_demand in products["yearly_demand"]]  # 产品需求量转换为每天的需求量

    return products

def prepare_network(config):

    if "overrides" in snakemake.input.keys():
        overrides = override_component_attrs(snakemake.input.overrides)
        network = pypsa.Network(override_component_attrs=overrides)
    else:
        network = pypsa.Network()

    # set times
    planning_horizons = snakemake.wildcards['planning_horizons']
    if int(planning_horizons) % 4 != 0:
        snapshots = pd.date_range(str(planning_horizons)+'-01-01 00:00', str(planning_horizons)+'-12-31 23:00', freq=config['freq'])
    else:
        snapshots = pd.date_range('2025-01-01 00:00', '2025-12-31 23:00', freq=config['freq'])
        snapshots = snapshots.map(lambda t: t.replace(year=int(planning_horizons)))

    network.set_snapshots(snapshots)
    # 从freq中解析出小时数来设置snapshot_weightings
    # 例如：'1h' -> 1, '2h' -> 2, '8h' -> 8
    freq_hours = float(config['freq'].replace('h', ''))
    network.snapshot_weightings[:] = freq_hours
    represented_hours = network.snapshot_weightings.sum().iloc[0]
    Nyears= represented_hours/8760.

    #load graph
    nodes = pd.Index(pro_names)
    pathway = snakemake.wildcards['pathway']

    tech_costs = snakemake.input.tech_costs
    cost_year = snakemake.wildcards.planning_horizons
    costs = load_costs(tech_costs,config['costs'],config['electricity'],cost_year, Nyears)

    date_range = pd.date_range('2025-01-01 00:00', '2025-12-31 23:00', freq=config['freq'])
    date_range = date_range.map(lambda t: t.replace(year=2020))

    ds_solar = xr.open_dataset(snakemake.input.profile_solar)
    ds_onwind = xr.open_dataset(snakemake.input.profile_onwind)
    ds_offwind = xr.open_dataset(snakemake.input.profile_offwind)

    solar_p_max_pu = ds_solar['profile'].transpose('time', 'bus').to_pandas()
    # Ensure solar_p_max_pu has naive timestamps to match date_range
    if solar_p_max_pu.index.tz is not None:
        solar_p_max_pu.index = solar_p_max_pu.index.tz_localize(None)
    solar_p_max_pu = solar_p_max_pu.loc[date_range].set_index(network.snapshots)
    onwind_p_max_pu = ds_onwind['profile'].transpose('time', 'bus').to_pandas()
    # Ensure onwind_p_max_pu has naive timestamps to match date_range
    if onwind_p_max_pu.index.tz is not None:
        onwind_p_max_pu.index = onwind_p_max_pu.index.tz_localize(None)
    onwind_p_max_pu = onwind_p_max_pu.loc[date_range].set_index(network.snapshots)
    offwind_p_max_pu = ds_offwind['profile'].transpose('time', 'bus').to_pandas()
    # Ensure offwind_p_max_pu has naive timestamps to match date_range
    if offwind_p_max_pu.index.tz is not None:
        offwind_p_max_pu.index = offwind_p_max_pu.index.tz_localize(None)
    offwind_p_max_pu = offwind_p_max_pu.loc[date_range].set_index(network.snapshots)

    def rename_province(label):
        rename = {
            "Nei Mongol": "InnerMongolia",
            "Ningxia Hui": "Ningxia",
            "Xinjiang Uygur": "Xinjiang",
            "Xizang": "Tibet"
        }

        for old, new in rename.items():
            if old == label:
                label = new
        return label

    pro_shapes = gpd.GeoDataFrame.from_file(snakemake.input.province_shape)
    pro_shapes = pro_shapes.to_crs(4326)
    pro_shapes.index = pro_shapes.NAME_1.map(rename_province)
    pro_centroid_x = pro_shapes.to_crs('+proj=cea').centroid.to_crs(pro_shapes.crs).x
    pro_centroid_y = pro_shapes.to_crs('+proj=cea').centroid.to_crs(pro_shapes.crs).y

    # add buses
    for suffix in config["bus_suffix"]:
        carrier = config["bus_carrier"][suffix]
        add_buses(network, nodes, suffix, carrier, pro_centroid_x, pro_centroid_y)

    # add carriers
    network.add("Carrier", "AC")  # 添加AC carrier定义
    if config["heat_coupling"]:
        network.add("Carrier", "heat")
    for carrier in config["Techs"]["vre_techs"]:
        network.add("Carrier", carrier)
    for carrier in config["Techs"]["store_techs"]:
        if carrier == 'battery':
            network.add("Carrier", "battery")
            network.add("Carrier", "battery discharger")
        else:
            network.add("Carrier", carrier)
    for carrier in config["Techs"]["conv_techs"]:
        if "gas" in carrier:
            network.add("Carrier", carrier, co2_emissions=costs.at['gas', 'co2_emissions'])  # in t_CO2/MWht
        if "coal" in carrier:
            network.add("Carrier", carrier, co2_emissions=costs.at['coal', 'co2_emissions'])
    if config["add_gas"]:
        network.add("Carrier", "gas", co2_emissions=costs.at['gas', 'co2_emissions'])  # in t_CO2/MWht
    if config["add_coal"]:
        network.add("Carrier", "coal", co2_emissions=costs.at['coal', 'co2_emissions'])
    if config["add_aluminum"]:
        network.add("Carrier", "aluminum")

    # add global constraint
    if not isinstance(config['scenario']['co2_reduction'], tuple):

        if config['scenario']['co2_reduction'] is not None:

            # extra co2
            # 791 TWh extra space heating demand + 286 Twh extra hot water demand
            # 60% CHP efficiency 0.468 40% coal boiler efficiency 0.97
            # (((791+286) * 0.6 /0.468) + ((791+286) * 0.4 /0.97))  * 0.34 * 1e6 = 0.62 * 1e9

            co2_limit = (5.288987673 + 0.628275682)*1e9 * (1 - config['scenario']['co2_reduction'][pathway][planning_horizons]) # Chinese 2020 CO2 emissions of electric and heating sector

            network.add("GlobalConstraint",
                        "co2_limit",
                        type="primary_energy",
                        carrier_attribute="co2_emissions",
                        sense="<=",
                        constant=co2_limit)

    #load demand data
    with pd.HDFStore(snakemake.input.elec_load, mode='r') as store:
        load = 1e6 * store['load']
        load = load.loc[network.snapshots]
        
    load.columns = pro_names

    ## LS从这里开始修改

    # 如果配置文件中启用了工业负荷切除，并且使用单节点（即不考虑区域划分），则设置柔性负荷
    if config['products_load_shedding'] and config['using_single_node']:
        
        # get the single node province from the config
        single_province = config['single_node_province']  # now "Anhui"
        if single_province in load.columns:
            # process parameters related to products
            AllTime = len(snapshots) # 试验次数，等于时间序列长度
            days = int(AllTime * freq_hours // 24)  # 天数
            products = process_products(days)
            # 将products存储为network的属性
            network.products = products
            # Add flexible load to the network
            if config['LS_scenario'] == 1:
                # 为每个产品添加带惩罚成本的弹性负荷
                for i, material in enumerate(products["materials"]):
                    network.add(
                        "Generator",
                        f"{single_province} {material} industrial load shedding",  # 使用省份前缀命名产品负荷
                        bus=single_province,
                        carrier=f"{material} products",  # carrier名称即为产品名称
                        p_max_pu=1,  # 最大出力为1
                        p_min_pu=0, 
                        p_nom=products["demand"][i] * products["electricity_consumption"][i] / 24,  # 需求规模
                        p_nom_extendable=False,  # 不允许模型优化此负荷的规模
                        marginal_cost=8072 # 切负荷成本设为平均成本8072欧元/MWh
                        # price是1单位产品的增加值，electricity_consumption是1单位产品的电能消耗，二者相除即为产品i生产用电1MWh产生的边际成本
                    )
                    network.add("Load",
                        f"{single_province} {material} industrial load",
                        bus=single_province,
                        p_set=products["demand"][i] * products["electricity_consumption"][i] / 24, 
                        carrier="AC"  # carrier为电能
                    )
                
            elif config['LS_scenario'] == 2:
                # 为每个产品添加带惩罚成本的弹性负荷
                for i, material in enumerate(products["materials"]):
                    network.add(
                        "Generator",
                        f"{single_province} {material} industrial load shedding",  # 使用省份前缀命名产品负荷
                        bus=single_province,
                        carrier=f"{material} products",  # carrier名称即为产品名称
                        p_max_pu=1,  # 最大出力为1
                        p_min_pu=0, 
                        p_nom=products["demand"][i] * products["electricity_consumption"][i] / 24,  # 需求规模
                        p_nom_extendable=False,  # 不允许模型优化此负荷的规模
                        marginal_cost=products["price"][i] / products["electricity_consumption"][i] # 切负荷成本（欧元）
                        # price是1单位产品的增加值，electricity_consumption是1单位产品的电能消耗，二者相除即为产品i生产用电1MWh产生的边际成本
                    )
                    network.add("Load",
                        f"{single_province} {material} industrial load",
                        bus=single_province,
                        p_set=products["demand"][i] * products["electricity_consumption"][i] / 24, 
                        carrier="AC"  # carrier为电能
                    )

            elif config['LS_scenario'] == 3:
                ## 第一部分：添加产品仓储设施

                # 添加产品节点（每种物料一个bus，对应文档bus_i）
                # 使用省份前缀命名产品bus
                for material in products["materials"]:
                    network.add("Bus", 
                                f"{single_province} {material} bus",  # 使用省份前缀命名产品bus
                                carrier="product"
                            )
                # 添加产品仓储设施（每个产品添加一个store，对应文档s_i）
                for i, material in enumerate(products["materials"]):
                    network.add(
                        "Store",
                        f"{single_province} {material} store",  # 使用省份前缀命名产品store
                        bus=f"{single_province} {material} bus", # 使用省份前缀命名产品store
                        carrier="product",  # 仓储类型
                        e_nom=products["max_inventory"][i],  # 仓储容量上限（可调）目前假设为14天需求量
                        e_initial=products["initial_inventory"][i],  # 初始库存
                        e_cyclic=False  # 是否周期性
                    )
                
                ## 第二部分：添加生产过程的Link
                
                # 使用单位用能向量修正关系矩阵，矩阵的每一列除以单位用能向量的对应元素
                # 此时矩阵[i][i]表示输入1MWh电能可生产的产品i的数量；矩阵[i][j]表示产品i生产过程中，输入1MWh电能时消耗的产品j的数量
                # 由于link输入的是功率，而关系矩阵对角元素表示的是每单位电能生产的产品数量，所以需要乘以时间分辨率
                products["relationship_matrix"] = np.array(products["relationship_matrix"]) / np.array(products["electricity_consumption"]).reshape(-1, 1) * freq_hours  # 每小时的能量需求
                # 添加产品生产Link（每个产品一个生产Link）
                for j, material in enumerate(products["materials"]):
                    inputs = []
                    efficiencies = []
                    for i, input_material in enumerate(products["materials"]):
                        if products["relationship_matrix"][i][j] > 0 and i != j:
                            inputs.append(input_material)
                            efficiencies.append(products["relationship_matrix"][i][j])
                    if inputs: 
                        link_kwargs = {
                            "bus0": single_province,  # 电力输入总线
                            "carrier": products["hierarchy"][j],  # 根据配置中的产品层级添加carrier
                            # p_nom为每个时段的功率需求，时段能量需求为日总用能除每个时段的长度（24/resolution），由于p_nom是输入功率，因此还需要再除以时间分辨率
                            # 故最终p_nom为产品的日总能量需求除以24小时
                            "p_nom": products["electricity_consumption"][j] * products["demand"][j] / 24,  
                            "p_max_pu": 1.0,#products['excess_rate'][j],  # 最大出力比例
                            "p_min_pu": 0.0,  # 最小出力比例
                            "p_nom_extendable": False,  
                        }
                        # 设置输入物料的效率（消耗量为负）
                        link_kwargs[f"efficiency"] = -efficiencies[0]
                        for idx, input_material in enumerate(inputs):
                            input_bus_name = f"{single_province} {input_material} bus"  # 添加省份前缀
                            link_kwargs[f"bus{idx + 1}"] = input_bus_name  # 添加输入bus
                            if idx + 1 != 1:
                                link_kwargs[f"efficiency{idx + 1}"] = -efficiencies[idx]
                        # 添加产品输出bus
                        output_bus_name = f"{single_province} {material} bus"  # 添加省份前缀
                        link_kwargs[f"bus{len(inputs) + 1}"] = output_bus_name
                        link_kwargs[f"efficiency{len(inputs) + 1}"] = products["relationship_matrix"][j][j]

                        network.add("Link", 
                                    f"{single_province} {material} production",
                                    **link_kwargs
                                )
                ## 第三部分：添加弹性负荷（负荷切除）
                
                # 构造需求时间序列：每个产品每天最后一个时段有需求
                demand_series = np.zeros(AllTime)
                freq_hours_int = int(freq_hours)
                for i in range(days):
                    demand_series[i * 24 // freq_hours_int + 24 // freq_hours_int - 1] = 1 
                # 为每个产品添加带惩罚成本的负荷
                for i, material in enumerate(products["materials"]):
                    product_bus_name = f"{single_province} {material} bus"  
                    network.add("Generator",
                        f"{single_province} {material} industrial load shedding",  # 使用省份前缀命名产品负荷
                        bus = product_bus_name,
                        carrier=f"{material} products",  # carrier名称即为产品名称
                        p_max_pu = demand_series,  
                        p_min_pu = 0,  
                        p_nom = products["demand"][i] * products["net_demand_rate"][i],  # 需求规模
                        p_nom_extendable = False,  # 允许模型优化此负荷的规模
                        marginal_cost = products["price"][i] / freq_hours # 切负荷成本（欧元）
                        # 这里需要注意，price_matrix是1单位产品的增加值，而实际输出的产品会乘以时间分辨率，所以需要除以时间分辨率以确保边际成本正确
                )
                    network.add("Load",
                        f"{single_province} {material} industrial load",
                        bus=product_bus_name,
                        p_set = products["demand"][i] * products["net_demand_rate"][i] * demand_series,  # 产品需求
                        carrier=f"{material}"  # carrier为产品名称
                    )
            else:
                raise ValueError("Invalid LS_scenario configuration. Please check the 'LS_scenario' parameter in the config.")

        # Subtract products load from electric load only for affected provinces
        load_minus_products = load.copy()
        load_minus_products[single_province] = load[single_province] - products["load_t"]
        network.madd("Load", nodes, bus=nodes, p_set=load_minus_products)
    else:
        # 如果没有负荷切割，则添加原有负荷，不设置products属性
        network.products = None
        network.madd("Load", nodes, bus=nodes, p_set=load[nodes])

    ## 这里LS修改结束

    if config["heat_coupling"]:

        central_fraction = pd.read_hdf(snakemake.input.central_fraction)
        with pd.HDFStore(snakemake.input.heat_demand_profile, mode='r') as store:
            heat_demand = store['heat_demand_profiles']
            heat_demand = heat_demand.loc[network.snapshots]

        network.madd("Load",
                     nodes,
                     suffix=" decentral heat",
                     bus=nodes + " decentral heat",
                     p_set=heat_demand[nodes].multiply(1-central_fraction))

        network.madd("Load",
                     nodes,
                     suffix=" central heat",
                     bus=nodes + " central heat",
                     p_set=heat_demand[nodes].multiply(central_fraction))

    if config["add_gas"]:
        # add converter from fuel source
        network.madd("Generator",
                     nodes,
                     suffix=' gas fuel',
                     bus=nodes + " gas",
                     carrier="gas",
                     p_nom_extendable=True,
                     marginal_cost=costs.at['OCGT', 'fuel'])

        network.add("Carrier", "biogas")
        network.madd("Store",
                     nodes + " gas Store",
                     bus=nodes + " gas",
                     e_nom_extendable=True,
                     carrier="biogas")

    if config["add_coal"]:
        network.madd("Generator",
                     nodes + " coal fuel",
                     bus=nodes + " coal",
                     carrier="coal",
                     p_nom_extendable=True,
                     marginal_cost=costs.at['coal', 'fuel'])

    if config['add_hydro']:

        #######
        df = pd.read_csv('data/hydro/dams_large.csv', index_col=0)
        points = df.apply(lambda row: Point(row.Lon, row.Lat), axis=1)
        dams = gpd.GeoDataFrame(df, geometry=points, crs=4236)

        hourly_rng = pd.date_range('1979-01-01', '2017-01-01', freq=config['freq'], inclusive='left')
        inflow = pd.read_pickle('data/hydro/daily_hydro_inflow_per_dam_1979_2016_m3.pickle').reindex(hourly_rng, fill_value=0)
        inflow.columns = dams.index

        water_consumption_factor = dams.loc[:, 'Water_consumption_factor_avg'] * 1e3 # m^3/KWh -> m^3/MWh


        #######
        # ### Add hydro stations as buses
        network.madd('Bus',
            dams.index,
            suffix=' station',
            carrier='stations',
            x=dams['geometry'].to_crs('+proj=cea').centroid.to_crs(pro_shapes.crs).x,
            y=dams['geometry'].to_crs('+proj=cea').centroid.to_crs(pro_shapes.crs).y)

        dam_buses = network.buses[network.buses.carrier=='stations']


        # ### add hydro reservoirs as stores

        initial_capacity = pd.read_pickle('data/hydro/reservoir_initial_capacity.pickle')
        effective_capacity = pd.read_pickle('data/hydro/reservoir_effective_capacity.pickle')
        initial_capacity.index = dams.index
        effective_capacity.index = dams.index
        initial_capacity = initial_capacity/water_consumption_factor
        effective_capacity=effective_capacity/water_consumption_factor

        network.madd('Store',
            dams.index,
            suffix=' reservoir',
            bus=dam_buses.index,
            e_nom=effective_capacity,
            e_initial=initial_capacity,
            e_cyclic=True,
            marginal_cost=config['costs']['marginal_cost']['hydro'])

        ### add hydro turbines to link stations to provinces
        network.madd('Link',
                    dams.index,
                    suffix=' turbines',
                    bus0=dam_buses.index,
                    bus1=dams['Province'],
                    carrier="hydroelectricity",
                    p_nom=10 * dams['installed_capacity_10MW'],
                    capital_cost=costs.at['hydro','capital_cost'],
                    efficiency= 1)


        ### add rivers to link station to station
        bus0s = [0, 21, 11, 19, 22, 29, 8, 40, 25, 1, 7, 4, 10, 15, 12, 20, 26, 6, 3, 39]
        bus1s = [5, 11, 19, 22, 32, 8, 40, 25, 35, 2, 4, 10, 9, 12, 20, 23, 6, 17, 14, 16]

        for bus0, bus2 in list(zip(dams.index[bus0s], dam_buses.iloc[bus1s].index)):

            # normal flow
            network.links.at[bus0 + ' turbines', 'bus2'] = bus2
            network.links.at[bus0 + ' turbines', 'efficiency2'] = 1.

        ### spillage
        for bus0, bus1 in list(zip(dam_buses.iloc[bus0s].index, dam_buses.iloc[bus1s].index)):
            network.add('Link',
                       "{}-{}".format(bus0,bus1) + ' spillage',
                       bus0=bus0,
                       bus1=bus1,
                       p_nom_extendable=True)

        dam_ends = [dam for dam in range(len(dams.index)) if (dam in bus1s and dam not in bus0s) or (dam not in bus0s+bus1s)]

        for bus0 in dam_buses.iloc[dam_ends].index:
            network.add('Link',
                        bus0 + ' spillage',
                        bus0=bus0,
                        bus1='Tibet',
                        p_nom_extendable=True,
                        efficiency=0.0)

        #### add inflow as generators
        # only feed into hydro stations which are the first of a cascade
        inflow_stations = [dam for dam in range(len(dams.index)) if not dam in bus1s ]

        for inflow_station in inflow_stations:

            # p_nom = 1 and p_max_pu & p_min_pu = p_pu, compulsory inflow

            date_range = pd.date_range('2025-01-01 00:00', '2025-12-31 23:00', freq=config['freq'])
            date_range = date_range.map(lambda t: t.replace(year=2016))

            # Resample inflow data to match network frequency
            resampled_inflow = inflow.resample(config['freq']).sum()
            # Ensure resampled_inflow has naive timestamps to match date_range
            if resampled_inflow.index.tz is not None:
                resampled_inflow.index = resampled_inflow.index.tz_localize(None)
            resampled_inflow = resampled_inflow.loc[date_range]

            p_nom = (resampled_inflow/water_consumption_factor).iloc[:,inflow_station].max()
            p_pu = (resampled_inflow/water_consumption_factor).iloc[:,inflow_station] / p_nom
            p_pu.index = network.snapshots
            network.add('Generator',
                       dams.index[inflow_station] + ' inflow',
                       bus=dam_buses.iloc[inflow_station].name,
                       carrier='hydro_inflow',
                       p_max_pu=p_pu.clip(1.e-6),
                       p_min_pu=p_pu.clip(1.e-6),
                       p_nom=p_nom)

            # p_nom*p_pu = XXX m^3 then use turbines efficiency to convert to power

        ## add otehr existing hydro power
        hydro_p_nom = pd.read_hdf("data/p_nom/hydro_p_nom.h5")
        hydro_p_max_pu = pd.read_hdf("data/p_nom/hydro_p_max_pu.h5", key="hydro_p_max_pu")

        date_range = pd.date_range('2025-01-01 00:00', '2025-12-31 23:00', freq=config['freq'])
        date_range = date_range.map(lambda t: t.replace(year=2020))
        # Ensure hydro_p_max_pu has naive timestamps to match date_range
        if hydro_p_max_pu.index.tz is not None:
            hydro_p_max_pu.index = hydro_p_max_pu.index.tz_localize(None)
        hydro_p_max_pu = hydro_p_max_pu.loc[date_range]
        hydro_p_max_pu.index = network.snapshots

        network.madd('Generator',
                    nodes,
                    suffix=' hydroelectricity',
                    bus=nodes,
                    carrier="hydroelectricity",
                    p_nom=hydro_p_nom,
                    p_nom_min=hydro_p_nom,
                    p_nom_extendable=False,
                    capital_cost=costs.at['hydro','capital_cost'],
                    p_max_pu=hydro_p_max_pu)

    # add components
    network.madd("Generator",
                 nodes,
                 suffix=' onwind',
                 bus=nodes,
                 carrier="onwind",
                 p_nom_extendable=False,
                 p_nom_max=ds_onwind['p_nom_max'].to_pandas(),
                 capital_cost = costs.at['onwind','capital_cost'],
                 marginal_cost=costs.at['onwind','marginal_cost'],
                 p_max_pu=onwind_p_max_pu,
                 lifetime=costs.at['onwind','lifetime'])

    offwind_nodes = ds_offwind['bus'].to_pandas().index
    network.madd("Generator",
                 offwind_nodes,
                 suffix=' offwind',
                 bus=offwind_nodes,
                 carrier="offwind",
                 p_nom_extendable=False,
                 p_nom_max=ds_offwind['p_nom_max'].to_pandas(),
                 capital_cost = costs.at['offwind','capital_cost'],
                 marginal_cost=costs.at['offwind','marginal_cost'],
                 p_max_pu=offwind_p_max_pu,
                 lifetime=costs.at['offwind', 'lifetime'])

    network.madd("Generator",
                 nodes,
                 suffix=' solar',
                 bus=nodes,
                 carrier="solar",
                 p_nom_extendable=False,
                 p_nom_max=ds_solar['p_nom_max'].to_pandas(),
                 capital_cost = costs.at['solar','capital_cost'],
                 marginal_cost=costs.at['solar','marginal_cost'],
                 p_max_pu=solar_p_max_pu,
                 lifetime=costs.at['solar', 'lifetime'])

    if "resistive heater" in config["Techs"]["vre_techs"]:
        for cat in [" decentral ", " central "]:
            network.madd("Link",
                         nodes + cat + "resistive heater",
                         bus0=nodes,
                         bus1=nodes + cat + "heat",
                         carrier="resistive heater",
                         efficiency=costs.at[cat.lstrip()+'resistive heater','efficiency'],
                         capital_cost=costs.at[cat.lstrip()+'resistive heater','efficiency']*costs.at[cat.lstrip()+'resistive heater','capital_cost'],
                         marginal_cost=costs.at[cat.lstrip()+'resistive heater','efficiency']*costs.at[cat.lstrip()+'resistive heater','marginal_cost'],
                         p_nom_extendable=True,
                         lifetime=costs.at[cat.lstrip()+'resistive heater','lifetime'])

    if "solar thermal" in config["Techs"]["vre_techs"]:
        #this is the amount of heat collected in W per m^2, accounting
        #for efficiency
        with pd.HDFStore(snakemake.input.solar_thermal_name, mode='r') as store:
            #1e3 converts from W/m^2 to MW/(1000m^2) = kW/m^2
            solar_thermal = config['solar_cf_correction'] * store['solar_thermal_profiles']/1e3

        date_range = pd.date_range('2025-01-01 00:00', '2025-12-31 23:00', freq=config['freq'])
        date_range = date_range.map(lambda t: t.replace(year=2020))

        # Ensure solar_thermal has naive timestamps to match date_range
        if solar_thermal.index.tz is not None:
            solar_thermal.index = solar_thermal.index.tz_localize(None)
        solar_thermal = solar_thermal.loc[date_range].set_index(network.snapshots)

        for cat in [" central "]:
            network.madd("Generator",
                         nodes,
                         suffix=cat + "solar thermal",
                         bus=nodes + cat + "heat",
                         carrier="solar thermal",
                         p_nom_extendable=False,
                         capital_cost=costs.at[cat.lstrip()+'solar thermal','capital_cost'],
                         p_max_pu=solar_thermal[nodes].clip(1.e-4),
                         lifetime=costs.at[cat.lstrip()+'solar thermal','lifetime'])

    if "coal boiler" in config["Techs"]["conv_techs"]:
        for cat in [" decentral "]:
            network.madd("Link",
                         nodes + cat + "coal boiler",
                         p_nom_extendable=True,
                         bus0=nodes + " coal",
                         bus1=nodes + cat + "heat",
                         carrier="coal boiler",
                         efficiency=costs.at[cat.lstrip()+'coal boiler','efficiency'],
                         marginal_cost=costs.at[cat.lstrip() + 'coal boiler', 'VOM'],
                         capital_cost=costs.at[cat.lstrip()+'coal boiler','efficiency']*costs.at[cat.lstrip()+'coal boiler','capital_cost'],
                         lifetime=costs.at[cat.lstrip()+'coal boiler','lifetime'])

    if "water tanks" in config["Techs"]["store_techs"]:
        for cat in [' decentral ', ' central ']:
            network.madd("Bus",
                         nodes,
                         suffix=cat + "water tanks",
                         x=pro_centroid_x,
                         y=pro_centroid_y,
                         carrier="water tanks")

            network.madd("Link",
                         nodes + cat + "water tanks charger",
                         bus0=nodes + cat + "heat",
                         bus1=nodes + cat + "water tanks",
                         carrier="water tanks",
                         efficiency=costs.at['water tank charger','efficiency'],
                         p_nom_extendable=True)

            network.madd("Link",
                         nodes + cat + "water tanks discharger",
                         bus0=nodes + cat + "water tanks",
                         bus1=nodes + cat + "heat",
                         carrier="water tanks",
                         efficiency=costs.at['water tank discharger','efficiency'],
                         p_nom_extendable=True)

            network.madd("Store",
                         nodes + cat + "water tank",
                         bus=nodes + cat + "water tanks",
                         carrier="water tanks",
                         e_cyclic=True,
                         e_nom_extendable=True,
                         standing_loss=1-np.exp(-1/(24.* (config["tes_tau"] if cat==' decentral ' else 180.))),  # [HP] 180 day time constant for centralised
                         capital_cost=costs.at[cat.lstrip()+'water tank storage','capital_cost'],
                         lifetime=costs.at[cat.lstrip()+'water tank storage','lifetime'])

    if "battery" in config["Techs"]["store_techs"]:
        network.madd("Bus",
                     nodes,
                     suffix=" battery",
                     x=pro_centroid_x,
                     y=pro_centroid_y,
                     carrier="battery")

        network.madd("Store",
                     nodes + " battery",
                     bus=nodes + " battery",
                     e_cyclic=True,
                     e_nom_extendable=True,
                     capital_cost=costs.at['battery storage','capital_cost'],
                     lifetime=costs.at['battery storage','lifetime'])

        network.madd("Link",
                     nodes + " battery charger",
                     bus0=nodes,
                     bus1=nodes + " battery",
                     efficiency=costs.at['battery inverter','efficiency']**0.5,
                     capital_cost=0.5*costs.at['battery inverter','capital_cost'],
                     p_nom_extendable=True,
                     carrier="battery",
                     lifetime=costs.at['battery inverter','lifetime'] )

        network.madd("Link",
                     nodes + " battery discharger",
                     bus0=nodes + " battery",
                     bus1=nodes,
                     efficiency=costs.at['battery inverter','efficiency']**0.5,
                     capital_cost=0.5*costs.at['battery inverter','capital_cost'],
                     carrier="battery discharger",
                     p_nom_extendable=True,
                     lifetime=costs.at['battery inverter','lifetime'])

    if "PHS" in config["Techs"]["store_techs"]:
        # pure pumped hydro storage, fixed, 6h energy by default, no inflow
        hydrocapa_df = pd.read_csv('data/hydro/PHS_p_nom.csv', index_col=0)
        phss = hydrocapa_df.index[hydrocapa_df['MW'] > 0].intersection(nodes)
        if config['hydro']['hydro_capital_cost']:
            cc=costs.at['PHS','capital_cost']
        else:
            cc=0.

        network.madd("StorageUnit",
                     phss,
                     suffix=" PHS",
                     bus=phss,
                     carrier="PHS",
                     p_nom_extendable=False,
                     p_nom=hydrocapa_df.loc[phss]['MW'],
                     p_nom_min=hydrocapa_df.loc[phss]['MW'],
                     max_hours=config['hydro']['PHS_max_hours'],
                     efficiency_store=np.sqrt(costs.at['PHS','efficiency']),
                     efficiency_dispatch=np.sqrt(costs.at['PHS','efficiency']),
                     cyclic_state_of_charge=True,
                     capital_cost = cc,
                     marginal_cost=0.)

    #add lines

    if not config['no_lines']:
        "Split bidirectional links into two unidirectional links to include transmission losses."

        edges_ext = pd.read_csv(snakemake.input.edges_ext, header=None)

        lengths = 1.25 * np.array([haversine([network.buses.at[name0,"x"],network.buses.at[name0,"y"]],
                                  [network.buses.at[name1,"x"],network.buses.at[name1,"y"]]) for name0,name1 in edges_ext[[0,1]].values])

        cc = (config['line_cost_factor'] * lengths * [HVAC_cost_curve(l) for l in
                                                          lengths]) * 1.5 * 1.02 * Nyears * annuity(40.,config['costs']['discountrate'])

        network.madd("Link",
                     edges_ext[0] + '-' + edges_ext[1],
                     bus0=edges_ext[0].values,
                     bus1=edges_ext[1].values,
                     suffix =" ext positive",
                     p_nom_extendable=False,
                     p_nom=edges_ext[2].values,
                     p_nom_min=edges_ext[2].values,
                     p_min_pu=0,
                     efficiency=config["transmission_efficiency"]["DC"]["efficiency_static"]* config["transmission_efficiency"]["DC"]["efficiency_per_1000km"]**(lengths/1000),
                     length=lengths,
                     build_year=2020,
                     lifetime=70,
                     capital_cost=cc)

        network.madd("Link",
                     edges_ext[1] + '-' + edges_ext[0],
                     bus0=edges_ext[1].values,
                     bus1=edges_ext[0].values,
                     suffix=" ext reversed",
                     p_nom_extendable=False,
                     p_nom=edges_ext[2].values,
                     p_nom_min=edges_ext[2].values,
                     p_min_pu=0,
                     efficiency=config["transmission_efficiency"]["DC"]["efficiency_static"]* config["transmission_efficiency"]["DC"]["efficiency_per_1000km"]**(lengths/1000),
                     length=lengths,
                     lifetime=70,
                     build_year=2020,
                     capital_cost=0)

        edges = pd.read_csv(snakemake.input.edges, header=None)

        lengths = 1.25 * np.array([haversine([network.buses.at[name0,"x"],network.buses.at[name0,"y"]],
                                  [network.buses.at[name1,"x"],network.buses.at[name1,"y"]]) for name0,name1 in edges[[0,1]].values])

        cc = (config['line_cost_factor'] * lengths * [HVAC_cost_curve(l) for l in
                                                          lengths]) * 1.5 * 1.02 * Nyears * annuity(40.,config['costs']['discountrate'])

        network.madd("Link",
                     edges[0] + '-' + edges[1],
                     bus0=edges[0].values,
                     bus1=edges[1].values,
                     suffix =" positive",
                     p_nom_extendable=True,
                     p_min_pu=0,
                     efficiency=config["transmission_efficiency"]["DC"]["efficiency_static"]* config["transmission_efficiency"]["DC"]["efficiency_per_1000km"]**(lengths/1000),
                     length=lengths,
                     capital_cost=cc)

        network.madd("Link",
                     edges[1] + '-' + edges[0],
                     bus0=edges[1].values,
                     bus1=edges[0].values,
                     suffix=" reversed",
                     p_nom_extendable=True,
                     p_min_pu=0,
                     efficiency=config["transmission_efficiency"]["DC"]["efficiency_static"]* config["transmission_efficiency"]["DC"]["efficiency_per_1000km"]**(lengths/1000),
                     length=lengths,
                     capital_cost=0)

    return network

if __name__ == '__main__':

    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('prepare_base_networks_2020',
                                   opts='ll',
                                   topology='current+Neighbor',
                                   pathway='exponential175',
                                   planning_horizons="2020")
    configure_logging(snakemake)

    network = prepare_network(snakemake.config)

    network.export_to_netcdf(snakemake.output.network_name)
