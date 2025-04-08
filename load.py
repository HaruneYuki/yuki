







# 该文件用于处理10G数据集，并生成统计结果和可视化结果
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from typing import Dict, Any, List
from tqdm import tqdm  # 添加进度条支持

# 配置全局参数
FOLDER = "10G_data/"
OUTPUT_DIR = "analysis_results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置全局字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 定义分析参数
COLS = ["id", "age", "income", "gender", "country", "credit_score", "is_active", "purchase_history"]
DTYPES = {
    "id": "int32",
    "age": "int8",
    "income": "float32",
    "gender": "category",
    "country": "category",
    "credit_score": "int16",
    "is_active": "bool"
}

RANGE_RULES = {
    "age": (0, 120),
    "income": (0, None),
    "credit_score": (300, 850)
}

# 初始化全局统计容器
class GlobalStats:
    def __init__(self):
        # 分箱配置
        self.age_bins = np.linspace(0, 120, 25)
        self.income_bins = np.geomspace(1000, 1e6, 20)
        self.credit_bins = np.linspace(300, 850, 15)
        
        # 累积容器
        self.age_dist = np.zeros(len(self.age_bins)-1)
        self.income_dist = np.zeros(len(self.income_bins)-1)
        self.scatter_samples = pd.DataFrame()
        self.country_counts = pd.Series(dtype=int)
        self.high_value_users = pd.DataFrame()
        self.high_value_country_counts = pd.Series(dtype=int)  # 存储完整国家统计

# 辅助函数
def safe_division(a, b):
    return a / b if b != 0 else 0

def count_outliers(df, column, lower, upper):
    """优化后的异常值计数"""
    col_data = df[column]
    if lower is not None and upper is not None:
        return ((col_data < lower) | (col_data > upper)).sum()
    elif lower is not None:
        return (col_data < lower).sum()
    elif upper is not None:
        return (col_data > upper).sum()
    return 0

# def stratified_sampling(df, sample_size=200):
#     quantile_bins = [0, 0.25, 0.5, 0.75, 0.95, 1.0]
#     df['strata'] = pd.qcut(df['income'], q=quantile_bins, labels=["Q1","Q2","Q3","Q4","Q5"])
    
#     samples = []
#     for stratum in ["Q1","Q2","Q3","Q4","Q5"]:
#         stratum_df = df[df["strata"] == stratum]
#         samples.append(
#             stratum_df.sample(
#                 n=min(sample_size, len(stratum_df)), 
#                 random_state=42
#             )
#         )
#     return pd.concat(samples)





def process_file(file_path: str, global_stats: GlobalStats, sample_size: int = 1000) -> Dict[str, Any]:
    """处理单个文件并更新全局统计"""
    # 读取数据（优化内存）
    df = pd.read_parquet(file_path, columns=COLS).astype(DTYPES)
    
    # --------------------------
    # 数据清洗
    # --------------------------
    # 处理缺失值
    df["age"] = df["age"].fillna(df["age"].median()).clip(*RANGE_RULES["age"])
    df["income"] = df["income"].fillna(df["income"].median()).clip(RANGE_RULES["income"][0])
    df["credit_score"] = df["credit_score"].fillna(df["credit_score"].median()).clip(*RANGE_RULES["credit_score"])
    
    # 解析JSON字段
    df["purchase_avg"] = df["purchase_history"].apply(
        lambda x: json.loads(x).get("average_price", 0) if pd.notnull(x) else 0
    )
    
    # --------------------------
    # 更新全局统计
    # --------------------------
    # 年龄分布
    age_counts, _ = np.histogram(df["age"], bins=global_stats.age_bins)
    global_stats.age_dist += age_counts
    
    # 收入分布
    income_counts, _ = np.histogram(df["income"], bins=global_stats.income_bins)
    global_stats.income_dist += income_counts
    
    # 散点图抽样（分层抽样）
    sample = df.sample(min(sample_size, len(df)), 
                      weights=df["income"]+1)  # 加权抽样保证高收入可见性
    global_stats.scatter_samples = pd.concat([global_stats.scatter_samples, sample])
    
    # 国家分布
    global_stats.country_counts = global_stats.country_counts.add(
        df["country"].value_counts(), fill_value=0
    ).astype(int)
    
    #高价值用户（使用条件筛选）
    # high_value = df[
    #     df["income"] > df["income"].quantile(0.8)
    # ]
    high_value = df[
        df["income"] > df["income"].quantile(0.8)
    ]
    global_stats.high_value_users = pd.concat([global_stats.high_value_users, high_value])

    # 当前文件的国家统计
    current_counts = high_value["country"].value_counts()
    
    # 累计到全局状态（关键操作）
    global_stats.high_value_country_counts = global_stats.high_value_country_counts.add(
        current_counts, 
        fill_value=0
    ).astype(int)
    
    # --------------------------
    # 生成文件级统计
    # --------------------------
    stats = {
        "file": os.path.basename(file_path),
        "total_users": len(df),
        "age_mean": df["age"].mean(),
        "income_median": df["income"].median(),
        "active_rate": df["is_active"].mean(),
        "high_value_ratio": len(high_value) / len(df)
    }
    
    # 释放内存
    del df
    return stats

def generate_visualizations(global_stats: GlobalStats):
    """生成全局可视化图表"""
    # --------------------------
    # 年龄分布直方图
    # --------------------------
    # 修改后的年龄分布可视化代码
    plt.figure(figsize=(12, 6))

    # 计算分箱中心点
    bin_centers = (global_stats.age_bins[:-1] + global_stats.age_bins[1:]) / 2

    # 使用条形图绘制
    plt.bar(
        x=bin_centers,
        height=global_stats.age_dist,
        width=np.diff(global_stats.age_bins),  # 自动计算条宽
        align='center',
        edgecolor='white'
    )

    plt.title("全局年龄分布")
    plt.xlabel("年龄")
    plt.ylabel("人数")
    plt.xticks(global_stats.age_bins[::2])  # 每两个分箱显示一个刻度
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"{OUTPUT_DIR}/global_age_dist.png")
    plt.close()

    # --------------------------
    # 收入-信用评分散点图
    # --------------------------
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x="income", 
        y="credit_score",
        data=global_stats.scatter_samples,
        alpha=0.3,
        hue=pd.cut(global_stats.scatter_samples["age"], 
                  bins=[0, 30, 50, 100],
                  labels=["青年", "中年", "老年"])
    )
    plt.title("收入与信用评分关系（全局抽样）")
    plt.xscale('log')  # 对数坐标显示收入
    plt.savefig(f"{OUTPUT_DIR}/global_income_credit.png")
    plt.close()
    
    # --------------------------
    # 国家分布（Top 15）
    # --------------------------
    plt.figure(figsize=(15, 8))
    top_countries = global_stats.country_counts.nlargest(15)
    sns.barplot(x=top_countries.values, y=top_countries.index)
    plt.title("用户国家分布 Top15")
    plt.savefig(f"{OUTPUT_DIR}/global_country_dist.png")
    plt.close()
    
    # --------------------------
    # 高价值用户分析
    # --------------------------
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x="country", 
        y="income",
        data=global_stats.high_value_users.nlargest(5000, "income"),
        showfliers=False
    )
    plt.xticks(rotation=45)
    plt.title("高价值用户收入分布（按国家）")
    plt.savefig(f"{OUTPUT_DIR}/global_high_value_income.png")
    plt.close()


    # --------------------------
    # 各年龄段高价值用户数量分布
    # --------------------------
    plt.figure(figsize=(12, 6))

    # 计算高价值用户的年龄分布
    age_counts, _ = np.histogram(global_stats.high_value_users['age'], bins=global_stats.age_bins)
    bin_centers = (global_stats.age_bins[:-1] + global_stats.age_bins[1:]) / 2

    # 绘制柱状图
    plt.bar(
        x=bin_centers,
        height=age_counts,
        width=np.diff(global_stats.age_bins),
        align='center',
        edgecolor='white',
        color='skyblue'
    )

    plt.title("高价值用户年龄分布")
    plt.xlabel("年龄")
    plt.ylabel("用户数量")
    plt.xticks(global_stats.age_bins[::2])  # 每两个分箱显示一个刻度
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"{OUTPUT_DIR}/high_value_users_age_distribution.png")
    plt.close()

    # --------------------------
    # 各年龄段平均收入（基于抽样数据）
    # --------------------------
    plt.figure(figsize=(12, 6))

    # 使用分层抽样数据计算各年龄段平均收入
    scatter_samples = global_stats.scatter_samples
    age_bins = global_stats.age_bins

    # 添加年龄分箱并计算平均收入
    scatter_samples['age_bin'] = pd.cut(scatter_samples['age'], bins=age_bins)
    avg_income = scatter_samples.groupby('age_bin')['income'].mean().values

    # 绘制柱状图
    plt.bar(
        x=bin_centers,
        height=avg_income,
        width=np.diff(age_bins),
        align='center',
        edgecolor='white',
        color='salmon'
    )

    plt.title("各年龄段平均收入（抽样数据）")
    plt.xlabel("年龄")
    plt.ylabel("平均收入（美元）")
    plt.xticks(age_bins[::2])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"{OUTPUT_DIR}/average_income_by_age_stratified.png")
    plt.close()



    # --------------------------
    # 完整国家高价值用户分布（新增）
    # --------------------------
    plt.figure(figsize=(15, max(8, len(global_stats.high_value_country_counts)*0.3)))
    sorted_counts = global_stats.high_value_country_counts.sort_values()
    
    # 创建颜色映射（数值越大颜色越深）
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(sorted_counts)))
    
    bars = plt.barh(
        y=sorted_counts.index.astype(str),
        width=sorted_counts.values,
        color=colors,
        edgecolor='black'
    )
    
    # 添加数据标签
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + max(sorted_counts)*0.01, 
            bar.get_y() + bar.get_height()/2,
            f'{int(width):,}',
            va='center',
            ha='left'
        )
    
    plt.title("各国高价值用户数量完整分布")
    plt.xlabel("高价值用户数量")
    plt.ylabel("国家")
    plt.xlim(0, max(sorted_counts)*1.15)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/global_full_high_value_country.png")
    plt.close()




# --------------------------
# 主处理流程
# --------------------------
def main():
    global_stats = GlobalStats()
    summary_stats = []
    
    # 获取文件列表
    files = [os.path.join(FOLDER, f) for f in os.listdir(FOLDER) 
            if f.endswith(".parquet")]
    
    # 分文件处理
    for file in tqdm(files, desc="Processing files"):
        stats = process_file(file, global_stats)
        summary_stats.append(stats)
        
        # 定期清理内存
        if len(global_stats.scatter_samples) > 50000:
            global_stats.scatter_samples = global_stats.scatter_samples.sample(50000)
    
    # 保存统计结果
    pd.DataFrame(summary_stats).to_csv(f"{OUTPUT_DIR}/file_stats_summary.csv", index=False)
    global_stats.high_value_users.to_parquet(f"{OUTPUT_DIR}/global_high_value_users.parquet")
    
    # 生成可视化
    generate_visualizations(global_stats)
    
    # 生成统计报告
    report = {
        "total_users": int(global_stats.age_dist.sum()),
        "avg_age": float(  # 转换为 Python float
            np.average(
                (global_stats.age_bins[:-1] + global_stats.age_bins[1:])/2,
                weights=global_stats.age_dist
            )
        ),
        "median_income": float(  # 转换为 Python float
            np.median(global_stats.scatter_samples["income"].astype("float64"))
        ),
        "high_value_users": int(  # 显式转换为 Python int
            len(global_stats.high_value_users)
        )
    }

    with open(f"{OUTPUT_DIR}/summary_report.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
