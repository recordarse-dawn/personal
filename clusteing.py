import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from matplotlib.colors import LinearSegmentedColormap

cool_metal = LinearSegmentedColormap.from_list(
    'nature3', ["#0f2d70", "#1c4c87", "#316098", "#5886bd", "#94acd9", '#e7e4f8','#f4dced',"#eab0cc","#de92ae", "#c85a72","#a12c3f"]
)

sns.set_theme(style='whitegrid',palette='muted')
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False #负号正常显示

#导入数据
df = pd.read_excel('D:/code/python/kmeans-competition-analysis/Original_Data.xlsx')

#columns赋新值
df.columns = [
    'sex', 'age','education','city','identity','income','frequency',
    'Q8_1','Q8_2','Q8_3','Q8_4','Q8_5',
    'Q9_1','Q9_2','Q9_3','Q9_4','Q9_5',
    'Q10_1','Q10_2','Q10_3','Q10_4','Q10_5',
    'Q11_1','Q11_2','Q11_3','Q11_4','Q11_5',
    'Q12_1','Q12_2','Q12_3','Q12_4','Q12_5',
    'Q13_1','Q13_2','Q13_3','Q13_4','Q13_5',
    'Q14_1','Q14_2','Q14_3','Q14_4','Q14_5',
    'Q15'
]

#第一组聚类Q8_Q9
group_a = df[['Q8_1','Q8_2','Q8_3','Q8_4','Q8_5',
              'Q9_1','Q9_2','Q9_3','Q9_4','Q9_5']]

scaler = StandardScaler()
group_a_scaled = scaler.fit_transform(group_a)#fit学习参数 transform标准化

print(group_a_scaled[:3])

#肘部法选择最优K+轮廓系数
inertias = []#空列表收集惯性值
silhouettes = []
k_range = range(2,7)

for k in k_range:
    kmeans = KMeans(n_clusters=k,random_state=42,n_init=10)#创建KMeans object
    kmeans.fit(group_a_scaled)#用标准化后数据训练，找到k个簇中心
    inertias.append(kmeans.inertia_)#存入惯性值
    score = silhouette_score(group_a_scaled,kmeans.labels_)
    silhouettes.append(score)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 左图：肘部法
axes[0].plot(k_range, inertias, 'o-', color='#2E86AB', linewidth=2, markersize=8)
axes[0].set_xlabel('K', fontsize=12)
axes[0].set_ylabel('惯性', fontsize=12)
axes[0].set_title('肘部法', fontsize=13)

# 右图：轮廓系数
axes[1].plot(k_range, silhouettes, 's-', color='#E84855', linewidth=2, markersize=8)
axes[1].set_xlabel('K', fontsize=12)
axes[1].set_ylabel('轮廓系数', fontsize=12)
axes[1].set_title('轮廓系数', fontsize=13)

fig.suptitle('品牌认知组 K值选择', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
#受李克特量表数据特性影响，轮廓系数整体偏低，但K=3时取得最大值，结合肘部法综合判断选取K=3

#聚类分析
kmeans_a = KMeans(n_clusters=3,random_state=42,n_init=10)
kmeans_a.fit(group_a_scaled)

df['cluster_a'] = kmeans_a.labels_

print(df['cluster_a'].value_counts().sort_index())

#计算各簇平均分
profile_a = df.groupby('cluster_a')[['Q8_1','Q8_2','Q8_3','Q8_4','Q8_5',
                                    'Q9_1','Q9_2','Q9_3','Q9_4','Q9_5']].mean()
print(profile_a.round(2))

#cluster_a
# 0    197
# 1    510
# 2    293
# 簇0：理性务实型（197人）— 认为品牌贴近生活，但不觉得特别专业
# 簇1：全面认可型（510人）— 对品牌整体印象最好，专业感和生活感都高
# 簇2：专业疏离型（293人）— 认可品牌专业性，但觉得和日常生活距离远

#热力图
# 给行列加上中文标签
profile_a.index = ['簇0 理性务实型\n(n=197)', '簇1 全面认可型\n(n=510)', '簇2 专业疏离型\n(n=293)']
profile_a.columns = ['专业智能', '年轻活力', '性价比', '设计颜值', '贴近生活',
                     '智能办公', '智能家居', '教育科技', '车载系统', '医疗健康']

# 画热力图
plt.figure(figsize=(10, 4))
sns.heatmap(profile_a, 
            annot=True,        # 显示数值
            fmt='.2f',         # 数值保留2位小数
            cmap=cool_metal,   # nature配色
            vmin=1, vmax=5,    # 颜色范围对应1-5分
            linewidths=0.5)    # 格子间加细线

plt.title('品牌认知组 各簇特征画像', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

#group_b聚类
group_b = df[['Q10_1','Q10_2','Q10_3','Q10_4','Q10_5',
            'Q11_1','Q11_2','Q11_3','Q11_4','Q11_5']]

scaler = StandardScaler()
group_b_scaled = scaler.fit_transform(group_b)

print(group_b_scaled[:3])

#肘部法+轮廓系数
inertias = []#空列表收集惯性值
silhouettes = []
k_range = range(2,7)

for k in k_range:
    kmeans = KMeans(n_clusters=k,random_state=42,n_init=10)#创建KMeans object
    kmeans.fit(group_b_scaled)#用标准化后数据训练，找到k个簇中心
    inertias.append(kmeans.inertia_)#存入惯性值
    score = silhouette_score(group_b_scaled,kmeans.labels_)
    silhouettes.append(score)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 左图：肘部法
axes[0].plot(k_range, inertias, 'o-', color='#2E86AB', linewidth=2, markersize=8)
axes[0].set_xlabel('K', fontsize=12)
axes[0].set_ylabel('惯性', fontsize=12)
axes[0].set_title('肘部法', fontsize=13)

# 右图：轮廓系数
axes[1].plot(k_range, silhouettes, 's-', color='#E84855', linewidth=2, markersize=8)
axes[1].set_xlabel('K', fontsize=12)
axes[1].set_ylabel('轮廓系数', fontsize=12)
axes[1].set_title('轮廓系数', fontsize=13)

fig.suptitle('产品需求组 K值选择', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

#聚类分析
kmeans_b = KMeans(n_clusters=2,random_state=42,n_init=10)
kmeans_b.fit(group_b_scaled)

df['cluster_b'] = kmeans_b.labels_

print(df['cluster_b'].value_counts().sort_index())

#计算各簇平均分
profile_b = df.groupby('cluster_b')[['Q10_1','Q10_2','Q10_3','Q10_4','Q10_5',
                                    'Q11_1','Q11_2','Q11_3','Q11_4','Q11_5']].mean()
print(profile_b.round(2))

#热力图
profile_b.index = ['簇0 积极需求型\n(n=553)', '簇1 品牌认同型\n(n=447)']
profile_b.columns = ['智能办公本', '健康管理', '沉浸教育', '车载助手', '智能家居',
                     '外观设计', '交互体验', '功能配置', '性价比', '品牌调性']

plt.figure(figsize=(12, 4))
sns.heatmap(profile_b, 
            annot=True,        # 显示数值
            fmt='.2f',         # 数值保留2位小数
            cmap=cool_metal,   # nature配色
            vmin=1, vmax=5,    # 颜色范围对应1-5分
            linewidths=0.5)    # 格子间加细线

plt.title('产品需求组 各簇特征画像', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()



#group_c聚类
group_c = df[['Q12_1','Q12_2','Q12_3','Q12_4','Q12_5',
            'Q13_1','Q13_2','Q13_3','Q13_4','Q13_5',
            'Q14_1','Q14_2','Q14_3','Q14_4','Q14_5']]

scaler = StandardScaler()
group_c_scaled = scaler.fit_transform(group_c)

print(group_c_scaled[:3])

#肘部法+轮廓系数
inertias = []#空列表收集惯性值
silhouettes = []
k_range = range(2,7)

for k in k_range:
    kmeans = KMeans(n_clusters=k,random_state=42,n_init=10)#创建KMeans object
    kmeans.fit(group_c_scaled)#用标准化后数据训练，找到k个簇中心
    inertias.append(kmeans.inertia_)#存入惯性值
    score = silhouette_score(group_c_scaled,kmeans.labels_)
    silhouettes.append(score)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 左图：肘部法
axes[0].plot(k_range, inertias, 'o-', color='#2E86AB', linewidth=2, markersize=8)
axes[0].set_xlabel('K', fontsize=12)
axes[0].set_ylabel('惯性', fontsize=12)
axes[0].set_title('肘部法', fontsize=13)

# 右图：轮廓系数
axes[1].plot(k_range, silhouettes, 's-', color='#E84855', linewidth=2, markersize=8)
axes[1].set_xlabel('K', fontsize=12)
axes[1].set_ylabel('轮廓系数', fontsize=12)
axes[1].set_title('轮廓系数', fontsize=13)

fig.suptitle('营销策略组 K值选择', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

#聚类分析
kmeans_C = KMeans(n_clusters=2,random_state=42,n_init=10)
kmeans_C.fit(group_c_scaled)

df['cluster_c'] = kmeans_C.labels_

print(df['cluster_c'].value_counts().sort_index())

#计算各簇平均分
profile_c = df.groupby('cluster_c')[['Q12_1','Q12_2','Q12_3','Q12_4','Q12_5',
                                    'Q13_1','Q13_2','Q13_3','Q13_4','Q13_5',
                                     'Q14_1','Q14_2','Q14_3','Q14_4','Q14_5']].mean()
print(profile_c.round(2))

#热力图
profile_c.index = ['簇0 全面改进型\n(n=555)', '簇1 渠道驱动型\n(n=445)']
profile_c.columns = ['产品设计', '品牌形象', '销售渠道', '客户服务', '市场推广',
                     '社媒互动', 'KOL合作', '线下活动', '广告内容', '品牌社群',
                     '线下体验', '电商旗舰', '社媒广告', '亲友推荐', '媒体评测']

plt.figure(figsize=(14, 4))
sns.heatmap(profile_c, 
            annot=True,        # 显示数值
            fmt='.2f',         # 数值保留2位小数
            cmap=cool_metal,   # nature配色
            vmin=1, vmax=5,    # 颜色范围对应1-5分
            linewidths=0.5,     # 格子间加细线
            annot_kws={'size': 8}
)   

plt.title('营销策略组 各簇特征画像', fontsize=14, fontweight='bold')
plt.xticks(fontsize=8, rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ============================================================
# 交叉分析
# ============================================================

#cluster_a
demo_vars = {
    'age':       {1:'16岁以下', 2:'16-30岁', 3:'30-45岁', 4:'45岁以上'},
    'education': {1:'初中及以下', 2:'高中/中专', 3:'大专/本科', 4:'研究生及以上'},
    'income':    {1:'2000以下', 2:'2000-5000', 3:'5000-10000', 4:'10000-30000', 5:'30000以上'},
    'identity':  {1:'在校学生', 2:'普通职工', 3:'专业人员', 4:'公务员', 5:'个体，私企等', 6:'管理层', 7:'已退休'},
    'sex':       {1:'女', 2:'男'},
}

# 中文标题对应
demo_titles = {
    'age': '年龄',
    'education': '学历',
    'income': '收入',
    'identity': '身份',
    'sex': '性别',
}

cluster_labels_a = {0:'簇0\n理性务实型', 1:'簇1\n全面认可型', 2:'簇2\n专业疏离型'}

for var, label_map in demo_vars.items():
    cross = df.groupby(['cluster_a', var]).size().unstack()
    cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
    cross_pct.columns = [label_map[k] for k in cross_pct.columns]
    cross_pct.index = [cluster_labels_a[k] for k in cross_pct.index]

    cross_pct.plot(kind='bar', stacked=True, figsize=(8, 5),
                   colormap='Blues', edgecolor='white', linewidth=0.5)
    plt.title(f'品牌认知组 × {demo_titles[var]}分布', fontsize=13, fontweight='bold')
    plt.ylabel('占比 (%)', fontsize=11)
    plt.xlabel('')
    plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'C:/Users/zwx/Desktop/粥粥/比赛考试项目/商赛聚类/figures/cross analysis/cross_a/cross_a_{var}.png', dpi=150, bbox_inches='tight')
    plt.show()

#cluster_b
demo_vars = {
    'age':       {1:'16岁以下', 2:'16-30岁', 3:'30-45岁', 4:'45岁以上'},
    'education': {1:'初中及以下', 2:'高中/中专', 3:'大专/本科', 4:'研究生及以上'},
    'income':    {1:'2000以下', 2:'2000-5000', 3:'5000-10000', 4:'10000-30000', 5:'30000以上'},
    'identity':  {1:'在校学生', 2:'普通职工', 3:'专业人员', 4:'公务员', 5:'个体，私企等', 6:'管理层', 7:'已退休'},
    'sex':       {1:'女', 2:'男'},
}

# 中文标题对应
demo_titles = {
    'age': '年龄',
    'education': '学历',
    'income': '收入',
    'identity': '身份',
    'sex': '性别',
}

cluster_labels_b = {0:'簇0\n积极需求型', 1:'簇1\n品牌认同型'}

for var, label_map in demo_vars.items():
    cross = df.groupby(['cluster_b', var]).size().unstack()
    cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
    cross_pct.columns = [label_map[k] for k in cross_pct.columns]
    cross_pct.index = [cluster_labels_b[k] for k in cross_pct.index]

    cross_pct.plot(kind='bar', stacked=True, figsize=(8, 5),
                   colormap='Blues', edgecolor='white', linewidth=0.5)
    plt.title(f'产品需求组 × {demo_titles[var]}分布', fontsize=13, fontweight='bold')
    plt.ylabel('占比 (%)', fontsize=11)
    plt.xlabel('')
    plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'C:/Users/zwx/Desktop/粥粥/比赛考试项目/商赛聚类/figures/cross analysis/cross_b/cross_b_{var}.png', dpi=150, bbox_inches='tight')
    plt.show()

#cluster_c
demo_vars = {
    'age':       {1:'16岁以下', 2:'16-30岁', 3:'30-45岁', 4:'45岁以上'},
    'education': {1:'初中及以下', 2:'高中/中专', 3:'大专/本科', 4:'研究生及以上'},
    'income':    {1:'2000以下', 2:'2000-5000', 3:'5000-10000', 4:'10000-30000', 5:'30000以上'},
    'identity':  {1:'在校学生', 2:'普通职工', 3:'专业人员', 4:'公务员', 5:'个体，私企等', 6:'管理层', 7:'已退休'},
    'sex':       {1:'女', 2:'男'},
}

# 中文标题对应
demo_titles = {
    'age': '年龄',
    'education': '学历',
    'income': '收入',
    'identity': '身份',
    'sex': '性别',
}

cluster_labels_c = {0:'簇0\n全面改进型', 1:'簇1\n渠道驱动型'}

for var, label_map in demo_vars.items():
    cross = df.groupby(['cluster_c', var]).size().unstack()
    cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
    cross_pct.columns = [label_map[k] for k in cross_pct.columns]
    cross_pct.index = [cluster_labels_c[k] for k in cross_pct.index]

    cross_pct.plot(kind='bar', stacked=True, figsize=(8, 5),
                   colormap='Blues', edgecolor='white', linewidth=0.5)
    plt.title(f'营销策略组 × {demo_titles[var]}分布', fontsize=13, fontweight='bold')
    plt.ylabel('占比 (%)', fontsize=11)
    plt.xlabel('')
    plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'C:/Users/zwx/Desktop/粥粥/比赛考试项目/商赛聚类/figures/cross analysis/cross_c/cross_c_{var}.png', dpi=150, bbox_inches='tight')
    plt.show()

#Q15交叉分析
#cluster_a
q15_labels = {1:'略微增加', 2:'适度增加', 3:'明显增加', 4:'大幅增加', 5:'不需要增加', 6:'不清楚/无意见'}

#cluster_a × Q15
cross_a_q15 = df.groupby(['cluster_a', 'Q15']).size().unstack()
cross_a_q15_pct = cross_a_q15.div(cross_a_q15.sum(axis=1), axis=0) * 100
cross_a_q15_pct.columns = [q15_labels[k] for k in cross_a_q15_pct.columns]
cross_a_q15_pct.index = [cluster_labels_a[k] for k in cross_a_q15_pct.index]

cross_a_q15_pct.plot(kind='bar', stacked=True, figsize=(8, 5),
                     colormap='Blues', edgecolor='white', linewidth=0.5)
plt.title('品牌认知组 × 研发投入态度', fontsize=13, fontweight='bold')
plt.ylabel('占比 (%)', fontsize=11)
plt.xlabel('')
plt.xticks(rotation=0)
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig(r'C:/Users/zwx/Desktop/粥粥/比赛考试项目/商赛聚类/figures/cross analysis/cross_Q15/cross_a_q15.png', dpi=150, bbox_inches='tight')
plt.show()

#cluster_b × Q15
cross_b_q15 = df.groupby(['cluster_b', 'Q15']).size().unstack()
cross_b_q15_pct = cross_b_q15.div(cross_b_q15.sum(axis=1), axis=0) * 100
cross_b_q15_pct.columns = [q15_labels[k] for k in cross_b_q15_pct.columns]
cross_b_q15_pct.index = [cluster_labels_b[k] for k in cross_b_q15_pct.index]

cross_b_q15_pct.plot(kind='bar', stacked=True, figsize=(8, 5),
                     colormap='Blues', edgecolor='white', linewidth=0.5)
plt.title('产品需求组 × 研发投入态度', fontsize=13, fontweight='bold')
plt.ylabel('占比 (%)', fontsize=11)
plt.xlabel('')
plt.xticks(rotation=0)
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig(r'C:/Users/zwx/Desktop/粥粥/比赛考试项目/商赛聚类/figures/cross analysis/cross_Q15/cross_b_q15.png', dpi=150, bbox_inches='tight')
plt.show()

#cluster_c × Q15
cross_c_q15 = df.groupby(['cluster_c', 'Q15']).size().unstack()
cross_c_q15_pct = cross_c_q15.div(cross_c_q15.sum(axis=1), axis=0) * 100
cross_c_q15_pct.columns = [q15_labels[k] for k in cross_c_q15_pct.columns]
cross_c_q15_pct.index = [cluster_labels_c[k] for k in cross_c_q15_pct.index]

cross_c_q15_pct.plot(kind='bar', stacked=True, figsize=(8, 5),
                     colormap='Blues', edgecolor='white', linewidth=0.5)
plt.title('营销策略组 × 研发投入态度', fontsize=13, fontweight='bold')
plt.ylabel('占比 (%)', fontsize=11)
plt.xlabel('')
plt.xticks(rotation=0)
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig(r'C:/Users/zwx/Desktop/粥粥/比赛考试项目/商赛聚类/figures/cross analysis/cross_Q15/cross_c_q15.png', dpi=150, bbox_inches='tight')
plt.show()