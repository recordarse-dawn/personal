import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] #设置中文字体为微软雅黑
plt.rcParams['axes.unicode_minus'] = False #负号正常显示

#导入数据
df = pd.read_excel('D:/code/python/kmeans-competition-analysis/Original_Data.xlsx')

#查看数据基本信息
print(df.info())
print(df.shape)
print(df.head())

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
print(df.columns.tolist())
print(df.head(10))

#值域合法性检查(可省)
valid_values = {
    'sex' : [1,2],
    'age' : [1,2,3,4],
    'education' : [1,2,3,4],
    'city' : [1,2,3,4],
    'identity' : [1,2,3,4,5,6,7],
    'income' : [1,2,3,4,5],
    'frequency' : [1,2,3,4]
}

#column拿列名，valid拿合法值，actual拿实际值，invalid拿异常值
for column,valid in valid_values.items():
    actual = df[column].unique().tolist()#取出各列所有不重复值转为列表，得区间
    invalid = [v for v in actual if v not in valid]
    if invalid:
        print(f'{column}:实际值={actual},异常值={invalid}')

#Likert scale检查(可省)
#startswith同时传多个前缀必须用元组，不能用列表
likert_columns = [col for col in df.columns if col.startswith(('Q8','Q9','Q10','Q11','Q12','Q13','Q14'))]

print(likert_columns)
print(len(likert_columns))

for column in likert_columns:
    actual = df[column].unique().tolist()
    invalid = [v for v in actual if v not in [1,2,3,4,5]]
    if invalid:
        print(f'{column}:实际值={actual},异常值={invalid}')

#检查Q15
print(df['Q15'].unique().tolist())

#特征描述+可视化
labels = {
    'sex':       {1: '女', 2: '男'},
    'age':       {1: '16岁以下', 2: '16-30岁', 3: '30-45岁', 4: '45岁以上'},
    'education': {1: '高中及以下', 2: '大专', 3: '本科', 4: '硕士及以上'},
    'city':      {1: '一线城市', 2: '二线城市', 3: '三线城市', 4: '县城或农村'},
    'identity':  {1: '在校学生', 2: '普通职工', 3: '专业人员', 4: '公务员', 5: '个体、私企等', 6: '企业管理层', 7: '已退休'},
    'income':    {1: '2000元以下', 2: '2000-5000元', 3: '5000-10000元', 4: '10000-30000元', 5: '30000元以上'},
    'frequency': {1: '从不购买', 2: '偶尔购买', 3: '经常购买', 4: '频繁购买'},
}

# 人口特征批量画图
demo_cols = ['sex', 'age', 'education', 'city', 'identity', 'income', 'frequency']
titles = ['性别分布', '年龄分布', '学历分布', '城市分布', '身份分布', '收入分布', '购买频率']

fig, axes = plt.subplots(2, 4, figsize=(30, 10), facecolor="#F4F7F9")#2行4列，共8个子图位置
fig.suptitle('受访者基础信息特征描述', fontsize=16, fontweight='bold', color='#333333')

for i, (col, title) in enumerate(zip(demo_cols, titles)):
    ax = axes[i // 4][i % 4]  # 定位到第几行第几列
    counts = df[col].value_counts().sort_index()
    x_labels = [labels[col][k] for k in counts.index]

    # 定义一组清冷金属感色系，每张图内部柱子深浅不同
    metal_colors = [
    '#B8D4E8', '#8FBDD3', '#6AA4BE', '#4A8BAA', '#2E7096', '#1A5A7A', '#0D4460'
    ]

    bars = ax.bar(x_labels, counts.values,
              color=metal_colors[:len(counts)],
              width=0.5, edgecolor='white', linewidth=1.2)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 3,
                str(int(height)), ha='center', fontsize=9, color='#555555')

    ax.set_title(title, fontsize=12, fontweight='bold', color='#333333')
    ax.set_facecolor('#F8F9FA')
    ax.tick_params(axis='x', labelsize=8, rotation=15)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[1][3].set_visible(False)  # 第8个格子空着
plt.tight_layout()
plt.savefig('demographics.png', dpi=150, bbox_inches='tight')#保存图像，dpi=150保证清晰，bbox_inches='tight'去掉多余空白
plt.show()