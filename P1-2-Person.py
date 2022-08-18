import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Times New Roman']
# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

index = 'GRA_deCor'
data = np.load('data/data/'+index+'_DATA.npy')
data = np.transpose(data, (1, 0))
name = np.load('data/data/'+index+'_NAME.npy')
print('数据集维度：', data.shape)
print('ID维度：', name.shape)

df = pd.DataFrame(data)
df.columns = name
dfData = df.corr().abs()

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 8, }

# YlGnBu，OrRd,RdBu
with sns.axes_style("white"):
    fig, ax = plt.subplots()
    h = sns.heatmap(dfData,
                    cmap="OrRd",
                    linewidths=0.000,
                    cbar=False,
                    xticklabels=1)
    sns.set(font_scale=1.5)
    ax.set_xticklabels(list(name), rotation=45, fontproperties=font,horizontalalignment='right')
    ax.set_yticklabels(list(name), fontproperties=font)
    cb = h.figure.colorbar(h.collections[0])
    cb.ax.tick_params(labelsize=8)
plt.savefig('data/'+index+'20.svg', dpi=1000,format="svg",bbox_inches='tight')
plt.show()
