import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import chi2
import seaborn as sns
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

processed_data = np.abs(np.load('data/data/RT_DATA.npy'))
processed_name = np.load('data/data/RT_NAME.npy')


for i in range(processed_data.shape[0]):
    for j in range(processed_data.shape[1]):
        if processed_data[i][j] == 0:
            processed_data[i][j] = 1e-8


kf = []
for i in range(processed_data.shape[0]):
    kfz = []
    for j in range(processed_data.shape[0]):
        data = np.transpose(np.array([processed_data[:,i],processed_data[:,j]]),(1,0))
        g = chi2_contingency(data)
        kfz.append(g[1])
    kf.append(kfz)
dfData = pd.DataFrame(kf)
print(dfData.shape)

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 12, }

index = 'rt-k'
#YlGnBu OrRd PuBuGn Greens Greens_r
with sns.axes_style("white"):
    fig, ax = plt.subplots(figsize=(25,5))

    mask = np.zeros_like(dfData, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    h = sns.heatmap(dfData,
                    mask=mask,
                    cmap="Greens_r",
                    linewidths=0.003,
                    cbar=False,
                    xticklabels=1)
    sns.set(font_scale=1.5)
    ax.set_xticklabels(list(processed_name), rotation=30, fontproperties=font,horizontalalignment='right')
    ax.set_yticklabels(list(processed_name), rotation=360, fontproperties=font)
    cb = h.figure.colorbar(h.collections[0])
    cb.ax.tick_params(labelsize=12)
plt.savefig('data/figs/'+index+'.svg', dpi=1000,format="svg",bbox_inches='tight')
# plt.show()