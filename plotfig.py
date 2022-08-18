import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
import xlrd

ER_activity = xlrd.open_workbook('data/ret/wucha.xls')
# table_ER_activity = ER_activity.sheet_by_name('My Worksheet')
# Caco_2 = table_ER_activity.col_values(0)
# CYP3A4 = table_ER_activity.col_values(1)
# hERG = table_ER_activity.col_values(2)
# HOB = table_ER_activity.col_values(3)
# MN = table_ER_activity.col_values(4)

table_ER_activity = ER_activity.sheet_by_name('1')
a = table_ER_activity.col_values(0)
b = table_ER_activity.col_values(1)
c = table_ER_activity.col_values(2)

font1 = {'family': 'Times New Roman',
          'weight': 'normal',
          'size': 15,
          }


plt.errorbar([i+1 for i in range(len(a))], a, yerr=[c[i]-b[i] for i in range(len(b))], fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)
plt.tick_params(labelsize=23)
plt.show()



# font1 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 15,
#          }
#
# figure, ax = plt.subplots()
# plt.plot([i + 1 for i in range(len(Caco_2))], Caco_2, 'b-', color='b', label='AAA', linewidth=1.5,
#          markerfacecolor='none', markevery=25)
# plt.plot([i + 1 for i in range(len(Caco_2))], CYP3A4, 'x-.', color='black', label='BBB', linewidth=1.5,
#          markerfacecolor='none', markevery=25)
# plt.plot([i + 1 for i in range(len(Caco_2))], hERG, 'o-.', color='r', label='BBB', linewidth=1.5,
#          markerfacecolor='none', markevery=25)
# plt.plot([i + 1 for i in range(len(Caco_2))], HOB, '<-.', color='g', label='BBB', linewidth=1.5,
#          markerfacecolor='none', markevery=25)
# plt.plot([i + 1 for i in range(len(Caco_2))], MN, 's-', color='y', label='BBB', linewidth=1.5,
#          markerfacecolor='none', markevery=25)
# plt.legend(prop=font1)
# plt.grid(axis="y", linestyle='--')
# plt.tick_params(labelsize=23)
# labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# plt.show()
