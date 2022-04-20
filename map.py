import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import csv
from scipy import interpolate
import statistics

DataIn = csv.reader(open('data/coordinate.csv', 'r'))
x, y, z = [], [], []
for row in DataIn:
    x.append(eval(row[1]))
    y.append(eval(row[2]))
    z.append(eval(row[3]))
print(min(x), max(x))
print(min(y), max(y))
print(min(z), max(z))
x = np.array(x)
y = np.array(y)
z = np.array(z)
#####################################################
n = 100
X = np.linspace(120.5, 122, n)
Y = np.linspace(30.5, 32, n)
X, Y = np.meshgrid(X, Y)
Z = np.zeros([n, n])
step_x = 1.5 / n
step_y=1.5/n

tmp = np.zeros([n, n, 2])
for k in range(len(x)):
    i = 0
    while x[k] > 120.5 + step_x*(i + 1):
        i += 1
    j = 0
    while y[k] > 30.5 + step_y*(j + 1):
        j += 1
    tmp[j][i][0] += z[k]
    tmp[j][i][1] += 1
print('done')
for i in range(n):
    for j in range(n):
        if tmp[i][j][1]:
            Z[i][j] = tmp[i][j][0]/tmp[i][j][1]

edge = 8
for i in range(n):
    for j in range(n):
        if 1:
        # if not Z[i][j]:
            start1 = i-edge if i>=edge else 0
            end1 = i+edge if i<100-edge else 99
            start2 = j-edge if j>=edge else 0
            end2 = j+edge if j<100-edge else 99
            node = []
            for a in range(start1, end1+1):
                for b in range(start2, end2+1):
                    if Z[a][b]:
                        node.append(Z[a][b])
            if node:
                Z[i][j] = statistics.mean(node)

CS=plt.contourf(X, Y, Z, 12, alpha=0.75, cmap=plt.cm.autumn_r)
C = plt.contour(X, Y, Z, 12, colors='black', linewidth=.03)
plt.xticks(np.arange(120.5, 122,0.2))  # Set label locations.
plt.yticks(np.arange(30.5, 32,0.2))
nm, lbl = CS.legend_elements()
plt.legend(nm, lbl, title= 'price per square-meter (Yuan)', fontsize= 6,loc='center left',bbox_to_anchor=(1, 0.5))
plt.title('Average Price Heatmap of Properties in Shanghai')
plt.xlabel('longitude')
plt.ylabel('latitude')
import seaborn as sns
sns.set(context="notebook", style="whitegrid",
        rc={"axes.axisbelow": False})
plt.tight_layout()
plt.savefig("heatmap.png")
plt.show()
# #####################################################
# gammas = [0.8, 0.5, 0.3]
# fig, axes = plt.subplots(nrows=2, ncols=2)
# axes[0, 0].set_title('Linear normalization')
# axes[0, 0].hist2d(x, y, bins=100)
# for ax, gamma in zip(axes.flat[1:], gammas):
#     ax.set_title('Power law $(\gamma=%1.1f)$' % gamma)
#     ax.hist2d(x, y,
#               bins=100, norm=mcolors.PowerNorm(gamma))
# fig.tight_layout()
# plt.show()
# #####################################################
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_trisurf(x, y, z, )
# ax.set_title('3-d visualization of Shanghai average property price by apartments')
# plt.savefig('3d-map-1.png')
# plt.show()
# #####################################################
# ax = plt.subplot(111, projection='3d')
# ax.scatter(x, y, z, cmap='rainbow', alpha=0.4)
#
# ax.set_zlabel('Z')
# ax.set_ylabel('Y')
# ax.set_xlabel('X')
# ax.set_title('3-d visualization of Shanghai average property price by apartments')
# plt.savefig('3d-map-2.png')
# plt.show()
# #####################################################
