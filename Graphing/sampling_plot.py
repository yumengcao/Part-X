import numpy as np
import matplotlib.pyplot as plt


def sample_plot1(sample_all:np.array, rob: np.array,  method: str, group: str):
    #CSV = pd.read_csv('/Users/candicetsao/Desktop/rose_cont_bo/points45.csv', header = None)#, names = ['x','y'])
    nx = []
    ny = []
    px = []
    py = []
    
    p = np.where(rob > 0)[0]
    n = np.where(rob < 0)[0]
    #print(p)       
    for index in p:
        px.append(sample_all[index][0])
        py.append(sample_all[index][1])

    for index in n:
        nx.append(sample_all[index][0])
        ny.append(sample_all[index][1])


    plt.figure(figsize=(8, 8))
    plt.scatter(nx, ny, color='red',s=2, label='negative')
    plt.scatter(px, py, color='green',s=2, label='positive')
    #plt.legend(loc=(1, 0))
    plt.title(method + '_'+ group + "_" + 'input samples')
    plt.show()
    #plt.savefig('gold sample distribution3.png')  #########modify names
    
def sample_plot(ax, sample_all: np.array, rob: np.array):
    px = sample_all[rob > 0]
    nx = sample_all[rob < 0]

    ax.scatter(nx[:, 0], nx[:, 1], color='red', s=2, label='negative')
    ax.scatter(px[:, 0], px[:, 1], color='green', s=2, label='positive')