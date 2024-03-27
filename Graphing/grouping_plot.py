import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.patches as mpatches 

def group_plot(grouping, theta_minus, theta_plus, region, test_function, method):
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111) 
    plt.title(method + '_'+ "subregions")
    plt.xlim(region[0][0], region[0][1]) 
    plt.ylim(region[1][0], region[1][1])
    
    g1_region = grouping['group1'].copy()
    for key in g1_region.keys():
        ax.add_patch(
            patches.Rectangle(
                (g1_region[key][0][0], g1_region[key][1][0]), # (x,y)
                g1_region[key][0][1] - g1_region[key][0][0], # weight
                g1_region[key][1][1] - g1_region[key][1][0],  # height
                alpha = .5,
                facecolor='r',
                edgecolor='black'
            )
        )
    for level in theta_minus.keys():
        minus_level = theta_minus[level].copy()
        for key in minus_level.keys(): 
    
            ax.add_patch(
                patches.Rectangle(
                    (minus_level[key][0][0],minus_level[key][1][0]), # (x,y)
                    minus_level[key][0][1]-minus_level[key][0][0], # weight
                    minus_level[key][1][1]-minus_level[key][1][0],  # height
                    alpha= .5,
                    facecolor='r',
                    edgecolor='black'
                )
            )
    for level in theta_plus.keys():
        plus_level = theta_plus[level].copy()
        for key in plus_level.keys(): 
    
            ax.add_patch(
                patches.Rectangle(
                    (plus_level[key][0][0],plus_level[key][1][0]), # (x,y)
                    plus_level[key][0][1]-plus_level[key][0][0], # weight
                    plus_level[key][1][1]-plus_level[key][1][0],  # height
                    alpha= .5,
                    facecolor='g',
                    edgecolor='black'
                )
            )
    g2_region = grouping['group2'].copy()
    for key in g2_region.keys():
        ax.add_patch(
            patches.Rectangle(
                (g2_region[key][0][0], g2_region[key][1][0]), # (x,y)
                g2_region[key][0][1] - g2_region[key][0][0], # weight
                g2_region[key][1][1] - g2_region[key][1][0],  # height
                alpha = .5,
                facecolor='orange',
                edgecolor='black'
            )
        )
    g3_region = grouping['group3'].copy()
    for key in g3_region.keys():
        ax.add_patch(
            patches.Rectangle(
                (g3_region[key][0][0], g3_region[key][1][0]), # (x,y)
                g3_region[key][0][1] - g3_region[key][0][0], # weight
                g3_region[key][1][1] - g3_region[key][1][0],  # height
                alpha = .5,
                facecolor='pink',
                edgecolor='black'
            )
        )
    
    g6_region = grouping['group6'].copy()
    for key in g6_region.keys():
        ax.add_patch(
            patches.Rectangle(
                (g6_region[key][0][0], g6_region[key][1][0]), # (x,y)
                g6_region[key][0][1] - g6_region[key][0][0], # weight
                g6_region[key][1][1] - g6_region[key][1][0],  # height
                alpha = .5,
                facecolor='g',
                edgecolor='black'
            )
        )
    g4_region = grouping['group4'].copy()
    for key in g4_region.keys():
        ax.add_patch(
            patches.Rectangle(
                (g4_region[key][0][0], g4_region[key][1][0]), # (x,y)
                g4_region[key][0][1] - g4_region[key][0][0], # weight
                g4_region[key][1][1] - g4_region[key][1][0],  # height
                alpha = .5,
                facecolor='purple',
                edgecolor='black'
            )
        )
    g5_region = grouping['group5'].copy()
    for key in g5_region.keys():
        ax.add_patch(
            patches.Rectangle(
                (g5_region[key][0][0], g5_region[key][1][0]), # (x,y)
                g5_region[key][0][1] - g5_region[key][0][0], # weight
                g5_region[key][1][1] - g5_region[key][1][0],  # height
                alpha = .5,
                facecolor='olive',
                edgecolor='black'
            )
        )
    g7_region = grouping['group7'].copy()
    for key in g7_region.keys():
        ax.add_patch(
            patches.Rectangle(
                (g7_region[key][0][0], g7_region[key][1][0]), # (x,y)
                g7_region[key][0][1] - g7_region[key][0][0], # weight
                g7_region[key][1][1] - g7_region[key][1][0],  # height
                alpha = .5,
                facecolor='yellow',
                edgecolor='black'
            )
        )
    xx = np.arange(region[0][0],region[0][1],0.05)
    yy = np.arange(region[1][0],region[1][1],0.05)
    a= test_function.replace('X[1]', 'Y')
    b = a.replace('X[0]', 'X')
    X, Y  = np.meshgrid(xx, yy)
    Z = eval(b) 
    plt.contour(X,Y,Z,[0],colors='k')
    g_1 = mpatches.Patch(color='r', label='group 1') 
    g_2 = mpatches.Patch(color='orange', label='group 2') 
    g_3 = mpatches.Patch(color='pink', label='group 3') 
    g_4 = mpatches.Patch(color='purple', label='group 4') 
    g_5 = mpatches.Patch(color='olive', label='group 5')  
    g_6 = mpatches.Patch(color='green', label='group 6')
    g_7 = mpatches.Patch(color='yellow', label='group 7')  
    plt.legend(handles=[g_1, g_2, g_3, g_4, g_5, g_6, g_7]) 
    #plt.legend(['red', 'orange', 'pink', 'purple', 'oliver', 'green'])#, ['g1', 'g2', 'g3', 'g4', 'g5', 'g6'], loc='best',
           #title='Groups')

    plt.show()