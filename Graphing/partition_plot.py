import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
def part_plot(theta_minus, theta_plus, theta_undefined, region, test_function, method):
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111) 
    plt.title(method + '_'+ "subregions")
    plt.xlim(region[0][0], region[0][1]) 
    plt.ylim(region[1][0], region[1][1])
    
    for level in theta_minus.keys():
        minus_level = theta_minus[level].copy()
        for key in minus_level.keys(): 
    
            ax.add_patch(
                patches.Rectangle(
                    (minus_level[key][0][0],minus_level[key][1][0]), # (x,y)
                    minus_level[key][0][1]-minus_level[key][0][0], # weight
                    minus_level[key][1][1]-minus_level[key][1][0],  # height
                    alpha=0.05,
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
                    alpha=0.05,
                    facecolor='g',
                    edgecolor='black'
                )
            )

    for key in theta_undefined.keys():
    
        ax.add_patch(
            patches.Rectangle(
                (theta_undefined[key][0][0],theta_undefined[key][1][0]), # (x,y)
                theta_undefined[key][0][1]-theta_undefined[key][0][0], # weight
                theta_undefined[key][1][1]-theta_undefined[key][1][0],  # height
                alpha=0.05,
                facecolor='b',
                edgecolor='black'
            )
        )
    xx = np.arange(region[0][0],region[0][1],0.05)
    yy = np.arange(region[1][0],region[1][1],0.05)
    a= test_function.replace('X[1]', 'Y')
    b = a.replace('X[0]', 'X')
    X, Y  = np.meshgrid(xx, yy)
    Z = eval(b) 
    contour = plt.contour(X,Y,Z,[0],colors='k')
    plt.show()