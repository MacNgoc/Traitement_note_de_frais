#import sys
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

def list_all_log_files(path_to_directory):
    dirs = os.listdir(path_to_directory)
    list_files = [os.path.join(path_to_directory, file) for file in dirs]
    return list_files


def merge_in_a_file(path_to_directory):
    dirs = list_all_log_files(path_to_directory)
    with open('file.log','w') as file:
        for file_name in dirs:
            with open(file_name) as f:
                file.write(f.read())

def read_log_file(path_to_directory):

    lines_avg = []
    with open('file.log') as f:
        for i, line in enumerate(f):
            if "avg" in line:
                lineParts = line.split(',')
                lines_avg.append((int(lineParts[0].split(':')[0]), float(lineParts[1].split()[0]) ))
            
            if "mean_average_precision" in line:
                lineParts = line.split('=')
                lines_avg.append(('mAP', float(lineParts[1])))
               
            if "class_id" in line:
                lineParts = line.split(',')
                lines_avg.append( (lineParts[0], lineParts[1], lineParts[2].split()[2]) )
                

    


    ap_class_0 = []
    ap_class_1 = []
    ap_class_2 = []
    ap_class_3 = []
    mAp = []
    avg_loss = []
    for i,x in enumerate(lines_avg):
        if 'class_id = 0' in x: # Etablissement
            iteration = lines_avg[i-1][0]
            map_class = float(x[2].split('%')[0])
            ap_class_0.append((iteration, map_class))
        elif 'class_id = 1' in x: # Montant Total
            iteration = lines_avg[i-2][0]
            map_class = float(x[2].split('%')[0])
            ap_class_1.append((iteration, map_class))
        elif 'class_id = 2' in x: # TVA
            iteration = lines_avg[i-3][0]
            map_class = float(x[2].split('%')[0])
            ap_class_2.append((iteration, map_class))
        elif 'class_id = 3' in x: # Date
            iteration = lines_avg[i-4][0]
            map_class = float(x[2].split('%')[0])
            ap_class_3.append((iteration, map_class))
        elif 'mAP' in x:
            iteration = lines_avg[i-5][0]
            mAp.append((int(iteration), float(x[1]) ))
        else:
            avg_loss.append(x)



            
    
    # Convert list into array
    array_mAp = np.array(mAp)
    array_avg_loss = np.array(avg_loss)
    

    ######## Draw Avg Loss and mAP #####################
    print(" Draw the avg loss")
    fig, ax = plt.subplots()
    
    color = 'tab:blue'
    ax.plot(array_avg_loss[:,0], array_avg_loss[:,1], '+',  markersize = 0.25, color = color)
    ax.set_ylim(0, 5)
    ax.set_xlim(0, 10000)
    ax.set_xlabel('Batch number')
    ax.set_ylabel('Avg Loss', color=color)
    
    ## draw grid ###############
    grid_x_ticks = np.arange(0, 10000, 100)
    grid_y_ticks = np.arange(0., 5., 0.05)
    x_ticks = np.arange(0., 10000, 1000)
    y_ticks = np.arange(0., 5., 0.5)

    #ax.set_axisbelow(True)
    #ax.minorticks_on()
    ax.set_xticks(grid_x_ticks , minor=True)
    ax.set_yticks(grid_y_ticks , minor=True)
    #ax.tick_params(which='both', bottom='off', top='off', left='off', right='off' )
    ax.grid(which='major', alpha = 0.3) # set two axis
    ax.grid(which='minor', alpha=0.2, linestyle='-')
    
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    
    
   



    ax2  = ax.twinx()
    color = 'tab:red'
    ax2.plot(array_mAp[:, 0], array_mAp[:, 1],  linewidth = 0.8, color = color)
    ax2.set_ylabel('mAP', color = color)
    ax2.set_ylim(0.5, 1)
    ax2.tick_params(axis='y', labelcolor=color)


    
    



    fig.tight_layout()
    plt.show()
    name = os.path.basename(path_to_directory)
    fig.savefig(name + '_avg_loss.png', dpi = 500)

    print('Done! Plot saved as {}_avg_loss.png'.format(name))
    plt.close()




    ##### Convertir des listes en array
    array_class_0 = np.array(ap_class_0)
    array_class_1 = np.array(ap_class_1)
    array_class_2 = np.array(ap_class_2)
    array_class_3 = np.array(ap_class_3)



    ######## Plot AP for each class ####################
    print(" Draw the ")
    fig2, ax = plt.subplots()
    
    ax.plot(array_class_0[:, 0], array_class_0[:, 1], 'r',  linewidth = 0.8, label = 'Etablissement')
    ax.plot(array_class_0[:, 0], array_class_1[:, 1], 'b',  linewidth = 0.8, label = 'Montant total')
    ax.plot(array_class_0[:, 0], array_class_2[:, 1], 'g',  linewidth = 0.8, label = 'TVA')
    ax.plot(array_class_0[:, 0], array_class_3[:, 1], 'orange',  linewidth = 0.8, label= 'Date')
    ax.set_xlabel('Batch number')
    ax.set_ylabel('AP(%)')
    leg = ax.legend()


    grid_x_ticks = np.arange(0, 10000, 100)
    grid_y_ticks = np.arange(50, 100, 5)
    x_ticks = np.arange(0., 10000., 1000)
    y_ticks = np.arange(0, 100, 10)

    ax.set_xticks(grid_x_ticks , minor=True)
    ax.set_yticks(grid_y_ticks , minor=True)
    ax.grid(which='major', alpha=0.3) # set two axis
    ax.grid(which='minor', alpha=0.2, linestyle='-')
    ax.set_xlim(1000, 10000)
    #plt.xticks(x_ticks)
    #plt.yticks(y_ticks)

    plt.show()
    fig2.savefig(name +'_AP_class.png', dpi = 500)

    print('Done! Plot saved as {} _AP_class.png'.format(name))
    plt.close()
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-log', '--log_path', required=True, help='path to log folder')
    args = ap.parse_args()


    if os.path.exists('file.log'):
        os.remove('file.log')
    
    merge_in_a_file(args.log_path)
    read_log_file(args.log_path)


if __name__== "__main__":
    main()


