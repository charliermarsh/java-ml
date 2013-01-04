import csv
import sys
import matplotlib.pyplot as plt

res = csv.reader(open(sys.argv[1]), delimiter=",")

cTrees = -1
current = [] #[numTrees, [train error], [test error]
total = [[],[],[]]

def appendPoint():
    total[0].append(cTrees)
    total[1].append(100-sum(current[1])/len(current[1]))
    total[2].append(100-sum(current[2])/len(current[2]))

for row in res:
    if (not row[0][0].isdigit()):
        continue
    #print(row)
    if (int(row[0]) != cTrees):
        if (cTrees > 0):
            appendPoint()
        current = [int(row[0]), [], []]
        cTrees = current[0]
    current[1].append(float(row[2])) #Test error
    current[2].append(float(row[3])) #Training error

#for i in range(len(total[0])):
    #print(total[0][i], end='\t')
    #print(total[1][i], end='\t')
    #print(total[2][i])


if (cTrees > 0):
    appendPoint()
    plt.ylabel("Error")
    plt.xlabel("Number trees")
    ax = plt.subplot(1,1,1)
    p1 = ax.plot(total[0], total[1],color='black', label="Test")
    p2 = ax.plot(total[0], total[2],color='black', ls=':', label="Train")
    handles,labels = ax.get_legend_handles_labels()
    plt.legend(handles,labels,loc=4)
    #plt.ylim(0,100)

    plt.show()
    #plt.savefig(base+"plot")
