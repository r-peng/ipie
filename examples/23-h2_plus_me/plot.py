import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True,threshold=10000000,linewidth=10000)
#import scipy
#M = 50
#N = 25
#for _ in range(100):
#    K = np.random.rand(M,M)
#    K -= K.T
#    C = scipy.linalg.expm(K)[:,:N]
#    K = np.random.rand(M,M)
#    K -= K.T
#    u = scipy.linalg.expm(K)[:,0]
#    Cu = np.dot(C.T,u)
#    print(np.linalg.norm(Cu))
#exit()

def read_column(filename):
    E = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.split(',')
            if line[0][:len('step=')]!='step=':
                continue
            #if line[0]=='Block':
            #    continue
            e = line[1].split('=')[-1]
            E.append(float(e)) 
    return np.array(E)

colors = 'blue','green','orange','pink','grey'

R = '1.9'
Eref = {'0.5':-1.4695117, 
        '0.6':-1.380611314,
        '0.7':-1.300982543,
        '0.8':-1.230289652,
        '0.9':-1.167887975,
        '1.0':-1.112973107,
        '1.1':-1.064661495,
        '1.2':-1.022065515,
        '1.3':-0.984350181,
        '1.4':-0.95076755,
        '1.5':-0.920675212,
        '1.6':-0.893541768,
        '1.7':-0.86894059,
        '1.8':-0.846535318,
        '1.9':-0.826061833,
        }[R]
colors = {'0.05':'pink','0.1':'blue','0.15':'green','0.2':'orange','0.4':'cyan'}
files = ('0.1',5000),('0.15',5000),('0.2',2000),('0.4',2000),
files = ('0.05',5000),('0.1',5000),('0.15',5000),('0.2',5000),
Nangle,Nr = 20,10
Nangle,Nr = 30,16
Nangle,Nr = 40,20
fig,ax = plt.subplots(nrows=1,ncols=1)
for (d,Nstep) in files:
    fname = f'Nangle={Nangle}_Nr={Nr}_d={d}_Nstep={Nstep}.out'
    E = read_column(fname)
    ax.plot(np.arange(E.size),E,linestyle='-',color='tab:'+colors[d],linewidth=1,label=f'd={d}')
ax.plot((0,E.size),(Eref,Eref),linestyle='-',color='k')
    
ax.set_xlabel("blk")
ax.set_ylabel("E")
#ax.set_ylim(-0.9,-0.8)
#ax.set_yscale('log')
ax.legend()
fig.subplots_adjust(left=0.15,bottom=0.15,right=0.98,top=0.97)
fig.savefig(f'R{R}_Nangle={Nangle}_Nr={Nr}.png',dpi=250)

