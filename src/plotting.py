import numpy as np
import itertools
import matplotlib.pyplot as plt

def plot_cityscape(rho):
    # plot cityscape
    dim = rho.shape[0]
    n_qubits = int(np.sqrt(dim))
    fig1 = plt.figure(44)
    ax1 = fig1.add_subplot(121, projection='3d')
    
    ticks = []
    ticks.append([i for i in range(dim)])
    ticks = np.array(ticks).flatten()
    
    labels = ['|' + ''.join(p) + '>' for p in itertools.product('01', repeat=n_qubits)]
    ticks = np.arange(dim)

    _x = np.arange(dim)
    _y = np.arange(dim)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    top_real = np.real(rho)
    bottom_real = np.zeros_like(top_real)
    width = depth = 1

    # Real part (3D)
    ax1.bar3d(x, -y, np.reshape(bottom_real,-1), width, depth, np.reshape(top_real,-1), shade=True)
    ax1.set_title('density matrix (real)')
    ax1.set_xticks(ticks,labels=labels)
    ax1.set_yticks(-1*ticks,labels=labels)
    ax1.set_zlim(-.5,.5)

    fig3 = plt.figure(46)
    ax3 = fig3.add_subplot(121)

    # Real part (2D)
    c1 = ax3.pcolor(_xx,-_yy,top_real,cmap='RdBu',vmin=-.5,vmax=.5)
    ax3.set_title('density matrix (real)')
    ax3.set_xticks(ticks,labels=labels)
    ax3.set_yticks(-1*ticks,labels=labels)
    fig3.colorbar(c1,ax=ax3)

    fig2 = plt.figure(45)
    ax2 = fig2.add_subplot(121, projection='3d')

    top_imag = np.imag(rho)
    bottom_imag = np.zeros_like(top_imag)
    width = depth = 1

    # Imaginary part (3D)
    ax2.bar3d(x, -y, np.reshape(bottom_imag,-1), width, depth, np.reshape(top_imag,-1), shade=True)
    ax2.set_title('density matrix (imag)')
    ax2.set_xticks(ticks,labels=labels)
    ax2.set_yticks(-1*ticks,labels=labels)
    ax2.set_zlim(-.5,.5)

    fig4 = plt.figure(47)
    ax4 = fig4.add_subplot(121)
    
    # Imaginary part (2D)
    c2 = ax4.pcolor(_xx,-_yy,top_imag,cmap='RdBu',vmin=-.5,vmax=.5)
    ax4.set_title('density matrix (imag)')
    ax4.set_xticks(ticks,labels=labels)
    ax4.set_yticks(-1*ticks,labels=labels)
    fig4.colorbar(c2,ax=ax4)

    plt.show()

                
                    


        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    