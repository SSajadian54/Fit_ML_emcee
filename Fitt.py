import numpy as np
import emcee
import sys
from numpy import matrix
import matplotlib.pyplot as plt
from IPython.display import display, Math
####================ constants ============================
year=float(365.2425);##days 
ndim=5
Nwalkers=10
nstep=100

#============== functions ====================================
def prior(p): 
    u0=p[0]; t0=p[1]; tE=p[2];  mbase=p[3]; fb=p[4]
    if(u0>0.0 and u0<1.5 and t0>0.0 and t0<float(10.0*year) and tE>0.0 and tE<float(5.0*ptr[2]) and mbase>0.0 and mbase<float(2.0*ptr[3]) and fb>0.0 and fb<1.0): 
        return(0); 
    return(-1);     
#==========================================================
def  magnification(timee, u01, t01, tE1):  
    u2= ((timee-t01)/tE1)**2.0+ u01**2.0 
    As= (u2+2.0)/np.sqrt(u2*(u2+4.0))
    return(As)
#==========================================================
def lnlike2(p,tim,Magn,errm):
    U0=p[0]; T0=p[1];  TE=p[2];  mbase=p[3];  fb=p[4]
    chi2=0.0 
    for i in range(len(tim)): 
        As=magnification(tim[i],  U0, T0, TE )
        Mag= mbase-2.5*np.log10(As*fb + 1.0-fb )
        chi2+=float((Mag-Magn[i])**2.0/errm[i]**2.0 )
    return float(-0.5*chi2)
#============================================================
def chi2(p,tim,Magn,errm):
    lp=prior(p)
    if(lp<0.0):
        return(-np.inf)
    return lnlike2(p, tim, Magn , errm)
#================= Main program ==============================
nam=[r"$u_{0, 1}$",r"$,~u_{0, 2}$",r"$,~\Delta t_{0}$",r"$,~\Delta mag$", r"$,~\Delta \chi^{2}$"]
best=np.zeros((ndim*3))
fit= np.zeros((ndim))
labels = ["u0", "t0", "tE",  "mbase",  "fb"]
save2=open("./results.txt","w")
save2.close()

nf=444241
par=np.zeros(( nf , 32 ))
par=np.loadtxt("./files/distribution/Q_0.dat")   

for i in range(nf):
    Nsim,strucl,Ml,RE, Dl, vl, tetE, ksi = int(par[i,0]), par[i,1], par[i,2], par[i,3], par[i,4], par[i,5], par[i,6], par[i,7]  
    strucs, cl, Ds, vs, Av, AI, Map1, Map2=par[i,8], par[i,9],par[i,10], par[i,11],par[i,12],par[i,13], par[i,14], par[i,15]
    semi, Vt, tE, u0, u02, t0 , t02, opt  =par[i,16],par[i,17],par[i,18],par[i,19],par[i,20],par[i,21], par[i,22], par[i,23]
    chi_real, chi_star1, chi_star2, det, peak, Dmag =  par[i,24], par[i,25], par[i,26], par[i,27], par[i, 28], par[i,29]
    nli, ndat= int(par[i, 30]), int(par[i,31])
    Dt0=t02-t0
    print("****************************************************")
    print("Counter:  ",  Nsim)
    print("t0,  t02,  u0,   u02:   ",  t0,  t02,  u0, u02 )
    print("semi, Dl, Ds:   ", semi,    Dl,   Ds)
    print(" det,   peak, dmag:    ", det, peak , Dmag)
    print("Nli, ndat:  ", nli, ndat)
    print("****************************************************")
    
    mod=np.zeros((nli,5))
    mod=np.loadtxt('./files/light/l_{0:d}.dat'.format(Nsim)) 
    dat=np.zeros((ndat,6))
    dat=np.loadtxt('./files/light/d_{0:d}.dat'.format(Nsim)) 
    ## u0, t0, tE, mbase, fb
    ptr=np.array([np.min(np.array(u0, u02)), np.mean(np.array(t0, t02)) , tE , np.min(np.array(Map1, Map2)) , 1.0 ])
    Epci=np.array([0.5, 100.0, 50.0,  2.5,  0.5])
    p0=np.zeros((Nwalkers,ndim))
    p0[:,:]=[ptr+Epci*np.random.randn(ndim) for i in range(Nwalkers)]


    tim=np.zeros((ndat)); 
    Magn=np.zeros((ndat));   
    errm=np.zeros((ndat));
    tim=dat[:,0]*year;    Magn=dat[:,2];     errm=dat[:,3];  
    #print("Lightcurve: Magnitude, time, error:    ",  Magn, tim, errm )
    #===============================================================================
    
    sampler=emcee.EnsembleSampler(Nwalkers,ndim, chi2, args=(tim,Magn,errm), threads=8)
    sampler.run_mcmc(p0, 5000, progress=True)
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    #print(flat_samples.shape)
    for k in range(ndim):
        mcmc = np.percentile(flat_samples[:, k], [16, 50, 84])
        q = np.diff(mcmc)
        fit[k]=mcmc[1]
        best[k*3+0], best[k*3+1], best[k*3+2]= mcmc[1],   q[0], q[1]
    save2=open("./results.txt","a+")
    np.savetxt(save2,best.reshape(-1,ndim*3),fmt='%.5f  %.4f  %.4f  %.5f  %.4f  %.4f  %.5f  %.4f  %.4f  %.5f  %.4f  %.4f  %.5f  %.4f  %.4f')
    save2.close()
    chi_fit=float(-2.0*chi2(fit,tim, Magn, errm))
    print("Chi_real, chi_star1, chi_star2, chi_fitted:  ", chi_real, chi_star1, chi_star2, chi_fit)
    print("Dchis:  ",  abs(chi_real-chi_star1),   abs(chi_real-chi_star2),  abs(chi_real-chi_fit)  )
    print ("Best fitted:  ", fit)
    dchi=np.min(np.array((abs(chi_real-chi_star1),   abs(chi_real-chi_star2),  abs(chi_real-chi_fit))))
    
#################################################################################


    nms=2000; 
    model=np.zeros((nms, 2))
    dt= float(6.0*fit[2]/nms)
    for s in range(nms): 
        model[s,0]=float(-3.0*fit[2] +s*dt+fit[1])
        model[s,1]=float(fit[3]-2.5*np.log10(fit[4]* magnification(model[s,0],fit[0],fit[1],fit[2]) +1.0-fit[4] ) ) 


    #emax, emin= np.max(mod[:,2]), np.min(mod[:,2])
    plt.clf()
    fig=plt.figure(figsize=(8,6))
    ax1=fig.add_subplot(111)
    plt.plot(mod[:,1]*tE+t0,mod[:,2],'g-',label="Binary stars", lw=1.5, alpha=0.95)
    plt.plot(mod[:,1]*tE+t0,mod[:,3],'r--',label="Star 1", lw=1.5, alpha=0.95)
    plt.plot(mod[:,1]*tE+t0,mod[:,4],'b:',label="Star 2", lw=1.5, alpha=0.95)
    plt.plot(model[:,0],model[:,1],'m:',label="best-fitted", lw=1.5, alpha=0.95)
    plt.errorbar(dat[:,1]*tE+t0,dat[:,2],yerr=dat[:,3], fmt=".", markersize=10.8,  color='m', ecolor='gray', elinewidth=0.3, capsize=0)
    plt.ylabel(r"$r_{\rm{LSST}}-\rm{magnitude}$",fontsize=18)
    plt.xlabel(r"$\rm{time}(\rm{days})-t_{0, 1}$",fontsize=18)
    plt.title(str(nam[0])+'={0:.2f}'.format(u0)+ str(nam[1])+'={0:.1f}'.format(u02)+str(nam[2])+ '={0:.1f}'.format(Dt0) + str(nam[3])+ '={0:.1f}'.format(Dmag) +str(nam[4])+ '={0:.1f}'.format(dchi) , fontsize=16,color='b')

    #plt.title('BEST:  u0={0:.2f}'.format(best[0])+ ',  t0={0:.1f}'.format(best[1])+ ',  tE={0:.1f}'.format(best[2]) +  ',  m_base={0:.1f}'.format(best[3]) + ',  f_b={0:.1f}'.format(best[4]) , fontsize=14,color='b')

    #plt.text(np.min(mod[:,1]*tE+t0)+0.5,emax-(emax-emin)/2.0,str(stat), color=col , fontsize=13)
    #pylab.ylim([emin-0.02*(emax-emin),emax+0.02*(emax-emin)])
    #pylab.xlim([np.min(mod[:,1]*tE),  np.max(mod[:,1]*tE)  ])
    plt.gca().invert_yaxis()
    ax1.legend(prop={"size":12.5})
    fig=plt.gcf()
    fig.savefig("./files/light/lights/FLC_{0:d}.jpg".format(Nsim),dpi=200)
    print ("Light curve is plotted:  ", Nsim)
    #input("Enter a number ")


