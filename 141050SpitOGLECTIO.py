### In the name of GOD #####  YAALIMADAD
import numpy as np
import emcee
import sys
sys.path.append("/home/sajadian/MCMC_2803/VBB2018")
import VBBinaryLensingLibrary as vb
from numpy import matrix



####========================== constants =======================================
NDIM=18
Nwalkers=150
Nstep=1000000
NUM1=380;#OGLE
NUM2=0;####102#Danish
NUM3=91#CTIO
NUM4=31##Spitzer
numg=int(NUM1+NUM2+NUM3)
numt=int(NUM1+NUM2+NUM3+NUM4)
RA=np.pi/180.0
Limbg=3.0*0.43/(2.0+0.43)
Limbs=3.0*0.16/(2.0+0.16)
Fso=np.power(10.0,-9.0);  Fbgo=np.power(10.0,-8.0); 
#Fsd=np.power(10.0,-5.0);  Fbgd=np.power(10.0,-4.0);
Fsc=np.power(10.0,-8.0);  Fbgc=np.power(10.0,-7.0);
Fss=np.power(10.0,-9.0);  Fbgs=np.power(10.0,-8.0);
Rho=0.0001;  Orbit=0.001

#  d, q, u0, alpha, rho*, tE, t0, piN, piE, omega1, omega2, omega3, Fso, Fbgo, FsD, FbgD, FsC, FbgC, FsS, FbgS


###   28/03/97
#prt=np.array([1.113,0.389,0.352,float(1.2*RA),6.05,73.3,6836.1317,-0.046,0.111,float(-0.1/0.36525),float(0.57/0.36525),1.0e-5,9.45396,6.62646,1.07112,1.18997,2.78458189617,2.3330540505])


prt=np.array([1.12422801,3.80860460e-01,3.47843541e-01,6.49487200e-02,6.06458017e+00,6.82329262e+01, 6.83526422e+03 , -4.21540855e-02,1.20934744e-01,-8.96606847e-01,2.58587611e+00,6.11913368e-01,9.60895686,6.60312567e+00,  1.05445871e+00 ,  1.16782300e+00,2.95696774e+00,2.30438275e+00])



Epci=np.array([0.05,0.05,0.03,float(2.5*RA),0.5,3.5,7.0,0.08,0.08,2.5,2.5,2.5,3.5,2.5,0.6,0.6,1.3,1.3])

a=np.zeros((NDIM,3)); b=np.zeros(NDIM); ac=np.zeros(1); tet= np.zeros(NDIM); chi=np.zeros(2);
magm1=np.zeros(numt); pr=vb.doubleArray(NDIM)
timed=np.zeros(numt); magnio=np.zeros(numt);   errom=np.zeros(numt)



####========================== functions =======================================
def prior(p): 
    tet=int(0)
    for i in range(NDIM):
        if(abs(float(p[i])-float(prt[i]))<=float(4.5*float(Epci[i]))):
            tet=tet+int(1)
    if(float(p[0])<0.0 or float(p[0])>=2.5 or float(p[1])<0.0 or float(p[1])>0.9 or float(abs(p[2]))>0.8 or 
    float(p[4]*Rho)>=0.05 or float(p[4])<0.0 or float(p[5])<=0.0 or float(p[5])>150.0 or float(p[6])<0.0 or float(p[12])<0.0 or 
    float(p[13])<0.0 or float(p[14])<0.0 or float(p[15])<0.0 or float(p[16])<0.0 or float(p[17])<0.0 or 
    float(p[12])==0.0 or float(p[14])==0.0 or float(p[16])==0.0 ):
        tet=int(0) 
    if(int(tet)==NDIM and np.isfinite(p).all()):
        return 0.0
    return -1.0
#===============================================================================
def lnlike2(p,timed,magnio,errom):
    VBB=vb.VBBinaryLensing()
    VBB.Tol=1.0e-3; ### accuracy
    VBB.parallaxsystem=1;### North-East parallax
    VBB.SetObjectCoordinates('./files/OB141050coords.txt',1)
    for k in range(NDIM): 
        pr[k]=float(p[k])
    pr[4]=pr[4]*Rho; 
    pr[9]=pr[9]*Orbit;   pr[10]=pr[10]*Orbit;    pr[11]=pr[11]*Orbit;
    fso=float(pr[12]*Fso);    fbgo=float(pr[13]*Fbgo);  
   # fsd=float(pr[14]*Fsd);    fbgd=float(pr[15]*Fbgd); 
    fsc=float(pr[14]*Fsc);    fbgc=float(pr[15]*Fbgc); 
    fss=float(pr[16]*Fss);    fbgs=float(pr[17]*Fbgs); 

    pr[0]=float(np.log(abs(pr[0]))); pr[1]=float(np.log(abs(pr[1])))
    pr[4]=float(np.log(abs(pr[4]))); pr[5]=float(np.log(abs(pr[5])))


    VBB.satellite=0
    for k in range(numg):  
        ma=float(VBB.BinaryLightCurveOrbital(pr,float(timed[k]),float(Limbg)))
        if(k<NUM1):  ### OGLE
            magm1[k]=float(-2.5*np.log10(abs(fso*ma+fbgo)))
        #elif(k>=NUM1 and k<(NUM1+NUM2)): 
        #    magm1[k]=float(-2.5*np.log10(abs(fsd*ma+fbgd)))
        elif(k>=NUM1 and k<numg): #### CTIO
            magm1[k]=float(-2.5*np.log10(abs(fsc*ma+fbgc)))
    VBB.satellite=1
    for k in range(NUM4):  
        ma=float(VBB.BinaryLightCurveOrbital(pr,float(timed[k+numg]),float(Limbs)))
        magm1[k+numg]=float(-2.5*np.log10(abs(fss*ma+fbgs)))
    del VBB  
    chii=float(np.sum((magm1-magnio)**2.0/(errom*errom)))
    #print (chii)
    #print "**************************************************" 
    return float(-0.5*chii)
#===============================================================================
def chi2(p,timed,magnio,errom):
    lp=prior(p)
    if(lp<0.0):
        return(-np.inf)
    return lnlike2(p,timed,magnio,errom)
#===============================================================================       



###============================= Main program ===================================
array1=np.zeros((NUM1,3));# array2=np.zeros((NUM2,3)); 
array3=np.zeros((NUM3,3)); array4=np.zeros((NUM4,3))
sda=np.zeros((numt,3))
array1=np.loadtxt("./files/dataOGLE141050b.pysis") 
#array2=np.loadtxt("./files/dataDANISH141050b.pysis") 
array3=np.loadtxt("./files/dataCTIO141050b.pysis") 
array4=np.loadtxt("./files/dataSPITZER141050.pysis" ) 
sda=np.concatenate((array1,array3,array4),axis=0)
timed=sda[:,0];  magnio=sda[:,1];  errom=sda[:,2];  
print "******************MAGNIFICATION *********************"
print(magnio)
print "******************TIME *********************"
print(timed)
print "******************ERROR *********************"
print(errom)
print(len(array1))
#print(len(array2))
print(len(array3))
print(len(array4))
print(len(sda))



print "chi2: ", float(-2.0*chi2(prt,timed,magnio,errom))
input ("Enter a number ")



#===============================================================================
count=np.arange(Nwalkers)
p0=np.zeros((Nwalkers,NDIM))
p0[:,:]=[prt+Epci*np.random.randn(NDIM) for i in range(Nwalkers)]

print "distance: ", p0[:,0]
print "tE: ", p0[:,5]
print "\omega_3: ", p0[:,11]
print "*******************************************"




sampler=emcee.EnsembleSampler(Nwalkers,NDIM,chi2,args=(timed,magnio,errom),threads=8)
print("sampler is made !!!!!!")

fil=open("./files/finalOCS141050_0504.dat","w")
fil.close()
for pos, prob1, state in sampler.sample(p0,iterations=Nstep,storechain=False):
    fil = open("./files/finalOCS141050_0504.dat","a")
    for i in range(Nwalkers):   
        if(np.isfinite(prob1[i])): 
            ssa=np.concatenate((pos[i,:].reshape((1,NDIM)),prob1[i].reshape((1,1))),axis=1)
            np.savetxt(fil,ssa,fmt="%.8f %.8f %.8e %.8f %.10e %.8f %.8f %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8f")
    fil.close()
print("***************** END OF MCMC *********************** ")




##========================== Saving ============================================
array_last=np.zeros((Nwalkers*Nstep,NDIM+1))
array_last=np.loadtxt("files/finalOCS141050_0504.dat") 
samples1 = array_last[:,:NDIM].reshape((-1,NDIM))


print("******* The Best Fitted parameters  ********")
save2=open("files/bestfitted1050b_0504.txt","w")
a=map(lambda v:(v[1],v[2]-v[1],v[1]-v[0]),zip(*np.percentile(samples1, [16, 50, 84],axis=0)))
b=matrix(a).transpose()[0].getA()[0]
np.savetxt(save2,a,delimiter='   ',fmt='%.12f')
ac[0]=float(np.mean(sampler.acceptance_fraction))
chi[0]=float(-2.0*chi2(prt,timed,magnio,errom))
chi[1]=float(-2.0*chi2(  b,timed,magnio,errom))
np.savetxt(save2,ac,fmt='%.13f')
np.savetxt(save2,chi.reshape(1,2),fmt='%.13f   %.13lf')
save2.close()

for i in range(NDIM):
    print a[i][0], " + ", a[i][1], " - ", a[i][2]
print("Mean acceptance fraction:{0:.3f}".format(np.mean(sampler.acceptance_fraction)))
print("reduced chi2 of paper model: ",chi[0])
print("reduced chi2 of mcmc  model: ",chi[1])
print(">>>>>>>>>>>>>>>>>>>>>>>>>>> END OF PROGRAM <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
#===============================================================================
