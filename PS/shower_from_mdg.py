import math as m
import random as r

from vector import Vec4
from particle import Particle, CheckEvent
from qcd import AlphaS, NC, TR, CA, CF

class Kernel:

    def __init__(self,flavs):
        self.flavs = flavs

class Pqq (Kernel):

    def Value(self,z,y):
        return CF*(2./(1.-z*(1.-y))-(1.+z))

    def Estimate(self,z):
        return CF*2./(1.-z)

    def Integral(self,zm,zp):
        return CF*2.*m.log((1.-zm)/(1.-zp))

    def GenerateZ(self,zm,zp):
        return 1.+(zp-1.)*m.pow((1.-zm)/(1.-zp),r.random())

    def Which(self):
        return "Pqq"

class Pgg (Kernel):

    def Value(self,z,y):
        return CA/2.*(2./(1.-z*(1.-y))-2.+z*(1.-z))

    def Estimate(self,z):
        return CA/(1.-z)

    def Integral(self,zm,zp):
        return CA*m.log((1.-zm)/(1.-zp))

    def GenerateZ(self,zm,zp):
        return 1.+(zp-1.)*m.pow((1.-zm)/(1.-zp),r.random())
    
    def Which(self):
        return "Pgg"

class Pgq (Kernel):

    def Value(self,z,y):
        return TR/2.*(1.-2.*z*(1.-z))

    def Estimate(self,z):
        return TR/2.

    def Integral(self,zm,zp):
        return TR/2.*(zp-zm)

    def GenerateZ(self,zm,zp):
        return zm+(zp-zm)*r.random()
    
    def Which(self):
        return "Pgq"

class Shower:

    def __init__(self,alpha,t0):
        self.t0 = t0
        self.alpha = alpha
        self.alphamax = alpha(self.t0)
        self.kernels = [ Pqq([fl,fl,21]) for fl in [-5,-4,-3,-2,-1,1,2,3,4,5] ]
        self.kernels += [ Pgq([21,fl,-fl]) for fl in [1,2,3,4,5] ]
        self.kernels += [ Pgg([21,21,21]) ]

    def MakeKinematics(self,z,y,phi,pijt,pkt):
        Q = pijt+pkt
        rkt = m.sqrt(Q.M2()*y*z*(1.-z))
        kt1 = pijt.Cross(pkt)
        if kt1.P() < 1.e-6: kt1 = pijt.Cross(Vec4(1.,0.,0.,0.))
        kt1 *= rkt*m.cos(phi)/kt1.P()
        kt2cms = Q.Boost(pijt).Cross(kt1)
        kt2cms *= rkt*m.sin(phi)/kt2cms.P()
        kt2 = Q.BoostBack(kt2cms)
        pi = z*pijt + (1.-z)*y*pkt + kt1 + kt2
        pj = (1.-z)*pijt + z*y*pkt - kt1 - kt2
        pk = (1.-y)*pkt
        return [pi,pj,pk]

    def MakeColors(self,flavs,colij,colk):
        self.c += 1
        if flavs[0] != 21:
            if flavs[0] > 0:
                return [ [self.c,0], [colij[0],self.c] ]
            else:
                return [ [0,self.c], [self.c,colij[1]] ]
        else:
            if flavs[1] == 21:
                if colij[0] == colk[1]:
                    if colij[1] == colk[0] and r.random()>0.5:
                        return [ [colij[0],self.c], [self.c,colij[1]] ]
                    return [ [self.c,colij[1]], [colij[0],self.c] ]
                else:
                    return [ [colij[0],self.c], [self.c,colij[1]] ]
            else:
                if flavs[1] > 0:
                    return [ [colij[0],0], [0,colij[1]] ]
                else:
                    return [ [0,colij[1]], [colij[0],0] ]

    def GeneratePoint(self,event):
        while self.t > self.t0:
            t = self.t0
            for split in event[2:]:
                for spect in event[2:]:
                    if spect == split: continue
                    if not split.ColorConnected(spect): continue
                    for sf in self.kernels:
                        if sf.flavs[0] != split.pid: continue
                        m2 = (split.mom+spect.mom).M2()
                        if m2 < 4.*self.t0: continue
                        zp = .5*(1.+m.sqrt(1.-4.*self.t0/m2))
                        g = self.alphamax/(2.*m.pi)*sf.Integral(1.-zp,zp)
                        tt = self.t*m.pow(r.random(),1./g)
                        if tt > t:
                            t = tt
                            s = [ split, spect, sf, m2, zp ]
            self.t = t
            if t > self.t0:
                z = s[2].GenerateZ(1.-s[4],s[4])
                y = t/s[3]/z/(1.-z)
                if y < 1.:
                    f = (1.-y)*self.alpha(t)*s[2].Value(z,y)
                    g = self.alphamax*s[2].Estimate(z)
                    if f/g > r.random():
                        phi = 2.*m.pi*r.random()
                        moms = self.MakeKinematics(z,y,phi,s[0].mom,s[1].mom)
                        cols = self.MakeColors(s[2].flavs,s[0].col,s[1].col)
                        event.append(Particle(s[2].flavs[2],moms[1],cols[1],True))
                        s[0].Set(s[2].flavs[1],moms[0],cols[0],True)
                        s[1].mom = moms[2]
                        return
    
    def Run(self,event,t):
        self.c = 1
        self.t = t
        while self.t > self.t0:
            self.GeneratePoint(event)
            
######################################################
#########################Run the shower###############

import sys
import pyhepmc as hp
import numpy as np

############################################Files################################
#The input_file has to be in .lhe, the extraction of the original .lhe.gz is necessary.
#It is also preferable to remove the hedder from the file so that only data for events are left.
input_file = "/home/guiwi/MG5_aMC_v3_5_1/eejjj/Events/run_06/unweighted_events.lhe"
output_file = "shower.txt"
#Number of events in the data set
dataset = 100000

alphas = AlphaS(91.1876,0.118)
shower = Shower(alphas,t0=1.)
r.seed(123456)
md_jets_4mom = np.zeros((dataset,4,4))
em_ep = np.zeros((dataset,2,4))
j_flav = np.zeros((dataset,4))
j_color = np.zeros((dataset,4,2))


file = open(input_file,'r')
lines = file.readlines()
file.close()

valid_line = ["       21","        1","       -1","        2","       -2"
    ,"        3","       -3","        4","       -4","        5","       -5"]
ev_num = -1
j_num = 0
first_lin = False
for line in lines:
    print(ev_num)
    if line[:7] == "<event>":
        j_num = 0
        ev_num += 1
        em_ep[ev_num,0,:] = [0,0,500,500]
        em_ep[ev_num,1,:] = [0,0,500,500]
    elif line[:9] in valid_line:
        j_flav[ev_num,j_num] = int(line[8:9])
        md_jets_4mom[ev_num,j_num,0] = float(line[33:50])
        md_jets_4mom[ev_num,j_num,1] = float(line[51:68])
        md_jets_4mom[ev_num,j_num,2] = float(line[69:86])
        md_jets_4mom[ev_num,j_num,3] = float(line[87:103])

        if int(line[24:27]) != 0: j_color[ev_num,j_num,0] = int(line[24:27])-500
        else : j_color[ev_num,j_num,0] = 0
        if int(line[29:32]) != 0: j_color[ev_num,j_num,1] = int(line[29:32])-500
        else : j_color[ev_num,j_num,1] = 0
        j_num += 1

md_jets_4mom = md_jets_4mom[:ev_num,:,:]
j_flav = j_flav[:ev_num,:]


#Launch the shower on each event and save the results 
#(a list of the hard partons and the partons from the final state of the shower)
#in the output file
f = open(output_file,"w")
for i in range(len(md_jets_4mom[:,0,0])):
    if md_jets_4mom[i,2,:].all() == 0:
        cur_event = [Particle(11,Vec4(em_ep[i,0,0],em_ep[i,0,1],em_ep[i,0,2],em_ep[i,0,3]),final=False),
            Particle(-11,Vec4(em_ep[i,1,0],em_ep[i,1,1],em_ep[i,1,2],em_ep[i,1,3]),final=False),
            Particle(int(j_flav[i,0]),Vec4(md_jets_4mom[i,0,0],md_jets_4mom[i,0,1],md_jets_4mom[i,0,2],md_jets_4mom[i,0,3]),j_color[i,0,:],final=True),
            Particle(int(j_flav[i,1]),Vec4(md_jets_4mom[i,1,0],md_jets_4mom[i,1,1],md_jets_4mom[i,1,2],md_jets_4mom[i,1,3]),j_color[i,1,:],final=True)]
    elif md_jets_4mom[i,3,:].all() == 0:
        cur_event = [Particle(11,Vec4(em_ep[i,0,0],em_ep[i,0,1],em_ep[i,0,2],em_ep[i,0,3]),final=False),
            Particle(-11,Vec4(em_ep[i,1,0],em_ep[i,1,1],em_ep[i,1,2],em_ep[i,1,3]),final=False),
            Particle(int(j_flav[i,0]),Vec4(md_jets_4mom[i,0,0],md_jets_4mom[i,0,1],md_jets_4mom[i,0,2],md_jets_4mom[i,0,3]),j_color[i,0,:],final=True),
            Particle(int(j_flav[i,1]),Vec4(md_jets_4mom[i,1,0],md_jets_4mom[i,1,1],md_jets_4mom[i,1,2],md_jets_4mom[i,1,3]),j_color[i,1,:],final=True),
            Particle(int(j_flav[i,2]),Vec4(md_jets_4mom[i,2,0],md_jets_4mom[i,2,1],md_jets_4mom[i,2,2],md_jets_4mom[i,2,3]),j_color[i,2,:],final=True)]
    else :
        cur_event = [Particle(11,Vec4(em_ep[i,0,0],em_ep[i,0,1],em_ep[i,0,2],em_ep[i,0,3]),final=False),
            Particle(-11,Vec4(em_ep[i,1,0],em_ep[i,1,1],em_ep[i,1,2],em_ep[i,1,3]),final=False),
            Particle(int(j_flav[i,0]),Vec4(md_jets_4mom[i,0,0],md_jets_4mom[i,0,1],md_jets_4mom[i,0,2],md_jets_4mom[i,0,3]),j_color[i,0,:],final=True),
            Particle(int(j_flav[i,1]),Vec4(md_jets_4mom[i,1,0],md_jets_4mom[i,1,1],md_jets_4mom[i,1,2],md_jets_4mom[i,1,3]),j_color[i,1,:],final=True),
            Particle(int(j_flav[i,2]),Vec4(md_jets_4mom[i,2,0],md_jets_4mom[i,2,1],md_jets_4mom[i,2,2],md_jets_4mom[i,2,3]),j_color[i,2,:],final=True),
            Particle(int(j_flav[i,3]),Vec4(md_jets_4mom[i,3,0],md_jets_4mom[i,3,1],md_jets_4mom[i,3,2],md_jets_4mom[i,3,3]),j_color[i,3,:],final=True)]
    
    t=250000
    
    
    f.write("<Event>\n")
    f.write(str(i))
    f.write("\n")
    f.write("<hard>\n")
    for p in cur_event:
        if p.col[0] != 0 or p.col[1] != 0:
            f.write("<hpid>")
            f.write(str(p.pid))
            f.write("<\\hpid>")
            f.write("<hmom>")
            f.write(str(p.mom))
            f.write("<\\hmom>")
            f.write(str(p.col))
            f.write("\n")
    f.write("<\\hard>\n")
    shower.Run(cur_event,t)
    sys.stdout.write('\rEvent {0}'.format(i))
    sys.stdout.flush()
    f.write("<soft>\n")
    for p in cur_event:
        f.write("<pid>")
        f.write(str(p.pid))
        f.write("<\\pid>")
        f.write("<mom>")
        f.write(str(p.mom))
        f.write("<\\mom>")
        if p.final: f.write(str(1))
        else: f.write(str(0))
        f.write("\n")
    f.write("<\\soft>\n")
    f.write("<\\Event>\n")



