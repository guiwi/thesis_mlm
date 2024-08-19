import numpy as np
import math
import pyhepmc as hp
from matplotlib import pyplot as plt
from particle import literals as lp
from IPython.display import display
import csv
import os
import gzip
import xml.etree.ElementTree as ET

#number of events
nEvents = 100000

#The run directory from the MadGraph event used for lhe.gz and hepmc files
input_directory = ".../MG5_aMC_v3_5_1/.../Events/run_01/"
#The output csv file
output_file = "weights_fit1.csv"

##Extract pt of tt and weiths from PY results
with hp.open(input_directory+"tag_1_pythia8_events.hepmc") as f:
    number = 0
    tot_t_pt = np.zeros(nEvents)
    t_pt = np.zeros(nEvents)
    weights = np.zeros((nEvents,3))
    for event in f:
        weights[number][0] = event.weights[1]
        weights[number][1] = event.weights[2]
        weights[number][2] = event.weights[3]
        tempfour = np.zeros(4)
        for p in event.particles:
            if p.pid == 6 and p.status == 22:
                tempfour += p.momentum
                t_pt[number] = hp.FourVector.pt(p.momentum)
            elif p.pid == -6 and p.status == 22:
                tempfour += p.momentum
        tot_t_pt[number] = hp.FourVector.pt(tempfour)
        number += 1
        print(number)

##Extract pt of tt from Madgraph results
with hp.open(input_directory+"unweighted_events.lhe.gz") as f:
    number = 0
    hard_tot_t_pt = np.zeros(nEvents)
    hard_t_pt = np.zeros(nEvents)
    hard_anti_t_pt = np.zeros(nEvents)
    j_pt = np.zeros(nEvents)
    g_pseudo = np.zeros(nEvents)
    g_energy = np.zeros(nEvents)
    g_rap = np.zeros(nEvents)

    hard_t_4mom = np.zeros((nEvents,4))
    anti_hard_t_4mom =np.zeros((nEvents,4))

    tt_pz = np.zeros(nEvents)

    j_invm = np.zeros(nEvents)
    j_phi = np.zeros(nEvents)

    momentum0 = np.zeros(nEvents)
    momentum1 = np.zeros(nEvents)
    momentum2 = np.zeros(nEvents)
    momentum3 = np.zeros(nEvents)
    for event in f:
        tempfour = np.zeros(4)
        for p in event.particles:
            if p.pid == 6:
                tempfour += p.momentum
                hard_t_pt[number] = hp.FourVector.pt(p.momentum)
                hard_t_4mom[number,:] = p.momentum
            elif p.pid == -6:
                tempfour += p.momentum
                hard_anti_t_pt[number] = hp.FourVector.pt(p.momentum)
                anti_hard_t_4mom[number,:] = p.momentum
            elif p.id == 5 and p.status == 1 and len(p.parents) == 2:
                if (p.parents[0].pid == 21 or p.parents[1].pid == 21 or p.parents[0].pid == 9 or p.parents[1].pid == 9):
                    j_pt[number] = hp.FourVector.pt(p.momentum)
                    g_pseudo[number] = hp.FourVector.eta(p.momentum)
                    g_energy[number] = p.momentum.e
                    g_rap[number] = hp.FourVector.rap(p.momentum)
                    momentum0[number] = p.momentum.px
                    momentum1[number] = p.momentum.py
                    momentum2[number] = p.momentum.pz
                    momentum3[number] = p.momentum.e
                    j_invm[number] = hp.FourVector.m(p.momentum)
                    j_phi[number] = hp.FourVector.phi(p.momentum)
        hard_tot_t_pt[number] = hp.FourVector.pt(tempfour)
        tt_pz[number] = tempfour[2]
        number += 1
        print(number)

##Find which event from PY came from which event from Mad
tempwei0 = weights[:,0]
tempwei1 = weights[:,1]
tempwei2 = weights[:,2]

input = gzip.open(input_directory+"unweighted_events.lhe.gz", 'r')
tree = ET.parse(input)
root = tree.getroot()

scales = np.zeros((nEvents,3))
iter = 0
for event in root:
    if iter != 0:
        for thing in event:
            if thing.tag == 'clustering':
                sc = 0
                for scale in thing:
                    print(scale.attrib)
                    scales[iter-2,sc]=scale.get("scale")
                    sc += 1
    iter +=1

scales0 = scales[:,0]
scales1 = scales[:,1]
scales2 = scales[:,2]



###################################################################
###################################################################
#Pythia will "break" some events, these event will thus be present in 
#the lhe but not in the hepmc file.
#We look for differences in the pt ot t quark to find those events
#We delete them. Sometimes, this selection will fail
#For each deleted event, the difference in t_pt is printed
#If the tolerated difference in the if statement (1 by default) is
#too small or too big, this will lead to the deletion on many events
#So, it is easy to catch when a problem occurs
k = 0
while k < len(hard_t_pt):
    if np.abs(hard_t_pt[k]-t_pt[k]) >= 1:
        print("Element deleted with diff of ")
        print(hard_t_pt[k]-t_pt[k])
        print("new diff :")
        hard_t_pt = np.delete(hard_t_pt,k)
        hard_anti_t_pt = np.delete(hard_anti_t_pt,k)

        momentum0 = np.delete(momentum0,k)
        momentum1 = np.delete(momentum1,k)
        momentum2 = np.delete(momentum2,k)
        momentum3 = np.delete(momentum3,k)

        hard_t_4mom = np.delete(hard_t_4mom,k,axis=0)
        anti_hard_t_4mom = np.delete(anti_hard_t_4mom,k,axis=0)

        scales0 = np.delete(scales0,k)
        scales1 = np.delete(scales1,k)
        scales2 = np.delete(scales2,k)

        tt_pz = np.delete(tt_pz,k)

        tot_t_pt = np.delete(tot_t_pt,k)
        j_pt = np.delete(j_pt,k)
        g_pseudo = np.delete(g_pseudo,k)
        g_energy = np.delete(g_energy,k)
        hard_tot_t_pt = np.delete(hard_tot_t_pt,k)

        j_invm = np.delete(j_invm,k)
        j_phi = np.delete(j_phi,k)

        g_rap = np.delete(g_rap,k)
        print(hard_t_pt[k]-t_pt[k])
        k -= 1
    k +=1

##Find which event were accepted and which were rejected from associated weights
accpeted = np.zeros((3,len(tempwei0)))
for k in range(len(tempwei0)):
    if tempwei0[k] != 0: accpeted[0,k] = 1
    if tempwei1[k] != 0: accpeted[1,k] = 1
    if tempwei2[k] != 0: accpeted[2,k] = 1

##Save the csv
#acc60,90 and 120 are the acceptance/veto from MLM at different Qcut
#sc0,sc1,sc2 are internal scales saved by MadGraph
#mom0,1,2,3 is the momenutm of the hard jet (or 0 if ther is non),
#j_pt is the pt computed from mom0 and mom1
with open(output_file, mode='w') as data_file:
    data_writer = csv.writer(data_file, delimiter=',')
    data_writer.writerow(['j_pt', 'mom0','mom1','mom2','mom3', 'acc60','acc90','acc120',
        'sc0','sc1','sc2','t_mom0','t_mom1','t_mom2','t_mom3','antit_mom0','antit_mom1','antit_mom2','antit_mom3'])
    for i in range(len(j_pt)-1):
        data_writer.writerow([j_pt[i],momentum0[i],momentum1[i],momentum2[i],momentum3[i],
            accpeted[0,i],accpeted[1,i],accpeted[2,i],scales0[i],scales1[i],
                scales2[i],hard_t_4mom[i,0],hard_t_4mom[i,1],hard_t_4mom[i,2],hard_t_4mom[i,3]
                    ,anti_hard_t_4mom[i,0],anti_hard_t_4mom[i,1],anti_hard_t_4mom[i,2],anti_hard_t_4mom[i,3]])
