import xml.etree.ElementTree as ET
import gzip
import numpy as np
import vector
import awkward as ak
import fastjet
import math
import matplotlib.pyplot as plt
import csv

def comp_R(mom4):
    phi = math.atan2(mom4["px"],mom4["py"])
    mag = math.sqrt(mom4["px"]**2+mom4["py"]**2+mom4["pz"]**2)
    nu = 0.5*math.log((mag+mom4["pz"])/(mag-mom4["pz"]))
    return math.sqrt(nu**2+phi**2)

def MatchJetsPartons(pseudo_jet_mom,parton_mom,r_clust):
    return (comp_R(pseudo_jet_mom) <= comp_R(parton_mom)+r_clust and 
        comp_R(pseudo_jet_mom) >= comp_R(parton_mom)-(r_clust*1.5))

#hepmc file from the event
input_hepmc = ".../MG5_aMC_v3_5_1/.../Events/run_01/tag_1_pythia8_events.hepmc"
#weights generated with the NN
input_pred_weight = "pred_fit.txt"
#csv used to make the NN predictions (used here to recover the original mlm veto)
input_csv = "weights_fit3.csv"

#Number of events
nEvents = 100000

#Re clustering radius
R_clust = 0.4



builder = ak.ArrayBuilder()
import pyhepmc as hp
tot_ev = 0
with hp.open(input_hepmc) as f:
    for event in f:
        tot_ev +=1
        builder.begin_list()
        for p in event.particles:
            if p.children == []:
                builder.append({"px":p.momentum[0],"py":p.momentum[1],"pz":p.momentum[2],"E":p.momentum[3]})
        print(tot_ev)
        builder.end_list()

print(tot_ev)



#Reclustering with fastjet
array = builder.snapshot()

jetdef = fastjet.JetDefinition(fastjet.kt_algorithm, R_clust)
cluster = fastjet.ClusterSequence(array, jetdef)

pseu_jets = cluster.inclusive_jets(20)


f = open(input_pred_weight,"r")
lines = f.readlines()
f.close()

ckkw_weights = np.zeros(nEvents)

i = 0
for line in lines:
    ckkw_weights[i] = float(line)
    i += 1

acceptedList = np.zeros(nEvents)

event_count = 0
with open(input_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if row[0] == 'j_pt':
                    print(row)
            else:
                event_count +=1
                acceptedList[event_count] = float(row[5])




from matplotlib import pyplot as plt



############Distributions in p_t#######################
#########################################################
fig, axs = plt.subplots(1,4, sharex=True,sharey=True)

n_pred = np.zeros((4,50))
n_pred_ps = np.zeros((4,50))


for ik in range(len(pseu_jets[:])):
    if len(pseu_jets[ik]) > 0:
        hardness = np.zeros(len(pseu_jets[ik]))
        for j in range(len(pseu_jets[ik])):
            hardness[j] = math.sqrt(pseu_jets[ik][j]["px"]**2+pseu_jets[ik][j]["py"]**2)
        hard_ind = np.argsort(hardness)

        if acceptedList[ik]==1 and hardness[hard_ind[-1]]/20 < 50:
            n_pred[0,math.floor(hardness[hard_ind[-1]]/20)] += 1
            if len(pseu_jets[ik]) > 1:
                n_pred[1,math.floor(hardness[hard_ind[-2]]/20)] += 1
                if len(pseu_jets[ik]) > 2:
                    n_pred[2,math.floor(hardness[hard_ind[-3]]/20)] += 1
                    if len(pseu_jets[ik]) > 3:
                        n_pred[3,math.floor(hardness[hard_ind[-4]]/20)] += 1

        if hardness[hard_ind[-1]]/20 < 50:
            n_pred_ps[0,math.floor(hardness[hard_ind[-1]]/20)] += ckkw_weights[ik]
            if len(pseu_jets[ik]) > 1:
                n_pred_ps[1,math.floor(hardness[hard_ind[-2]]/20)] += ckkw_weights[ik]
                if len(pseu_jets[ik]) > 2:
                    n_pred_ps[2,math.floor(hardness[hard_ind[-3]]/20)] += ckkw_weights[ik]
                    if len(pseu_jets[ik]) > 3:
                        n_pred_ps[3,math.floor(hardness[hard_ind[-4]]/20)] += ckkw_weights[ik]






temp_lin = np.linspace(0,1000,51)
axs[0].stairs(n_pred[0,:],temp_lin,color='r',linewidth=2.5)
axs[0].stairs(n_pred_ps[0,:],temp_lin,color='g',linewidth=2.5)

axs[0].set_title("1st jet")
axs[0].set_xlabel("p_t of jet")
axs[0].set_ylabel("# of jets")

axs[1].stairs(n_pred[1,:],temp_lin,color='r',linewidth=2.5)
axs[1].stairs(n_pred_ps[1,:],temp_lin,color='g',linewidth=2.5)

axs[1].set_title("2nd jet")
axs[1].set_xlabel("p_t of jet")

axs[2].stairs(n_pred[2,:],temp_lin,color='r',linewidth=2.5)
axs[2].stairs(n_pred_ps[2,:],temp_lin,color='g',linewidth=2.5)

axs[2].set_title("3rd jet")
axs[2].set_xlabel("p_t of jet")

axs[3].stairs(n_pred[3,:],temp_lin,color='r',linewidth=2.5)
axs[3].stairs(n_pred_ps[3,:],temp_lin,color='g',linewidth=2.5)

axs[3].set_title("4th jet")
axs[3].set_xlabel("p_t of jet")


plt.suptitle("Phythia_MLM (red) vs reweighted (green), clustered with fastjet_kt")
plt.show()


############Distributions in energy#######################
#########################################################


fig, axs = plt.subplots(1,4, sharex=True,sharey=True)

n_pred = np.zeros((4,50))
n_pred_ps = np.zeros((4,50))


for ik in range(len(pseu_jets[:])):
    if len(pseu_jets[ik]) > 0:
        hardness = np.zeros(len(pseu_jets[ik]))
        for j in range(len(pseu_jets[ik])):
            hardness[j] = pseu_jets[ik][j]["E"]
        hard_ind = np.argsort(hardness)

        if acceptedList[ik]==1 and hardness[hard_ind[-1]]/20 < 50:
            n_pred[0,math.floor(hardness[hard_ind[-1]]/20)] += 1
            if len(pseu_jets[ik]) > 1:
                n_pred[1,math.floor(hardness[hard_ind[-2]]/20)] += 1
                if len(pseu_jets[ik]) > 2:
                    n_pred[2,math.floor(hardness[hard_ind[-3]]/20)] += 1
                    if len(pseu_jets[ik]) > 3:
                        n_pred[3,math.floor(hardness[hard_ind[-4]]/20)] += 1

        if hardness[hard_ind[-1]]/20 < 50:
            n_pred_ps[0,math.floor(hardness[hard_ind[-1]]/20)] += ckkw_weights[ik]
            if len(pseu_jets[ik]) > 1:
                n_pred_ps[1,math.floor(hardness[hard_ind[-2]]/20)] += ckkw_weights[ik]
                if len(pseu_jets[ik]) > 2:
                    n_pred_ps[2,math.floor(hardness[hard_ind[-3]]/20)] += ckkw_weights[ik]
                    if len(pseu_jets[ik]) > 3:
                        n_pred_ps[3,math.floor(hardness[hard_ind[-4]]/20)] += ckkw_weights[ik]






temp_lin = np.linspace(0,1000,51)
axs[0].stairs(n_pred[0,:],temp_lin,color='r',linewidth=2.5)
axs[0].stairs(n_pred_ps[0,:],temp_lin,color='g',linewidth=2.5)

axs[0].set_title("1st jet")
axs[0].set_xlabel("p_t of jet")
axs[0].set_ylabel("# of jets")

axs[1].stairs(n_pred[1,:],temp_lin,color='r',linewidth=2.5)
axs[1].stairs(n_pred_ps[1,:],temp_lin,color='g',linewidth=2.5)

axs[1].set_title("2nd jet")
axs[1].set_xlabel("p_t of jet")

axs[2].stairs(n_pred[2,:],temp_lin,color='r',linewidth=2.5)
axs[2].stairs(n_pred_ps[2,:],temp_lin,color='g',linewidth=2.5)

axs[2].set_title("3rd jet")
axs[2].set_xlabel("p_t of jet")

axs[3].stairs(n_pred[3,:],temp_lin,color='r',linewidth=2.5)
axs[3].stairs(n_pred_ps[3,:],temp_lin,color='g',linewidth=2.5)

axs[3].set_title("4th jet")
axs[3].set_xlabel("p_t of jet")


plt.suptitle("Phythia_MLM (red) vs reweighted (green), clustered with fastjet_kt")
plt.show()

