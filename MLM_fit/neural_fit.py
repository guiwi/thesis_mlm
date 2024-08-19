import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import numpy as np
import math
from matplotlib import pyplot as plt
import pyhepmc as hp

def comp_R(mom4):
    if (mom4[0] != 0 and mom4[1] != 0 and mom4[2] != 0):
        phi = math.atan2(mom4[0],mom4[1])
        mag = math.sqrt(mom4[0]**2+mom4[1]**2+mom4[2]**2)
        nu = 0.5*math.log((mag+mom4[2])/(mag-mom4[2]))
        return math.sqrt(nu**2+phi**2)
    else : return 0

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(9, 15),
            nn.Linear(15,15),
            nn.ReLU(),
            nn.Linear(15,15),
            nn.Linear(15,15),
            nn.Linear(15, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits
    

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 64 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            for i in range(len(pred)):
                if pred[i] == y[i]:
                    correct += 1

    test_loss /= num_batches
    correct /= size
    print(f"Avg loss: {test_loss:>8f} \n")
    return test_loss

def make_data_set(file,forgraph,hinge):
    #number of expected events (can be bigger than the actual number but not smaller)
    nEvents = 1000000
    
    jet_moms = np.zeros((nEvents,3,4))
    g_data_2 = np.zeros((nEvents,9))
    g_data_3 = np.zeros((nEvents,9))
    g_scales = np.zeros((nEvents,4))
    accepted0 = np.zeros(nEvents)

    g_type = np.zeros((nEvents,2))
    g_for_R = np.zeros((nEvents,4))

    event_count = 0
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if row[0] == 'j_pt':
                    print(row)
            else:
                event_count +=1
                accepted0[event_count] = float(row[5])
                jet_moms[event_count,0,:] = [float(row[1]),float(row[2]),float(row[3]),float(row[4])]
                jet_moms[event_count,1,:] = [float(row[11]),float(row[12]),float(row[13]),float(row[14])]
                jet_moms[event_count,2,:] = [float(row[15]),float(row[16]),float(row[17]),float(row[18])]


    #The two jet multiplicities are splitted
    len_2 = 0
    len_3 = 0
    indi_2 = np.zeros(nEvents)
    indi_3 = np.zeros(nEvents)
    accepted0_2 = np.zeros(nEvents)
    accepted0_3 = np.zeros(nEvents)
    for i in range(event_count):
        hardness = np.argsort(jet_moms[i,:,3])
        if jet_moms[i,hardness[0],0] == 0:
            accepted0_2[len_2] = accepted0[i]
            indi_2[len_2] = i
            
            g_data_2[len_2,0] = abs(jet_moms[i,hardness[-1],0])
            g_data_2[len_2,1] = abs(jet_moms[i,hardness[-1],1])
            g_data_2[len_2,2] = jet_moms[i,hardness[-1],3]

            g_data_2[len_2,3] = abs(jet_moms[i,hardness[-2],0])
            g_data_2[len_2,4] = abs(jet_moms[i,hardness[-2],1])
            g_data_2[len_2,5] = jet_moms[i,hardness[-2],3]

            g_data_2[len_2,6] = abs(jet_moms[i,hardness[-3],0])
            g_data_2[len_2,7] = abs(jet_moms[i,hardness[-3],1])
            g_data_2[len_2,8] = jet_moms[i,hardness[-3],3]

            len_2 +=1
        else :
            accepted0_3[len_3] = accepted0[i]
            indi_3[len_3] = i

            g_data_3[len_3,0] = abs(jet_moms[i,hardness[-1],0])
            g_data_3[len_3,1] = abs(jet_moms[i,hardness[-1],1])
            g_data_3[len_3,2] = jet_moms[i,hardness[-1],3]

            g_data_3[len_3,3] = abs(jet_moms[i,hardness[-2],0])
            g_data_3[len_3,4] = abs(jet_moms[i,hardness[-2],1])
            g_data_3[len_3,5] = jet_moms[i,hardness[-2],3]

            g_data_3[len_3,6] = abs(jet_moms[i,hardness[-3],0])
            g_data_3[len_3,7] = abs(jet_moms[i,hardness[-3],1])
            g_data_3[len_3,8] = jet_moms[i,hardness[-3],3]

            len_3 +=1



    print("------------------------")
    print(event_count)
    print("---------------------")

    accepted0 = accepted0[:event_count]
    g_data_2 = g_data_2[:len_2,:]
    g_data_3 = g_data_3[:len_3,:]
    accepted0_2 = accepted0_2[:len_2]
    accepted0_3 = accepted0_3[:len_3]
    indi_2 = indi_2[:len_2]
    indi_3 = indi_3[:len_3]

    
    if hinge:
        for kj in range(len(accepted0)):
            if accepted0[kj] == 0: accepted0[kj] =-1


    accepted0.astype(np.float32)
    

    #Normalize the data by their maximum
   

    for i in range(len(g_data_2[0,:])-3):
        each_max = np.max(np.abs(g_data_2[:,i]))
        for j in range(len(g_data_2[:,0])):
            g_data_2[j,i] /= each_max

    for i in range(len(g_data_3[0,:])-3):
        each_max = np.max(np.abs(g_data_3[:,i]))
        for j in range(len(g_data_3[:,0])):
            g_data_3[j,i] /= each_max

            

    g_data_2.astype(np.float32)

    data_X_2 = torch.tensor(g_data_2,device=device,dtype=torch.float32)
    data_X_2 = data_X_2.to(device)

    data_y_2 = torch.tensor((accepted0_2),device=device,dtype=torch.float32)
    data_y_2 = data_y_2.to(device)

    data_set_2 = torch.utils.data.TensorDataset(data_X_2,data_y_2.unsqueeze(1))
    data_loader_2 = torch.utils.data.DataLoader(data_set_2, batch_size=64, shuffle=False)

    g_data_3.astype(np.float32)

    data_X_3 = torch.tensor(g_data_3,device=device,dtype=torch.float32)
    data_X_3 = data_X_3.to(device)

    data_y_3 = torch.tensor((accepted0_3),device=device,dtype=torch.float32)
    data_y_3 = data_y_3.to(device)

    data_set_3 = torch.utils.data.TensorDataset(data_X_3,data_y_3.unsqueeze(1))
    data_loader_3 = torch.utils.data.DataLoader(data_set_3, batch_size=64, shuffle=False)

    #Neural training takes data_loader but graph don't
    if forgraph:
        return data_X_2,data_y_2,data_X_3,data_y_3,indi_2,indi_3
    else:
        return data_loader_2,data_loader_3



device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


#Input csv files
#Good practice would require 3 different files here
#One for the training, one for the testing and one for the graphs/output
made_train_2,made_train_3 = make_data_set("weights_fit1.csv",False,False)
made_test_2,made_test_3 = make_data_set("weights_fit2.csv",False,False)
graph_X_2,graph_y_2,graph_X_3,graph_y_3,graph_ind_2,graph_ind_3 = make_data_set("3.csv",True,False)
ckkw_X_2,ckkw_Y_2,ckkw_X_3,ckkw_Y_3,ckkw_ind_2,ckkw_ind_3 = make_data_set("3.csv",True,False)

#output file
output_file = "pred_fit.txt"

###################Network#####################################

### Model for 2 jets
model_2 = NeuralNetwork().to(device)
print(model_2)

#loss_fn = nn.BCELoss()
loss_fn = nn.MSELoss()
#loss_fn = nn.L1Loss()
#loss_fn = nn.NLLLoss()

final_error = 0
optimizer = torch.optim.SGD(model_2.parameters(), lr=0.05)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(made_train_2, model_2, loss_fn, optimizer)
    final_error = test_loop(made_test_2, model_2, loss_fn)
    scheduler.step()
print("Done!")

## Model for 3 jets
model_3 = NeuralNetwork().to(device)
print(model_3)



final_error = 0
optimizer = torch.optim.SGD(model_3.parameters(), lr=0.02)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.93)
epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(made_train_3, model_3, loss_fn, optimizer)
    final_error = test_loop(made_test_3, model_3, loss_fn)
    scheduler.step()
print("Done!")

###############################################xw 
################   Graph   ##########################
###############################################


##"Classic" graphs

pred_tograph_2 = model_2(graph_X_2)
pred_tograph_3 = model_3(graph_X_3)



fig, axs = plt.subplots(2,9, sharex=True)

steps = 101
temp_lin = np.linspace(0,1,steps)

graph_X_2_np = graph_X_2.cpu().detach().numpy()
graph_X_3_np = graph_X_3.cpu().detach().numpy()
print("-----------------")


graph_X = np.zeros(((len(graph_ind_2)+len(graph_ind_3)),9))
graph_y = np.zeros((len(graph_ind_2)+len(graph_ind_3)))
pred_tograph = np.zeros((len(graph_ind_2)+len(graph_ind_3)))

#To make the graphs, data from both jet multiplicities have to be reasembled
incr2 = 0
incr3 = 0
for lm in range(len(graph_X[:,0])):
    if lm in graph_ind_2:
        graph_X[lm,:] = graph_X_2_np[incr2,:]
        graph_y[lm] = graph_y_2[incr2]
        pred_tograph[lm] = pred_tograph_2[incr2]
        incr2 +=1
    else :
        graph_X[lm,:] = graph_X_3_np[incr3,:]
        graph_y[lm] = graph_y_3[incr3]
        pred_tograph[lm] = pred_tograph_3[incr3]
        incr3 +=1

for j in range(9):

    minus = np.min(graph_X[:,j])
    maxus = np.max(np.abs(graph_X[:,j]))

    if minus < 0:
        fraction = 2*maxus/(steps-1)
    else:
        fraction = maxus/(steps-1)

    vec_hist = np.zeros(steps)
    normalise = np.zeros(steps)
    for i in range(len(graph_X[:,j])):
        if minus < 0:
            index = math.floor((graph_X[i,j]+minus)/fraction)
        else :
            index = math.floor((graph_X[i,j])/fraction)
        vec_hist[index] += graph_y[i]
        normalise[index] += 1

    new_hist = np.zeros(steps)
    for k in range(steps):
        if normalise[k] != 0:
            new_hist[k] = vec_hist[k]/normalise[k]


    axs[0,j].plot(temp_lin,new_hist,label="Qcut = 60")
    axs[1,j].bar(temp_lin,normalise,0.01)


    vec_hist = np.zeros(steps)
    normalise = np.zeros(steps)
    for i in range(len(graph_X[:,j])):
        if minus < 0:
            index = math.floor((graph_X[i,j]+minus)/fraction)
        else :
            index = math.floor((graph_X[i,j])/fraction)
        vec_hist[index] += pred_tograph[i]
        normalise[index] += 1

    new_hist = np.zeros(steps)
    for k in range(steps):
        if normalise[k] != 0:
            new_hist[k] = vec_hist[k]/normalise[k]


    axs[0,j].plot(temp_lin,new_hist,label="Pred")




axs[1,0].set(xlabel="pt of j",ylabel="# of Events")


axs[0,0].set(ylabel="Selected Fraction")


axs[0,0].set_title("px of jet 1")

axs[0,1].set_title("py of jet 1")
axs[0,2].set_title("E of jet 1")
axs[0,3].set_title("px of jet 2")
axs[0,4].set_title("py of jet 2")
axs[0,5].set_title("E of jet 2")
axs[0,6].set_title("px of jet 3")
axs[0,7].set_title("py of jet 3")
axs[0,8].set_title("E of jet 3")


axs[0,0].legend()

titel = "e+e- > jjj 100k events, scaled 0-1"
fig.suptitle(titel)

plt.show()

temp_lin = np.linspace(0,1,50)
n_pred = np.zeros(50)
for ik in range(len(pred_tograph)):
    if pred_tograph[ik] > 1 : pred_tograph[ik] = 1
    if graph_y[ik] == 0:
        n_pred[math.floor(pred_tograph[ik]*49.999)] += 1

fig, axs = plt.subplots(1, sharex=True)

axs.bar(temp_lin,n_pred,width=0.01)

temp_lin = np.linspace(0,1,50)
n_pred2 = np.zeros(50)
for ik in range(len(pred_tograph)):
    if pred_tograph[ik] > 1 : pred_tograph[ik] = 1
    if graph_y[ik] == 1:
        n_pred2[math.floor(pred_tograph[ik]*49.999)] += 1

axs.bar(temp_lin,n_pred2,bottom=n_pred,width=0.01)
plt.suptitle("distribution of prediction, event without hard jet removed")

plt.show()



pred_ckkw_2 = model_2(ckkw_X_2)
pred_ckkw_2 = pred_ckkw_2.cpu().detach().numpy()
pred_ckkw_3 = model_3(ckkw_X_3)
pred_ckkw_3 = pred_ckkw_3.cpu().detach().numpy()

pred_ckkw = np.zeros(len(ckkw_ind_2)+len(ckkw_ind_3))

incr2 = 0
incr3 = 0
for lm in range(len(graph_X[:,0])):
    if lm in ckkw_ind_2:
        pred_ckkw[lm] = pred_ckkw_2[incr2]
        incr2 +=1
    else :
        pred_ckkw[lm] = pred_ckkw_3[incr3]
        incr3 +=1

for i in range(len(pred_ckkw)):
    if pred_ckkw[i] > 1 : pred_ckkw[i] = 1

f = open(output_file,"w")
for i in range(len(pred_ckkw)):
    f.write(str(float(pred_ckkw[i])))
    f.write("\n")
