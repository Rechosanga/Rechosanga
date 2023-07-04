import torch
import torch.nn as nn
from ST_Transformer import STTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    days = 10        # Select the number of days to train
    val_days = 3    # Select the number of days to verify
    
    train_num = 288*days
    val_num = 288*val_days
    row_num = train_num + val_num

    v = pd.read_csv("PEMSD7/V_25.csv", nrows = row_num, header= None)
    A = pd.read_csv("PEMSD7/W_25.csv", header= None)  # get adjacency matrix
 
    A = np.array(A)
    A = torch.tensor(A, dtype=torch.float32)
       
    v = np.array(v)
    v = v.T
    v = torch.tensor(v, dtype=torch.float32)
    # finally v shape:[N, T]。  N=25, T=row_num
    # Model parameters
    A = A           # adjacency matrix
    in_channels=1   # Enter the number of channels. Only speed information, so channel is 1
    embed_size=64   # Transformer number of channels
    time_num = 288  # 1Number of day intervals
    num_layers=1    # Spatial-temporal block stacked layers
    T_dim=12        # Enter the time dimension. Enter the previous 1 hour data, so 60min/5min = 12
    output_T_dim=3  # Output time dimension. Predict the speed of the next 15, 30, 45 minutes
    heads=1         # transformer head quantity。 Same number of time and space transformer heads
    
    # model input shape: [1, N, T]   
    # 1:Initial number of channels, N: number of sensors, T: number of times
    # model output shape: [N, T]    
    model = STTransformer(
        A,
        in_channels, 
        embed_size, 
        time_num, 
        num_layers, 
        T_dim, 
        output_T_dim, 
        heads
    )   
    
    # optimizer, lr, loss  According to the requirements of the thesis
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    criterion = nn.L1Loss()                             
    #   ----training part----
    # t Indicates the specific time traversed
    pltx=[]
    plty=[]

    for t in range(train_num - 21):
        x = v[:, t:t+12]
        x = x.unsqueeze(0)        
        y = v[:, t+14:t+21:3]
        # x shape:[1, N, T_dim] 
        # y shape:[N, output_T_dim]
        
        out = model(x, t)
        loss = criterion(out, y) 
        
        if t%100 == 0:
            print("MAE loss:", loss)
        
        # normal operation
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() 
        
        pltx.append(t)
        plty.append(loss.detach().numpy())
    
    plt.plot(pltx, plty, label="STTN train")
    plt.title("ST-Transformer train")
    plt.xlabel("t")
    plt.ylabel("MAE loss")
    plt.legend()
    plt.show() 
 
    # save the model
    print(model.pth)
    torch.save(model, "model.pth")