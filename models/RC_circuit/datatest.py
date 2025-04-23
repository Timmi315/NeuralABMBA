import torch
from os.path import dirname as up
import numpy as np
from DataGeneration import generate_synthetic_data
import matplotlib.pyplot as plt  # Corrected import

def plot_data(data, dt, cfg):
    
    # Plot the data
    time_steps = dt * torch.arange(cfg['num_steps'] + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, data[:, 0, 0], label='T_in')
    #plt.plot(time_steps, data[:, 1, 0], label='T_out')
    #plt.plot(time_steps, data[:, 2, 0], label='Q_H')
    #plt.plot(time_steps, data[:, 3, 0], label='Q_O')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title('RC Circuit Data')
    plt.legend()
    plt.grid(True)
    plt.show()
    

def ls_estimation(data, dt, gamma):
    # Assuming data is a torch.Tensor of shape [num_steps, 4, 1] with columns [T_in, T_out, Q_H, Q_O]
    # and dt is defined

    # Initialize parameters
    C = torch.tensor([1.0], requires_grad=True)  # Initial guess and requires_grad=True to enable gradient computation
    R = torch.tensor([89.0], requires_grad=True)  # Initial guess

    # Optimizer setup
    optimizer = torch.optim.Adam([C, R], lr=0.25)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Number of epochs for the optimization
    num_epochs = 1000

    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Clear gradients from the previous iteration

        # Compute the predicted next T_in using the model equation
        T_in_predicted = data[:-1, 0, 0] + dt / C * ((data[:-1, 1, 0] - data[:-1, 0, 0]) / R + data[:-1, 2, 0] + data[:-1, 3, 0])

        # Calculate the loss (Mean Squared Error)
        loss = (T_in_predicted - data[1:, 0, 0]).pow(2).mean()  # Comparing to the next actual T_in value

        # Compute gradients
        loss.backward()

        # Update parameters
        optimizer.step()
        scheduler.step()

        # Optional: Print loss every 100 epochs
        if epoch % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(current_lr)
            print(f'Epoch {epoch+1}, Loss: {loss.item()}, C: {C.item()}, R: {R.item()}')

    # Final parameters
    print(f'Estimated C: {C.item()}, Estimated R: {R.item()}')

#TIMO GND --||-- T_in --[__]-- T_out ?
#TIMO       C            R              is that the circuit?
cfg = {
    "initial_conditions": {
        "T_in": {
            "distribution": "uniform",
            "parameters": {"lower": 290, "upper": 290}
        },
        "T_out": {
            "distribution": "uniform",
            "parameters": {"lower": 270, "upper": 270}
        },
        #Heating inside
        "Q_H": {
            "distribution": "uniform",
            "parameters": {"lower": 0, "upper": 0}
        },
        #Heating outside
        "Q_O": {
            "distribution": "uniform",
            "parameters": {"lower": 0, "upper": 0}
        }
    },
    "num_steps": 10*24*60,  # You can adjust the number of steps as needed
    "C": 3.0,          # Capacitance
    "R": 100,           # Resistance
    "T_out_std": 0,   # Standard deviation for T_out fluctuation
    "Q_std": 0        # Standard deviation for Q_H and Q_O fluctuation
    }

dt = 1  #Time differential
data = generate_synthetic_data(cfg, dt=dt) # type: torch.Tensor, size [num_steps, num_variables, 1], Generate synthetic data

plot_data(data, dt, cfg)
#ls_estimation(data, dt, gamma=1)