import torch
from os.path import dirname as up
import numpy as np
from DataGeneration import generate_weather_based_data
import matplotlib.pyplot as plt  # Corrected import
import Physicals

def plot_data(data, dt, cfg):
    
    plot_args = getattr(Physicals, cfg["model_type"]).plot_args
    # Plot the data
    time_steps = dt * torch.arange(cfg['num_steps'] + 1)
    plt.figure(figsize=(10, 6))
    for i, key in enumerate(plot_args.keys()):
        plt.plot(time_steps, data[:, i, 0]+plot_args[key]["offset"], label=key, **plot_args[key]["kwargs"])
    plt.xlabel('Time/s')
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
    "model_type": "RC",
    "initial_conditions": {
        "T_in": 290,
        "T_out": 270,
        #Heating inside
        "Q_H": 0,
        #Heating outside
        "Q_O": 0
    },
    "effWinArea": 7.89, #[m2] so given Solar radiance [W/m2]*effWinArea = [W]
    "maxHeatingPower": 5000, #[W]
    "controller": "PControl", #PControl, TwoPointControl
    "num_steps": 1000,  # You can adjust the number of steps as needed (rn: 1year in minutes)
    "T_min": 290, #[K]
    "T_max": 294, #[K]
    "C": 7452000,          # Capacitance [Ws/°C]
    "R": 0.00529,           # Resistance [°C/W]
    "C1": 4896000,
    "R1": 0.00531,
    "C2": 1112400,
    "R2": 0.000639
    }

dt = 900 #Time differential in seconds
data = generate_weather_based_data(cfg, dt=dt) # type: torch.Tensor, size [num_steps, num_variables, 1], Generate synthetic data

plot_data(data, dt, cfg)
#ls_estimation(data, dt, gamma=1)