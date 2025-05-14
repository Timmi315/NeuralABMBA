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
        plt.plot(time_steps, (data[:, i, 0]+plot_args[key]["offset"])*plot_args[key]["multi"], label=key, **plot_args[key]["kwargs"])
    plt.xlabel('Time/s')
    plt.ylabel('Values')
    plt.title('RC Circuit Data')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_parameter_space(cfg, dt, res = 100, searchspace = 10, maxloss = 100):
    target = generate_weather_based_data(cfg, dt = dt)
    #plot_data(target, dt, cfg)
    #c_range = torch.linspace(cfg["C"] - cfg["C"]/searchspace*10, cfg["C"] + cfg["C"]/searchspace*20, steps = res)
    c_range = torch.linspace(0.6e7, 0.9e7, steps = res)
    #r_range = torch.linspace(cfg["R"] - cfg["R"]/searchspace, cfg["R"] + cfg["R"]/searchspace, steps = res)
    r_range = torch.linspace(0.00525, 0.00535, steps = res)
    grid1, grid2 = torch.meshgrid(c_range, r_range, indexing = 'ij')
    print(f"Starting to explore parameter space from (C = {grid1.min()}, R = {grid2.min()}) to (C = {grid1.max()}, R = {grid2.max()})")
    loss_func = torch.nn.MSELoss(reduction = "sum")
    physical = getattr(Physicals, cfg["model_type"])
    losses = torch.empty_like(grid1)
    dat = target[:,physical.dynamic_variables:,:]
    init = target[:,:physical.dynamic_variables,:][0]
    target = target[:,:physical.dynamic_variables,:]

    for i in range(grid1.shape[0]):
        for j in range(grid1.shape[1]):
            dense = init
            loss = torch.tensor(0.0, requires_grad=False)
            for k in range(cfg["num_steps"]):
                dense = torch.stack(physical.step(dense, dat[k], (grid1[i,j], grid2[i,j]), dt))
                loss = loss+loss_func(dense, target[k])
                if (loss > maxloss):
                    loss = maxloss
                    break
            losses[i,j] = loss
        print(f"explored (C = {grid1[i,j]}, R [{grid2[i,0]}, {grid2[i,-1]}]): {losses[i].mean()}")

    plt.contourf(grid1.numpy(), grid2.numpy(), losses.detach().numpy(), levels=2000)
    plt.colorbar(label='Loss')
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot_surface(grid1, grid2, losses)
    plt.xlabel('C')
    plt.ylabel('R')
    plt.title('Loss landscape')
    
    plt.show()




    

def ls_estimation(cfg, data, dt, gamma, num_epochs = 1000):
    physical = getattr(Physicals, cfg["model_type"])
    # Assuming data is a torch.Tensor of shape [num_steps, 4, 1] with columns [T_in, T_out, Q_H, Q_O]
    # and dt is defined

    # Initialize parameters
    #params = [torch.tensor([10.0], requires_grad=True) for _ in physical.parameter_names]
    #C = torch.tensor([1.0], requires_grad=True)  # Initial guess and requires_grad=True to enable gradient computation
    #R = torch.tensor([89.0], requires_grad=True)  # Initial guess
    #
    params = []
    scales = []
    for key in physical.parameter_names:
        params.append(torch.tensor([0.5], requires_grad=True))
        if key in cfg["scales"]:
            scales.append(cfg["scales"][key])
        else:
            scales.append(1)
    

    #params = [torch.tensor([10.0], requires_grad=True) for _ in physical.parameter_names]
    #print(params)
    #scales = [10e6, 10e-3]
    # Optimizer setup
    optimizer = torch.optim.Adam(params, lr=0.002)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Number of epochs for the optimization

    lossFun = torch.nn.MSELoss()

    opt = [10]
    opt.append(params)

    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Clear gradients from the previous iteration
        loss = 0
        prms = [params[i]*scales[i] for i in range(len(params))]
        #print(prms)
        for t in range(data.shape[0]-1):
        

            # Compute the predicted next T_in using the model equation
            predicted = torch.stack(physical.step(data[t, :physical.dynamic_variables, 0], data[t, physical.dynamic_variables:, 0], prms, dt))

            loss = loss + lossFun(predicted[:,0], data[t+1, :physical.dynamic_variables, 0])

        if loss < opt[0]:
            opt[0] = loss
            opt[1] = prms

        # Compute gradients
        loss.backward()

        # Update parameters
        optimizer.step()
        scheduler.step()

        # Optional: Print loss every 100 epochs
        if epoch % 20 == 0:
            current_lr = scheduler.get_last_lr()[0]
            #print(current_lr)
            print(f'===== Epoch {epoch+1}, lr: {current_lr}, Loss: {loss.item()} =====')
            for i in range(len(params)):
                print(f"{physical.parameter_names[i]} = {scales[i]*params[i].item()} = {scales[i]} * {params[i].item()}")
            print("\n")

    # Final parameters
    print(f'Estimated params: {[f'{physical.parameter_names[i]}: {prms[i].item()}' for i in range(len(params))]}')

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
    "num_steps": 1440,  # You can adjust the number of steps as needed (rn: 1/100year in minutes)
    "T_min": 290, #[K]
    "T_max": 294, #[K]
    "C": [7452000, 7500000],         # Capacitance [Ws/°C]
    "R": [0.00529, 0.006],          # Resistance [°C/W]
    "C1": 4896000,
    "R1": 0.00531,
    "C2": 1112400,
    "R2": 0.000639,
    "scales": {"C": 10e6, "R": 10e-3}
    }

dt = 900 #Time differential in seconds
data = generate_weather_based_data(cfg, dt=dt)[0] # type: torch.Tensor, size [num_steps, num_variables, 1], Generate synthetic data
print(data.shape)

#plot_data(data, dt, cfg)
#plot_parameter_space(cfg, dt)
ls_estimation(cfg, data, dt, gamma=1, num_epochs = 10000)