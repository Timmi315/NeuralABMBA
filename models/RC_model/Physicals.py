import torch
import torch.nn as nn

class Physical(nn.Module):
    def __init__(self):
        super().__init__()
    def reset(self):
        pass

    def set_params(self, params):
        self.params = params
        #print("setting params")

    def forward(self, t, state):
        #print(f"fw input: ({t}, {state})")
        #print(state[0] + self.dt/self.params[0]*((self.external_data[int(t.item())][0]-state[0])/self.params[1] + self.external_data[int(t.item())][1] + self.external_data[int(t.item())][2]))
        out = self.step(state, self.external_data[int(t.item())], self.params, int(self.dt))
        #print(out)
        #out = torch.reshape(out, (1,self.dynamic_variables))
        #print(f"fw output: {out-state}")
        return out - state

class RC(Physical):
    parameter_names = ["C", "R"]
    dynamic_variables = 1
    data_names = ["T_in", "T_out", "heatPower", "solarGains"]
    plot_args = {
        "T_in": {
            "offset": -273.15,
            "multi": 1,
            "kwargs": {}
        },
        "T_out":{
            "offset": -273.15,
            "multi": 1,
            "kwargs": {}
        },
        "heatPower [kW]": {
            "offset": 0,
            "multi": 0.001,
            "kwargs": {"alpha": 0.5}
        },
        "Q_solar [kW]": {
            "offset" : 0,
            "multi": 0.001,
            "kwargs": {"alpha": 0.5}
        }
    }

    def initial_condition(self, cfg, wdata):
        return [cfg["initial_conditions"]["T_in"], wdata[0, 2] + 273.15, 0, 0]


    def step(self, densities, data, params, dt):
        #for generation:
        # densities: dynamic variables
        # data: [T_ambient, heatPower(T_in), Q_solar]
        # params: parameters defined in config
        #for simulation in NN:
        # densities: first dynamic variables elements of data (one point in time)
        # data: rest of data generated through step() (one point in time)
        # params: estimated or set parameters in shape of parameter_names

        return (densities[0]+dt/params[0]*((data[0]-densities[0])/params[1] + data[1] + data[2])).unsqueeze(0)

class TiTh(Physical):
    parameter_names = ["C1", "R1", "C2", "R2"]
    dynamic_variables = 2
    data_names = ["T_in", "T_heater", "T_out", "heatPower", "solarGains"]
    plot_args = {
        "T_in": {
            "offset": -273.15,
            "multi": 1,
            "kwargs": {}
        },
        "T_heater": {
            "offset": -273.15,
            "multi": 1,
            "kwargs": {"alpha": 0.5}
        },
        "T_out":{
            "offset": -273.15,
            "multi": 1,
            "kwargs": {}
        },
        "heatPower [kW]": {
            "offset": 0,
            "multi": 0.001,
            "kwargs": {"alpha": 0.5}
        },
        "Q_solar": {
            "offset" : 0,
            "multi": 0.001,
            "kwargs": {"alpha": 0.5}
        }
    }

    def initial_condition(self, cfg, wdata):
        return [cfg["initial_conditions"]["T_in"], cfg["initial_conditions"]["T_in"], wdata[0, 2] + 273.15, 0, 0]


    def step(self, densities, data, params, dt):
        #for generation:
        # densities: dynamic variables
        # data: [T_ambient, heatPower(T_in), Q_solar]
        # params: parameters defined in config
        #for simulation in NN:
        # densities: first dynamic variables elements of data (one point in time)
        # data: rest of data generated through step() (one point in time)
        # params: estimated or set parameters in shape of parameter_names
        T_heater = densities[1] + dt/params[2] * (data[1] + (densities[0] - densities[1])/params[3])
        T_in = densities[0] + dt/params[0] * ((T_heater - densities[0])/params[3] + (data[0] - densities[0])/params[1] + data[2])
        return torch.stack([T_in, T_heater])

    class Hidden(RC):
        parameter_names = ["C1", "R1", "C2", "R2"]
        T_heater = None

        def step(self, densities, data, params, dt):
            # densities[1] -> data[0]
            if self.T_heater is None:
                self.T_heater = [densities[0]]
            self.T_heater.append(self.T_heater[-1] + dt/params[2] * (data[1] + (densities[0] - self.T_heater[-1]) / params[3]))
            return (densities[0] + dt/params[0] * ((self.T_heater[-1] - densities[0])/params[3] + (data[0] - densities[0])/params[1] + data[2])).unsqueeze(0)

        def reset(self):
            self.T_heater = None
