
class RC:
    parameter_names = ["C", "R"]
    dynamic_variables = 1
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

    def initial_condition(cfg, wdata):
        return [cfg["initial_conditions"]["T_in"], wdata[0, 2] + 273.15, 0, 0]


    def step(densities, data, params, dt):
        #for generation:
        # densities: dynamic variables
        # data: [T_ambient, heatPower(T_in), Q_solar]
        # params: parameters defined in config
        #for simulation in NN:
        # densities: first dynamic variables elements of data (one point in time)
        # data: rest of data generated through step() (one point in time)
        # params: estimated or set parameters in shape of parameter_names

        return [densities[0]+dt/params[0]*((data[0]-densities[0])/params[1] + data[1] + data[2])]


class TiTh:
    parameter_names = ["C1", "R1", "C2", "R2"]
    dynamic_variables = 2
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
        "heatPower": {
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

    def initial_condition(cfg, wdata):
        return [cfg["initial_conditions"]["T_in"], cfg["initial_conditions"]["T_in"], wdata[0, 2] + 273.15, 0, 0]


    def step(densities, data, params, dt):
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
        return [T_in, T_heater]

