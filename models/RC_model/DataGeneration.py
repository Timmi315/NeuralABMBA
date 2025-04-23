import h5py as h5
import logging
import sys
import torch
from os.path import dirname as up
import numpy as np
import pandas as pd
import Physicals

sys.path.append(up(up(up(__file__))))

from dantro._import_tools import import_module_from_path

base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")
print(up(up(up(__file__))))

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------------------------------
# Data loading and generation utilities
# ----------------------------------------------------------------------------------------------------------------------


def apply_controller(cfg, data, heatPower_index):
    if cfg["controller"] == "TwoPointControl":
        if data[0] < cfg["T_min"]:
            heatPower = cfg["maxHeatingPower"]
        elif [0] >= cfg["T_max"]:
            heatPower = 0
        else:
            heatPower = data[heatPower_index]
    elif cfg["controller"] == "PControl":
        heatPower = int(data[0]<=cfg["T_max"])*(cfg["T_max"]-data[0])*cfg["maxHeatingPower"]/(cfg["T_max"]-cfg["T_min"])
        if heatPower > cfg["maxHeatingPower"]:
            heatPower = cfg["maxHeatingPower"]
    return heatPower


def generate_weather_based_data(cfg: dict, *, dt: float) -> torch.Tensor:
    """ Function that generates weather data based time series of length num_steps for T_in, T_out, Q_H, Q_O.

    :param cfg: configuration of data settings
    :param dt: time differential for the numerical solver (Euler in this case)
    :return: torch.Tensor of the time series for T_in, T_out, Q_H, Q_O. Tensor has shape (4, num_steps)
    """

    # Draw an initial condition for the data using the prior defined in the config
    # Uses the random_tensor function defined in include.utils

    wdata, _ = read_mos(up(up(up(__file__))) + "/data/RC_model/weatherData/Munich_5years.mos")
    wdata = wdata.to_numpy(dtype = float)

    model = getattr(Physicals, cfg["model_type"])
    initial_condition = torch.Tensor(model.initial_condition(cfg, wdata))

    data = [initial_condition]
    parameters = [cfg[param] for param in model.parameter_names]

    # Generate some synthetic time series
    for i in range(cfg['num_steps']):

        # Solve the equation for T_in and generate a time series for T_out and the Q values dt/C*((T_in-T_out)/R + QH + QO)
        heatPower = apply_controller(cfg, data[-1], list(model.plot_args).index("heatPower"))

        # these format acrobatics are done to use the same step function in the NN and here
        densities = data[-1][:model.dynamic_variables]
        dat = [wdata[int(i*dt/3600)][2]+273.15, heatPower, wdata[int(i*dt/3600)][8]*cfg["effWinArea"]]
 
        data.append(torch.cat((
            torch.Tensor(model.step(densities, dat, parameters, dt)),
            torch.Tensor(dat)
            )))
    return torch.reshape(torch.stack(data), (len(data), len(model.plot_args), 1))


def get_RC_circuit_data(*, data_cfg: dict, h5group: h5.Group):
    """Returns the training data for the RC_circuit model. If a directory is passed, the
    data is loaded from that directory (csv output file from BuildA). Otherwise, synthetic training data is generated
    by iteratively solving the temporal ODE system.

    :param data_cfg: dictionary of config keys
    :param h5group: h5.Group to write the training data to
    :return: torch.Tensor of training data

    """
    if "load_from_dir" in data_cfg.keys():
        #with h5.File(data_cfg["load_from_dir"], "r") as f:
        #    data = torch.from_numpy(np.array(f["RC_model"]["RC_data"])).float()
        with open(data_cfg["load_from_dir"]["path"], "r") as f:
            df = pd.read_csv(f)
            data = torch.from_numpy(np.array([[df["thermalZone.TAir"],
                                                df["weaBus.TDryBul"],
                                                df["totalHeatingPower.y"],
                                                np.array(df["weaDat.weaBus.HGloHor"])*data_cfg["load_from_dir"]["effWinArea"]]]
                                                )).float()[:, :, :data_cfg["load_from_dir"]["subset"]].T
        data_cfg["dt"] = 900 #since BuildA calculates in quarterhourly steps
        print(data.shape)

    elif "synthetic_data" in data_cfg.keys():
        data_cfg["synthetic_data"]["model_type"] = data_cfg["model_type"]
        data = generate_weather_based_data(
            cfg=data_cfg["synthetic_data"],
            dt=data_cfg["dt"]
        )
        attributes = list(getattr(Physicals, data_cfg["model_type"]).plot_args.keys())
    
    else:
        raise ValueError(
            f"You must supply one of 'load_from_dir' or 'synthetic data' keys!"
        )

    # Store the synthetically generated data in an h5 file
    dset = h5group.create_dataset(
        "RC_data",
        data.shape,
        maxshape=data.shape,
        chunks=True,
        compression=3,
        dtype=float,
    )

    dset.attrs["dim_names"] = ["time", "kind", "dim_name__0"]
    dset.attrs["coords_mode__time"] = "trivial"
    dset.attrs["coords_mode__kind"] = "values"
    dset.attrs["coords__kind"] = attributes
    dset.attrs["coords_mode__dim_name__0"] = "trivial"

    dset[:, :] = data

    return data

def read_mos(filename):

    print(f"Reading reference file {filename}")
    with open(filename, "r") as f:
        data = f.read()

    # scans for header and data and splits it
    n_cols = int(data[data.find(",")+1: data.find(")")])  #start of header contains "*double tab1(rows,cols)\n*"
    last_header_line = data.find(f"C{n_cols}")
    header_end = data[last_header_line:].find("\n")+last_header_line
    header = data[:header_end+1] 
    dat = data[header_end+1:]

    #converts the data to a numpy array
    arr1 = dat.split("\n")[:-1]
    arr2 = np.array([i.split("\t") for i in arr1])
    header_shape = (int(header[header.find("(")+1:header.find(",")]), int(header[header.find(",")+1:header.find(")")]))
    if arr2.shape != header_shape:
        print(f"ERROR while reading .mos file: list dimensions {arr2.shape} do not match header {header_shape}!")
        exit()
    df = pd.DataFrame(arr2, columns = [f"C{i+1}" for i in range(n_cols)])
    return df, header

#get_RC_circuit_data(data_cfg = {"load_from_dir": {"path": "C:/Users/Timo/Documents/SublimeProjects/NeuralABMBA/data/RC_model/MunichPI.csv", "effWinArea": 1.5}}, h5group = None)