import sys
from os.path import dirname as up

import coloredlogs
import h5py as h5
import numpy as np
import torch
from dantro import logging
from dantro._import_tools import import_module_from_path
import random

sys.path.append(up(up(__file__)))
sys.path.append(up(up(up(__file__))))

RC_model = import_module_from_path(mod_path=up(up(__file__)), mod_str="RC_model")
base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")

log = logging.getLogger(__name__)
coloredlogs.install(fmt="%(levelname)s %(message)s", level="INFO", logger=log)


class RC_model_NN:
    def __init__(
        self,
        *,
        rng: np.random.Generator,
        h5group: h5.Group,
        neural_net: base.NeuralNet,
        loss_function: dict,
        to_learn: list,
        dt: float,
        true_parameters: dict = {},
        write_every: int = 1,
        write_start: int = 1,
        training_data: torch.Tensor,
        external_data: torch.Tensor,
        physical,
        batch_size: int,
        input_size: int,
        scaling_factors: dict = {},
        **__,
    ):
        """Initialize the model instance with a previously constructed RNG and
        HDF5 group to write the output data to.

        :param rng (np.random.Generator): The shared RNG
        :param h5group (h5.Group): The output file group to write data to
        :param neural_net: The neural network
        :param loss_function (dict): the loss function to use
        :param to_learn: the list of parameter names to learn
        :param true_parameters: the dictionary of true parameters
        :param training_data: the time series of T_in data to calibrate
        :param external_data: the time series of the external data (T_out, Q_H, Q_O)
        :param write_every: write every iteration
        :param write_start: iteration at which to start writing
        :param batch_size: epoch batch size: instead of calculating the entire time series,
            only a subsample of length batch_size can be used. The likelihood is then
            scaled up accordingly.
        :param scaling_factors: factors by which the parameters are to be scaled
        """
        self._h5group = h5group
        self._rng = rng

        self.neural_net = neural_net
        self.neural_net.optimizer.zero_grad()
        self.loss_function = base.LOSS_FUNCTIONS[loss_function.get("name").lower()](
            loss_function.get("args", None), **loss_function.get("kwargs", {})
        )

        self.current_loss = torch.tensor(0.0)

        self.to_learn = {key: idx for idx, key in enumerate(to_learn)}
        self.true_parameters = {
            key: torch.tensor(val, dtype=torch.float)
            for key, val in true_parameters.items()
        }
        self.current_predictions = torch.zeros(len(to_learn))

        # Training data (time series of T_in) and external forcing (T_out, Q_H, Q_O)
        self.training_data = training_data
        self.external_data = external_data
        self.physical = physical

        # Time differential to use for the numerical solver
        self.dt = torch.tensor(dt).float()

        # Generate the batch ids
        '''
        batches = np.arange(0, self.training_data.shape[0], batch_size)
        if len(batches) == 1:
            batches = np.append(batches, training_data.shape[0] - 1)
        else:
            if batches[-1] != training_data.shape[0] - 1:
                batches = np.append(batches, training_data.shape[0] - 1)
        self.batches = batches
        '''

        # Scaling factors to use for the parameters, if given
        self.scaling_factors = scaling_factors

        # --- Set up chunked dataset to store the state data in --------------------------------------------------------
        # Write the loss after every batch
        self._dset_loss = self._h5group.create_dataset(
            "loss",
            (0,),
            maxshape=(None,),
            chunks=True,
            compression=3,
        )
        self._dset_loss.attrs["dim_names"] = ["batch"]
        self._dset_loss.attrs["coords_mode__batch"] = "start_and_step"
        self._dset_loss.attrs["coords__batch"] = [write_start, write_every]

        # Write the computation time of every epoch
        self.dset_time = self._h5group.create_dataset(
            "computation_time",
            (0,),
            maxshape=(None,),
            chunks=True,
            compression=3,
        )
        self.dset_time.attrs["dim_names"] = ["epoch"]
        self.dset_time.attrs["coords_mode__epoch"] = "trivial"

        # Write the parameter predictions after every batch
        self.dset_parameters = self._h5group.create_dataset(
            "parameters",
            (0, len(self.to_learn.keys())),
            maxshape=(None, len(self.to_learn.keys())),
            chunks=True,
            compression=3,
        )
        self.dset_parameters.attrs["dim_names"] = ["batch", "parameter"]
        self.dset_parameters.attrs["coords_mode__batch"] = "start_and_step"
        self.dset_parameters.attrs["coords__batch"] = [write_start, write_every]
        self.dset_parameters.attrs["coords_mode__parameter"] = "values"
        self.dset_parameters.attrs["coords__parameter"] = to_learn

        # The training data and batch ids
        timeseries_length = training_data.shape[1]
        number_of_timeseries = training_data.shape[0]
        self.training_data = training_data

        self.batch_size = batch_size

        batches = np.arange(0, training_data.shape[0], batch_size)
        if len(batches) == 1:
            batches = np.append(batches, training_data.shape[0] - 1)
        else:
            if batches[-1] != training_data.shape[0] - 1:
                batches = np.append(batches, training_data.shape[0] - 1)
        self.batches = batches

        # Batches processed
        self._time = 0
        self._write_every = write_every
        self._write_start = write_start

        self.input_size = input_size


    def shuffle_data(self):
        windows_per_series = int(self.training_data.shape[1]/self.batch_size)
        out = []
        for i in range(windows_per_series):
            for k in range(self.training_data.shape[0]):
                index = np.random.randint(self.training_data.shape[1]-self.batch_size)
                out.append((k, slice(index, index+self.batch_size)))
        random.shuffle(out)
        return out

    def epoch(self):
        """
        An epoch is a pass over the entire dataset. The dataset is processed in batches, where B < L is the batch
        number. After each batch, the parameters of the neural network are updated. For example, if L = 100 and
        B = 50, two passes are made over the dataset -- one over the first 50 steps, and one
        over the second 50. The entire time series is processed, even if L is not divisible into equal segments of
        length B. For instance, is B is 30, the time series is processed in 3 steps of 30 and one of 10.

        """

        # Process the training data in batches
        
        for batch_no, batch in enumerate(self.shuffle_data()):
            # Make a prediction
            
            predicted_parameters = self.neural_net(torch.flatten(torch.cat(
                                                        (self.training_data[batch][:self.input_size],
                                                        self.external_data[batch][:self.input_size]), 1)))

            # get the parameters
            parameters = [self.scaling_factors.get(key , 1.0)*predicted_parameters[self.to_learn[key]]
                            if key in self.to_learn.keys() else self.true_parameters[key]
                            for key in self.physical.parameter_names]

             # Get current initial condition and make traceable
            current_densities = self.training_data[batch][0].clone()
            current_densities.requires_grad_(True)

            loss = torch.tensor(0.0, requires_grad=True)

            for t in range(1, self.batch_size):
                # simulate using the parameters
                current_densities = torch.stack(self.physical.step(current_densities, self.external_data[batch][t-1], parameters, self.dt))

                # calculate loss
                loss = loss + self.loss_function(
                        current_densities, self.training_data[batch][t])
                #    ) /self.batch_size
            loss.backward()
            self.neural_net.optimizer.step()
            self.neural_net.optimizer.zero_grad()
            self.current_loss = loss.clone().detach().cpu().numpy().item()
            self.current_predictions = predicted_parameters.clone().detach().cpu()

            self._time += 1
            self.write_data()
        '''
            
        for batch_no, batch_idx in enumerate(self.batches[:-1]):

            # Make a prediction
            newdata = torch.flatten(torch.cat((self.training_data[batch_idx:batch_idx+self.input_size],
                                                self.external_data[batch_idx:batch_idx+self.input_size]), 1))
            predicted_parameters = self.neural_net(
                #torch.flatten(self.training_data[batch_idx])
                newdata
            )

            # Get the parameters: resistance and capacity
            parameters = [self.scaling_factors.get(key , 1.0)*predicted_parameters[self.to_learn[key]]
                            if key in self.to_learn.keys() else self.true_parameters[key]
                            for key in self.physical.parameter_names]
            #parameters = [4896000, 0.00531, 1112400, 0.000639]
            #parameters = [7452000, 0.00529]
            R = (
                self.scaling_factors.get("R", 1.0) * predicted_parameters[self.to_learn["R"]]
                if "R" in self.to_learn.keys()
                else self.true_parameters["R"]
            )
            C = (
                self.scaling_factors.get("C", 1.0) * predicted_parameters[self.to_learn["C"]]
                if "C" in self.to_learn.keys()
                else self.true_parameters["C"]
            )

            # Get current initial condition and make traceable
            current_densities = self.training_data[batch_idx].clone()
            current_densities.requires_grad_(True)

            loss = torch.tensor(0.0, requires_grad=True)

            for t in range(batch_idx + 1, self.batches[batch_no + 1] + 1):
                #print(current_densities)
                #print(self.external_data[:,t])
                #print(C)
                #print(R)
                # Generate the next step of the time series
                current_densities = torch.stack(self.physical.step(current_densities, self.external_data[t], parameters, self.dt))
                current_densities = current_densities + self.dt / C * (
                        (self.external_data[t][0] - current_densities) / R
                        + self.external_data[t][1]
                        + self.external_data[t][2]
                )

                # Calculate loss
                loss = loss + self.loss_function(
                    current_densities, self.training_data[t]
                ) * (
                    self.training_data.shape[0]
                    / (self.batches[batch_no + 1] - batch_idx)

                )
            '''
            

    def write_data(self):
        """Write the current state (loss and parameter predictions) into the state dataset.

        In the case of HDF5 data writing that is used here, this requires to
        extend the dataset size prior to writing; this way, the newly written
        data is always in the last row of the dataset.
        """
        if self._time >= self._write_start and (self._time % self._write_every == 0):
            self._dset_loss.resize(self._dset_loss.shape[0] + 1, axis=0)
            self._dset_loss[-1] = self.current_loss
            self.dset_parameters.resize(self.dset_parameters.shape[0] + 1, axis=0)
            self.dset_parameters[-1, :] = [
                self.current_predictions[self.to_learn[p]] for p in self.to_learn.keys()
            ]
