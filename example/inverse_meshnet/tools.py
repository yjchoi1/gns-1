import numpy as np
import torch
from scipy.optimize import curve_fit


def vel_autogen(ly, shape_option, args):
    def quadratic(x, a, b, c):
        return a * x ** 2 + b * x + c

    # if sim_config["initial_vel"]["autogen"]:
    print("Generate velocity boundary condition randomly")

    if shape_option == "uniform":
        if isinstance(args, dict):
            peak = np.random.uniform(args['peak'][0], args['peak'][1])
            npoints = args['npoints']
            v_left_np = np.full(npoints, peak)
        else:
            if args[2] != ly:
                raise ValueError("Size of input initial velocity array should be same as `ly`")
            else:
                v_left_np = np.random.uniform(args[0], args[1], args[2])

        v_left_np[0] = 0
        v_left_np[-1] = 0

    elif shape_option == "normal":
        raise NotImplementedError

    elif shape_option == "quad":
        peak = np.random.uniform(args['peak'][0], args['peak'][1])
        npoints = args['npoints']
        x_data = np.array([0, ly / 2, ly])
        y_data = np.array([0, peak, 0])

        coefficients, _ = curve_fit(quadratic, x_data, y_data)
        x_values = np.linspace(x_data.min(), x_data.max(), npoints)

        v_left_np = quadratic(x_values, *coefficients)

    elif shape_option == "multi_quad":
        peak = np.random.uniform(args['peak'][0], args['peak'][1])
        npoints = args['npoints']
        if npoints % 4 == 0:
            # set x for 1/2 of the entire left boundary
            x_data = np.array([0, ly / 4, ly / 2])
            y_data = np.array([0, peak, 0])
            coefficients, _ = curve_fit(quadratic, x_data, y_data)
            x_values = np.linspace(x_data.min(), x_data.max(), int(npoints / 2))
            partial_v = quadratic(x_values, *coefficients)
            # Sample first half of the `partial_v_left`
            partial_v_half = partial_v[0:int(npoints / 4)]
            v_left_np = np.concatenate((np.flip(partial_v_half), partial_v, partial_v_half))
        else:
            raise NotImplementedError("ly should be divisible by 4")

    elif shape_option == "from_csv":
        raise NotImplementedError

    else:
        raise ValueError("Not implemented velocity option. Choose among `normal`, `quad, `multi_quad`")

    v_left_np[0] = 0
    v_left_np[-1] = 0

    return v_left_np


class To_Torch_Model_Param(torch.nn.Module):
    def __init__(self, parameters):
        super(To_Torch_Model_Param, self).__init__()
        self.current_params = torch.nn.Parameter(parameters)

