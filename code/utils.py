import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def method_type(value):
    value_upper = value.upper()
    if value_upper not in {"AP", "WSAP"}:
        raise argparse.ArgumentTypeError("Method must be 'AP' or 'WSAP' (case-insensitive)")
    return value_upper

def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    # Required positional argument: input file
    parser.add_argument("filename", type=str, help="Path to the input data file")
    # Optional string argument: method
    parser.add_argument('--method', type=method_type, default="AP",
                        help="METHOD to use: AP or WSAP (case-insensitive). Default: AP")
    # Optional boolean argument: non-inverted Y axis
    parser.add_argument('--non-inverseY', action='store_true', default=False,
                        help='Use non-inverted Y axis')
    return parser.parse_args()

def plot_result(t_obs, m_obs, 
                t_array, y_array_fit, 
                C4, C5, 
                time_of_extremum, time_extr_sig,
                mag_of_extremum, mag_extr_sig,
                inverseY):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 14
    plt.rcParams["figure.figsize"] = (11,7)
    
    fig, ax = plt.subplots()
    if inverseY:
        ax.invert_yaxis()
    ax.scatter(t_obs, m_obs, c="blue")
    ax.plot(t_array, y_array_fit, color='green', linewidth=2)
    ax.axvline(x = C4, color = 'maroon', linewidth=1)
    ax.axvline(x = C5, color = 'maroon', linewidth=1)
    
    rect = Rectangle(
        (time_of_extremum - time_extr_sig, mag_of_extremum - mag_extr_sig),
        2 * time_extr_sig, 2 * mag_extr_sig,
        linewidth=2, edgecolor='r', facecolor='none'
    )
    
    ax.add_patch(rect)
    ax.plot([time_of_extremum - time_extr_sig, time_of_extremum + time_extr_sig], [mag_of_extremum, mag_of_extremum], 'r-', linewidth=2)
    ax.plot([time_of_extremum, time_of_extremum], [mag_of_extremum - mag_extr_sig, mag_of_extremum + mag_extr_sig], 'r-', linewidth=2)
    
    plt.show()

def save_result(method,
                time_of_extremum, time_extr_sig,
                mag_of_extremum, mag_extr_sig,
                params_opt, param_errors):
    m = 0
    with open("approx_result.txt", "w") as f:
        s = f"Method: {method}"
        print(s)
        f.write(s + "\n")
        s = f"Time of extremum   = \t{time_of_extremum:.5f}"
        print(s)
        f.write(s + "\n")
        s = f"Time of extr. err. = \t{time_extr_sig:.5f}"
        print(s)
        f.write(s + "\n")
        s = f"Mag of extremum    = \t{mag_of_extremum:.4f}"
        print(s)
        f.write(s + "\n")
        s = f"Mag of extr. err.  = \t{mag_extr_sig:.4f}"
        print(s)
        f.write(s + "\n")
        print()
        f.write("\n")
        s = "Param\tValue\tUncertainty"
        print(s)
        f.write(s + "\n")
        for p, e in zip(params_opt, param_errors):
            m += 1
            s = f"C{m}\t{p}\t{e}"
            print(s)
            f.write(s + "\n")
