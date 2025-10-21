from colorama import Fore, Back
import argparse
import io
import base64
import numpy as np
from . import ila

def printWarning(msg):
    print(Fore.LIGHTRED_EX + Back.LIGHTYELLOW_EX + msg + Fore.RESET + Back.RESET)

def method_type(value):
    value_upper = value.upper()
    if value_upper not in {"AP", "WSAP", "WSL", "A", "0"}:
        raise argparse.ArgumentTypeError("Method must be AP, WSAP, WSL, A, or 0 (case-insensitive). Use 0 to plot the data without approximation.")
    return value_upper

def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    # Required positional argument: input file
    parser.add_argument("filename", type=str, help="Path to the input data file")
    # Optional string argument: method
    parser.add_argument('--method', type=method_type, default="AP",
                        help="METHOD to use: AP, WSAP, WSL, A or 0 (case-insensitive). Default: AP")
    # Optional boolean argument: non-inverted Y axis
    parser.add_argument('--non-inverseY', action='store_true', default=False,
                        help='Use non-inverted Y axis')
    # Optional boolean argument: do not show plot in the single-extremum mode
    parser.add_argument('--no-plot', action='store_true', default=False,
                        help='Do not display plot in the single-extremum mode')
    # Optional string argument: file of ranges for batch processing
    parser.add_argument('--ranges', type=str, default="",
                        help='File of ranges for batch processing')
    # Optional string argument: result file name
    parser.add_argument('--result', type=str, default="result.txt",
                        help='File with the resulting parameters of the approximation')
    # Optional string argument: preview file name
    parser.add_argument('--preview', type=str, default="result.html",
                        help='HTML file with the plot')
    return parser.parse_args()

def generate_curve(method, params_opt, t_obs):
    t_min = min(t_obs)
    t_max = max(t_obs)
    t_array = np.linspace(t_min, t_max, 10000)
    if method == "AP":
        C1, C2, C3, C4, C5 = params_opt
        y_array_fit = ila.f_AP_a(t_array, C1, C2, C3, C4, C5)
        y_array_fit_at_points = ila.f_AP_a(t_obs, C1, C2, C3, C4, C5)
    elif method == "WSAPA":
        C1, C2, C3, C4, C5, C6, C7 = params_opt
        y_array_fit = ila.f_WSAPA_a(t_array, C1, C2, C3, C4, C5, C6, C7)
        y_array_fit_at_points = ila.f_WSAPA_a(t_obs, C1, C2, C3, C4, C5, C6, C7)
    elif method == "WSAP":
        C1, C2, C3, C4, C5 = params_opt
        y_array_fit = ila.f_WSAP_a(t_array, C1, C2, C3, C4, C5)
        y_array_fit_at_points = ila.f_WSAP_a(t_obs, C1, C2, C3, C4, C5)
    elif method == "WSL":
        C1, C2, C3, C4, C5 = params_opt
        y_array_fit = ila.f_WSL_a(t_array, C1, C2, C3, C4, C5)
        y_array_fit_at_points = ila.f_WSL_a(t_obs, C1, C2, C3, C4, C5)
    elif method == "A":
       C1, C2, C3, C4 = params_opt
       C5 = None
       y_array_fit = ila.f_A_a(t_array, C1, C2, C3, C4)
       y_array_fit_at_points = ila.f_A_a(t_obs, C1, C2, C3, C4)
    else:
        raise Exception(f"Unsupported method: {method}")
       
    return t_array, y_array_fit, y_array_fit_at_points


def plot_result(t_obs, m_obs, 
                t_array, y_array_fit, 
                C4, C5, 
                time_of_extremum, time_extr_sig,
                mag_of_extremum, mag_extr_sig,
                inverseY,
                info_message,
                to_buf = False):
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 12
    plt.rcParams["figure.figsize"] = (8,5)
    
    fig, ax = plt.subplots()
    if inverseY:
        ax.invert_yaxis()
    ax.scatter(t_obs, m_obs, c="blue")
    if not (t_array is None):
        ax.plot(t_array, y_array_fit, color='green', linewidth=2)
    if not (C4 is None):
        ax.axvline(x = C4, color = 'maroon', linewidth=1)
    if not (C5 is None):
        ax.axvline(x = C5, color = 'maroon', linewidth=1)

    #print(time_of_extremum)
    #print(mag_of_extremum)
    
    if (time_of_extremum is not None and 
        time_extr_sig is not None and 
        mag_of_extremum is not None and
        mag_extr_sig is not None):
        rect = Rectangle(
            (time_of_extremum - time_extr_sig, mag_of_extremum - mag_extr_sig),
            2 * time_extr_sig, 2 * mag_extr_sig,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        ax.plot([time_of_extremum - time_extr_sig, time_of_extremum + time_extr_sig], [mag_of_extremum, mag_of_extremum], 'r-', linewidth=2)
        ax.plot([time_of_extremum, time_of_extremum], [mag_of_extremum - mag_extr_sig, mag_of_extremum + mag_extr_sig], 'r-', linewidth=2)

    if time_of_extremum is not None and mag_of_extremum is not None:
        ax.plot(time_of_extremum, mag_of_extremum, 'o', markersize=4, color='red')

    if info_message is not None:
        #print(info_message)
        ax.set_title(info_message, fontsize=10, color="red")
    
    if not to_buf:
        plt.show()
        return None
    else:
        # Save figure to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)
        return encoded
        