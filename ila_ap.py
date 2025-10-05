###############################################################################

DESCRIPTION = \
"""
I.L.A. approximations -- Version 0.01 --

Methods:
    AP: Asymptotic Parabola;
    WSAP: Wall-Supported Asymptotic Parabola;
    A: two straight lines crossed near the extremum
"""

###############################################################################

# Set to True to get better error info
DEBUG = False

# Try to increase this parameter (number of iterations) if the solution cannot be found
#MAXFEV = 50000
MAXFEV = 100000

###############################################################################

import os
import sys
from colorama import init as colorama_init
import numpy as np
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code'))
import utils
import ila

colorama_init()

###############################################################################

def process_data(data_file_name, method, inverseY, showPlot, range_file_name, result_file_name, preview_file_name):
    
    data = pd.read_csv(data_file_name, 
                       comment='#', skip_blank_lines=True,
                       sep="\\s+",
                       names=['time', 'mag'], 
                       dtype={'time': 'float64', 'mag': 'float64'}, 
                       usecols=['time', 'mag'])
    print(f"File loaded: {len(data['mag'])} points")
    
    t_obs = data['time']
    m_obs = data['mag']

    # Sort by times (essential in batch mode)
    t_obs, m_obs = zip(*sorted(zip(t_obs, m_obs)))
    t_obs = np.array(t_obs)
    m_obs = np.array(m_obs)
    
    if method == "0":
        # Plot and exit
        utils.plot_result(t_obs, m_obs, 
                          None, None,
                          None, None, 
                          None, None,
                          None, None,
                          inverseY,
                          None)
        sys.exit()
    
    if range_file_name == "":
        range_file_name = None
    
    if range_file_name is None:
        # One-extremum mode
        print('One-extremum mode')
        ranges = pd.DataFrame(
            np.array([[1, min(t_obs), len(t_obs), max(t_obs)]]),
            columns=['point1', 'time1', 'point2', 'time2'],                          
            )
        ranges = ranges.astype({'point1': 'int32', 'time1': 'float64', 'point2': 'int32', 'time2': 'float64'})        
    else:
        ranges = pd.read_csv(range_file_name, 
                             comment='#', 
                             skip_blank_lines=True,
                             sep="\\s+",
                             names=['point1', 'time1', 'point2', 'time2'],
                             dtype={'point1': 'int32', 'time1': 'float64', 'point2': 'int32', 'time2': 'float64'},
                             usecols=['point1', 'time1', 'point2', 'time2'])
        print(f"Range file loaded: {len(ranges)} ranges")

    info_keys = [
        'Method',
        'Points',
        'Start Time',
        'End Time',
        'Sigma',
        'Time of Extremum (TOM)',
        'TOM Uncertainty',
        'Magnitude',
        'Magnitude Uncertainty',
        'C4',
        'C4 Uncertainty',
        'C5',
        'C5 Uncertainty',
        'Eclipse Duration',
        'Eclipse Duration Uncertainty'
        ]

    info = {
        info_keys[0]: method,
        info_keys[1]: None,
        info_keys[2]: None,
        info_keys[3]: None,
        info_keys[4]: None,
        info_keys[5]: None,
        info_keys[6]: None,
        info_keys[7]: None,
        info_keys[8]: None,
        info_keys[9]: None,
        info_keys[10]: None,
        info_keys[11]: None,
        info_keys[12]: None,
        info_keys[13]: None,
        info_keys[14]: None
        }

    info_str = "\t".join(info_keys[:-2])
    if method == "WSL":
        info_str += "\t" + info_keys[-2]
        info_str += "\t" + info_keys[-1]
    with open(result_file_name, "w") as f:
        f.write(info_str + "\n")
        f.flush()
        
    with open(preview_file_name, "w") as f_preview:
        f_preview.write("<html><body>\n")
        f_preview.write("<h2>Preview</h2>\n")
        f_preview.write("<hr>\n")

    for idx, row in ranges.iterrows():
        t_start = row['time1']
        t_stop = row['time2']
        mask = (t_obs >= t_start) & (t_obs <= t_stop)
        time_subset = t_obs[mask]
        mag_subset  = m_obs[mask]

        info['Method'] = method
        info['Points'] = len(mag_subset)
        info['Start Time'] = t_start
        info['End Time'] = t_stop
        info_str = "\t".join(f"{k}: {info[k]}" for k in info_keys[:4])
        print(info_str)

        params_opt, params_cov, param_warning = ila.approx(method, time_subset, mag_subset, maxfev=MAXFEV)
        if param_warning is not None:
            utils.printWarning(param_warning)
        
        if method == "AP" or method == "WSAP" or method == "WSL":
            if params_opt[3] >= params_opt[4]:
                info_str = info_str + f"\tFailed: C4 must be less than C5. C4 = {params_opt[3]}; C5 = {params_opt[4]}"
                utils.printWarning(info_str)
                with open(result_file_name, "a") as f:
                    f.write(info_str + "\n")
                with open(preview_file_name, "a") as f_preview:
                    f_preview.write("<p>Failed. See the file with results.</p>\n")
                    
                continue

        # 1-sigma uncertainties
        param_errors = np.sqrt(np.diag(params_cov))
        
        [time_of_extremum, 
         time_extr_sig, 
         mag_of_extremum,
         mag_extr_sig,
         eclipse_duration,
         eclipse_sig,
         param_warning1
        ] = ila.method_result(method, params_opt, params_cov, min(time_subset), max(time_subset))
        if param_warning1 is not None:
            utils.printWarning(param_warning1)
        
        t_array, y_array_fit, y_array_fit_at_points = utils.generate_curve(method, params_opt, time_subset)
        C4 = params_opt[3]
        C4_err = param_errors[3]
        if len(params_opt) > 4:
            C5 = params_opt[4]
            C5_err = param_errors[4]
        else:
            C5 = None
            C5_err = None

        sigma = np.sqrt(np.sum((mag_subset - y_array_fit_at_points)**2) / (len(mag_subset) - len(params_opt)))
        
        info['Sigma'] = sigma
        info['Time of Extremum (TOM)'] = time_of_extremum
        info['TOM Uncertainty'] = time_extr_sig
        info['Magnitude'] = mag_of_extremum
        info['Magnitude Uncertainty'] = mag_extr_sig
        info['C4'] = C4
        info['C4 Uncertainty'] = C4_err
        info['C5'] = C5
        info['C5 Uncertainty'] = C5_err
        info['Eclipse Duration'] = eclipse_duration
        info['Eclipse Duration Uncertainty'] = eclipse_sig
        
        if param_warning is not None or param_warning1 is not None:
            if param_warning is None:
                param_warning = ""    
            if param_warning1 is None:
                param_warning1 = ""
            if param_warning != "" and param_warning1 != "":
                param_warning += ";"
            param_warning += param_warning1
            info['Method'] = method + " WARNING! " + param_warning
        else:
            info['Method'] = method
            
        info_str = "\t".join(str(info[k]) for k in info_keys[:-2])
        info_str2 = " | ".join(f"{k}: {info[k]}" for k in info_keys[:-2])
        if method == "WSL":
            info_str += "\t" + str(info[info_keys[-2]])
            info_str += "\t" + str(info[info_keys[-1]])
            info_str2 += " | " + f"{info_keys[-2]}: {str(info[info_keys[-2]])}"
            info_str2 += " | " + f"{info_keys[-1]}: {str(info[info_keys[-1]])}"
        with open(result_file_name, "a") as f:            
            f.write(info_str + "\n")
            f.flush()

        if range_file_name is None:
            # One-extremum mode
            print('-' * 80)
            info_list2 = info_str2.split(" | ")
            for i in range(4, len(info_list2)): print(info_list2[i])
            if showPlot:
                utils.plot_result(time_subset, mag_subset, 
                                  t_array, y_array_fit, 
                                  C4, C5, 
                                  time_of_extremum, time_extr_sig,
                                  mag_of_extremum, mag_extr_sig,
                                  inverseY,
                                  param_warning,
                                  False)

        encoded = utils.plot_result(time_subset, mag_subset, 
                                    t_array, y_array_fit, 
                                    C4, C5, 
                                    time_of_extremum, time_extr_sig,
                                    mag_of_extremum, mag_extr_sig,
                                    inverseY,
                                    param_warning,
                                    True)
        
        with open(preview_file_name, "a") as f_preview:
            f_preview.write(f"<p>{info_str2}</p>\n")
            f_preview.write(f"<img src='data:image/png;base64,{encoded}'><br><br>\n")
            f_preview.write("<hr>\n")
            f_preview.flush()

    with open(preview_file_name, "a") as f_preview:
        f_preview.write("<p>End of file</p>\n")
        f_preview.write("\n</body></html>")                

###############################################################################

def main():
    args = utils.parse_args(DESCRIPTION)
    range_file_name = args.ranges
    result_file_name = args.result
    preview_file_name = args.preview
    method = args.method.upper()
    if range_file_name != "":
        if method == "0":
            raise Exception(f"Method {method} is not applicable in this context")            
    process_data(args.filename, method, not args.non_inverseY, not args.no_plot, range_file_name, result_file_name, preview_file_name)

if __name__ == "__main__":
    if DEBUG:
        import scipy
        import numpy
        print("Executable: ", sys.executable)
        print("scipy.__version__: ", scipy.__version__)
        print("numpy.__version__", numpy.__version__)
        main()
    else:    
        try:
            main()
        except Exception as e:
            print(f"Fatal Error: {e}")
        finally:
            #input("Press ENTER to continue: ")
            sys.exit()
