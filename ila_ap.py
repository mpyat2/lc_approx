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
DEBUG = True

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

if sys.version_info < (3, 8):
    raise RuntimeError("Python 3.8 or higher is required")

colorama_init()

###############################################################################

def process_data(data_file_name, method, inverseY):
    
    data = pd.read_csv(data_file_name, 
                       comment='#', skip_blank_lines=True,
                       sep="\\s+",
                       names=['time', 'mag'], 
                       dtype={'time': 'float64', 'mag': 'float64'}, 
                       usecols=['time', 'mag'])
    print(f"File loaded: {len(data['mag'])} points")
    
    t_obs = data['time']
    m_obs = data['mag']
    
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
    
    print("Approximation started....")
    params_opt, params_cov, param_warning = ila.approx(method, t_obs, m_obs, maxfev=MAXFEV)
    print("Approximation successful.")
    print()
        
    if method == "AP" or method == "WSAP" or method == "WSL":
        if params_opt[3] >= params_opt[4]:
            raise Exception("C4 must be less than C5. Aborting.")
    
    # 1-sigma uncertainties
    param_errors = np.sqrt(np.diag(params_cov))
    
    [time_of_extremum, 
     time_extr_sig, 
     mag_of_extremum,
     mag_extr_sig,
     eclipse_duration,
     eclipse_sig
    ] = ila.method_result(method, params_opt, params_cov, min(t_obs), max(t_obs))
    
    utils.save_result(f"approx_result-{method}.txt",
                      method,
                      time_of_extremum, time_extr_sig,
                      mag_of_extremum, mag_extr_sig,
                      params_opt, param_errors,
                      param_warning)
    
    t_array, y_array_fit, y_array_fit_at_points = utils.generate_curve(method, params_opt, t_obs)
    C4 = params_opt[3]
    if len(params_opt) > 4:
        C5 = params_opt[4]
    else:
        C5 = None
    
    utils.save_approx(f"approx_data-{method}.txt",
                      t_obs, y_array_fit_at_points, m_obs)
       
    print()
    sigma = np.sqrt(np.sum((m_obs - y_array_fit_at_points)**2) / (len(m_obs) - len(params_opt)))
    print("sigma        = ", sigma)
    sigma = np.sqrt(np.sum((m_obs - y_array_fit_at_points)**2) * len(params_opt) / len(m_obs) / (len(m_obs) - len(params_opt)))    
    print("sigma_m[x_c] = ", sigma, " # r.m.s. accuracy of the fit")
    if method == "WSL":
        print()        
        print("Eclipse duration (C5 - C4) = ", eclipse_duration)
        print("Eclipse duration error     = ", eclipse_sig)

    utils.plot_result(t_obs, m_obs, 
                      t_array, y_array_fit, 
                      C4, C5, 
                      time_of_extremum, time_extr_sig,
                      mag_of_extremum, mag_extr_sig,
                      inverseY,
                      param_warning)


def process_batch(data_file_name, range_file_name, method, inverseY):
    
    data = pd.read_csv(data_file_name, 
                       comment='#', skip_blank_lines=True,
                       sep="\\s+",
                       names=['time', 'mag'], 
                       dtype={'time': 'float64', 'mag': 'float64'}, 
                       usecols=['time', 'mag'])
    print(f"File loaded: {len(data['mag'])} points")
    
    times = data['time']
    mags  = data['mag']

    # Sort by times (just in case)
    times, mags = zip(*sorted(zip(times, mags)))
    times = np.array(times)
    mags = np.array(mags)
    
    ranges = pd.read_csv(range_file_name, 
                         comment='#', 
                         skip_blank_lines=True,
                         sep="\\s+",
                         names=['point1', 'time1', 'point2', 'time2'],
                         dtype={'point1': 'int32', 'time1': 'float64', 'point2': 'int32', 'time2': 'float64'},
                         usecols=['point1', 'time1', 'point2', 'time2'])
 
    print(f"Ranges loaded: {len(ranges)} ranges")
    
    result_file = f"approx_batch_result-{method}.txt"
    preview_file = f"approx_batch_result-{method}.html"

    info = {
        'Method': method,
        'Points': None,
        'Start Time': None,
        'End Time': None,
        'Sigma': None,
        'Time of Extremum (TOM)': None,
        'TOM Uncertainty': None,
        'Magnitude': None,
        'Magnitude Uncertainty': None,
        'Eclipse Duration': None,
        'Eclipse Duration Uncertainty': None
        }

    info_str = "\t".join(list(info.keys())[:-2])
    if method == "WSL":
        info_str += "\t" + list(info.keys())[-2]
        info_str += "\t" + list(info.keys())[-1]
    with open(result_file, "w") as f:
        f.write(info_str + "\n")
        f.flush()
        
    with open(preview_file, "w") as f_preview:
        f_preview.write("<html><body>\n")
        f_preview.write("<h2>Preview</h2>\n")
        f_preview.write("<hr>\n")
        
    
    for idx, row in ranges.iterrows():
        t_start = row['time1']
        t_stop = row['time2']
        mask = (times >= t_start) & (times <= t_stop)
        time_subset = times[mask]
        mag_subset  = mags[mask]
       
        params_opt, params_cov, param_warning = ila.approx(method, time_subset, mag_subset, maxfev=MAXFEV)
        
        info['Method'] = method
        info['Points'] = len(mag_subset)
        info['Start Time'] = t_start
        info['End Time'] = t_stop
        info_str = "\t".join(f"{k}: {v}" for k, v in list(info.items())[:4])
        print(info_str)
        
        if method == "AP" or method == "WSAP" or method == "WSL":
            if params_opt[3] >= params_opt[4]:
                info_str = info_str + "\tFailed: C4 must be less than C5."
                f.write(info_str + "\n")
                continue

        # 1-sigma uncertainties
        #param_errors = np.sqrt(np.diag(params_cov))
        
        [time_of_extremum, 
         time_extr_sig, 
         mag_of_extremum,
         mag_extr_sig,
         eclipse_duration,
         eclipse_sig
        ] = ila.method_result(method, params_opt, params_cov, min(time_subset), max(time_subset))
        
        t_array, y_array_fit, y_array_fit_at_points = utils.generate_curve(method, params_opt, time_subset)
        C4 = params_opt[3]
        if len(params_opt) > 4:
            C5 = params_opt[4]
        else:
            C5 = None

        sigma = np.sqrt(np.sum((mag_subset - y_array_fit_at_points)**2) / (len(mag_subset) - len(params_opt)))
        
        info['Sigma'] = sigma
        info['Time of Extremum (TOM)'] = time_of_extremum
        info['TOM Uncertainty'] = time_extr_sig
        info['Magnitude'] = mag_of_extremum
        info['Magnitude Uncertainty'] = mag_extr_sig
        info['Eclipse Duration'] = eclipse_duration
        info['Eclipse Duration Uncertainty'] = eclipse_sig
        
        if param_warning is not None:
            info['Method'] = method + " WARNING! " + param_warning
        else:
            info['Method'] = method
            
        info_str = "\t".join(str(v) for v in list(info.values())[:-2])
        info_str2 = " | ".join(f"{k}: {v}" for k, v in list(info.items())[:-2])
        if method == "WSL":
            info_str += "\t" + str(list(info.values())[-2])
            info_str += "\t" + str(list(info.values())[-1])
            info_str2 += " | " + f"{list(info.keys())[-2]}: {str(list(info.values())[-2])}"
            info_str2 += " | " + f"{list(info.keys())[-1]}: {str(list(info.values())[-1])}"
        with open(result_file, "a") as f:            
            f.write(info_str + "\n")
            f.flush()

        encoded = utils.plot_result(time_subset, mag_subset, 
                                    t_array, y_array_fit, 
                                    C4, C5, 
                                    time_of_extremum, time_extr_sig,
                                    mag_of_extremum, mag_extr_sig,
                                    inverseY,
                                    param_warning,
                                    True)
        
        with open(preview_file, "a") as f_preview:
            f_preview.write(f"<p>{info_str2}</p>\n")
            f_preview.write(f"<img src='data:image/png;base64,{encoded}'><br><br>\n")
            f_preview.write("<hr>\n")
            f_preview.flush()

    with open(preview_file, "a") as f_preview:
        f_preview.write("<p>End of file</p>\n")
        f_preview.write("\n</body></html>")                
       

###############################################################################

def main():
    args = utils.parse_args(DESCRIPTION)
    ranges = args.ranges
    method = args.method.upper()
    if ranges != "":
        if method == "0":
            raise Exception(f"Method {method} is not applicable in this context")            
        process_batch(args.filename, ranges, method, not args.non_inverseY)
    else:
        process_data(args.filename, method, not args.non_inverseY)

if __name__ == "__main__":
    if DEBUG:
        main()
    else:    
        try:
            main()
        except Exception as e:
            print(f"Fatal Error: {e}")
        finally:
            #input("Press ENTER to continue: ")
            sys.exit()
