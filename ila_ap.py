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
import numpy as np
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code'))
import utils
import ila

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
        
    if method == "AP" or method == "WSAP":
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
    
    t_array, y_array_fit, y_array_fit_at_points, C4, C5 = utils.generate_curve(method, params_opt, t_obs)
    
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
    
    result_file = "approx_batch_result.txt"
    preview_file = "approx_batch_result.html"
    
    with open(result_file, "w") as f:
        info_str = "method\tt_start\tt_stop\ttime_of_extremum\ttime_extr_sig\tmag_of_extremum\tmag_extr_si\teclipse_duration\teclipse_sig"
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
        
        info_str = f"{method}\t{t_start}\t{t_stop}"
        print(info_str)
        
        if method == "AP" or method == "WSAP":
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
        
        t_array, y_array_fit, y_array_fit_at_points, C4, C5 = utils.generate_curve(method, params_opt, time_subset)
        
        info_str += f"\t{time_of_extremum}\t{time_extr_sig}\t{mag_of_extremum}\t{mag_extr_sig}\t{eclipse_duration}\t{eclipse_sig}"
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
            f_preview.write(f"<p>{info_str}</p>\n")
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
            input("Press ENTER to continue: ")
            sys.exit()
