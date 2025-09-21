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

JD_SHIFT = 0 #-2400000

# Try to increase this parameter (number of iterations) if the solution cannot be found
#MAXFEV = 50000
MAXFEV = 100000

###############################################################################

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code'))
import utils

###############################################################################

def process_data(data_file_name, method, inverseY=True):
    import ila
    import numpy as np
    import pandas as pd

    data = pd.read_csv(data_file_name, 
                       comment='#', skip_blank_lines=True,
                       sep="\\s+",
                       #header=0,
                       names=['time', 'mag'], 
                       dtype={'time': 'float64', 'mag': 'float64'}, 
                       usecols=['time', 'mag'])
    print(f"File loaded: {len(data['mag'])} points")
    
    t_obs = data['time'] + JD_SHIFT
    m_obs = data['mag']
    
    if method == "0":
        # Plot and exit
        utils.plot_result(t_obs, m_obs, 
                          None, None,
                          None, None, 
                          None, None,
                          None, None,
                          inverseY)
        sys.exit()
    
    print("Approximation started....")
    params_opt, params_cov = ila.approx(method, t_obs, m_obs, maxfev=MAXFEV)
    print("Approximation successful.")
    print()
        
    if method == "AP" or method == "WSAP":
        if params_opt[3] >= params_opt[4]:
            raise Exception("C4 must be less than C5. Aborting.")
    
    # 1-sigma uncertainties
    param_errors = np.sqrt(np.diag(params_cov))
    
    time_of_extremum, time_extr_sig, mag_of_extremum, mag_extr_sig = ila.method_result(method, params_opt, params_cov, min(t_obs), max(t_obs))
    
    utils.save_result(f"approx_result-{method}.txt",
                      method,
                      time_of_extremum, time_extr_sig,
                      mag_of_extremum, mag_extr_sig,
                      params_opt, param_errors)
    
    t_min = min(t_obs)
    t_max = max(t_obs)
    t_array = np.linspace(t_min, t_max, 10000)
    if method == "AP":
        C1, C2, C3, C4, C5 = params_opt
        y_array_fit = ila.f_AP_a(t_array, C1, C2, C3, C4, C5)
        y_array_fit_at_points = ila.f_AP_a(t_obs, C1, C2, C3, C4, C5)
    elif method == "WSAP":
        C1, C2, C3, C4, C5, C6 = params_opt
        y_array_fit = ila.f_WSAP_a(t_array, C1, C2, C3, C4, C5, C6)
        y_array_fit_at_points = ila.f_WSAP_a(t_obs, C1, C2, C3, C4, C5, C6)
    elif method == "WSAPA":
        C1, C2, C3, C4, C5, C6, C7 = params_opt
        y_array_fit = ila.f_WSAPA_a(t_array, C1, C2, C3, C4, C5, C6, C7)
        y_array_fit_at_points = ila.f_WSAPA_a(t_obs, C1, C2, C3, C4, C5, C6, C7)
    else: #method == "A"
       C1, C2, C3, C4 = params_opt
       C5 = None
       y_array_fit = ila.f_A_a(t_array, C1, C2, C3, C4)
       y_array_fit_at_points = ila.f_A_a(t_obs, C1, C2, C3, C4)

    utils.save_approx(f"approx_data-{method}.txt",
                      t_obs, y_array_fit_at_points, m_obs)
       
    print()
    sigma = np.sqrt(np.sum((m_obs - y_array_fit_at_points)**2) / (len(m_obs) - len(params_opt)))
    print("sigma        = ", sigma)
    sigma = np.sqrt(np.sum((m_obs - y_array_fit_at_points)**2) * len(params_opt) / len(m_obs) / (len(m_obs) - len(params_opt)))    
    print("sigma_m[x_c] = ", sigma, " # r.m.s. accuracy of the fit")

    utils.plot_result(t_obs, m_obs, 
                      t_array, y_array_fit, 
                      C4, C5, 
                      time_of_extremum, time_extr_sig,
                      mag_of_extremum, mag_extr_sig,
                      inverseY)

###############################################################################

def main():
    args = utils.parse_args(DESCRIPTION)
    process_data(args.filename, args.method.upper(), not args.non_inverseY)

if __name__ == "__main__":
    if DEBUG:
        main()
    else:    
        try:
            main()
        except Exception as e:
            print(f"Fatal Error: {e}.")
        finally:
            print()
            sys.exit()
