###############################################################################

DESCRIPTION = \
"""I.L.A. approximations
"""

###############################################################################

JD_SHIFT       = 0 #-2400000

# Try to increase this parameter (number of iterations) if the solution cannot be found
#MAXFEV         = 50000
MAXFEV         = 100000

###############################################################################

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code'))
import utils

###############################################################################

def process_data(data_file_name, method, inverseY=True):
    import ila
    #
    import numpy as np
    import pandas as pd

    data = pd.read_csv(data_file_name, 
                       comment='#', skip_blank_lines=True,
                       sep="\\s+",
                       header=0,
                       names=['time', 'mag'], 
                       dtype={'time': 'float64', 'mag': 'float64'}, 
                       usecols=['time', 'mag'])
    print("File loaded")
    
    t_obs = data['time'] + JD_SHIFT
    m_obs = data['mag']
    
    print("Approximation started....")
    params_opt, params_cov = ila.approx(method, t_obs, m_obs, maxfev=MAXFEV)
    print("Approximation successful.")
    
    print();
    
    # 1-sigma uncertainties
    param_errors = np.sqrt(np.diag(params_cov))
    
    time_of_extremum, time_extr_sig, mag_of_extremum, mag_extr_sig = ila.method_result(params_opt, params_cov)

    utils.save_result(method,
                      time_of_extremum, time_extr_sig,
                      mag_of_extremum, mag_extr_sig,
                      params_opt, param_errors)
   
    t_min = min(t_obs)
    t_max = max(t_obs)
    t_array = np.linspace(t_min, t_max, 10000)
    if method == "AP":
        C1, C2, C3, C4, C5 = params_opt
        y_array_fit = ila.f_AP_a(t_array, C1, C2, C3, C4, C5)
    else:
        C1, C2, C3, C4, C5, C6 = params_opt
        y_array_fit = ila.f_WSAP_a(t_array, C1, C2, C3, C4, C5, C6)
    
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
    try:
        main()
    except Exception as e:
        print(f"Fatal Error: {e}.")
        #input("")
    finally:
        print()
        #print("End")
        sys.exit()
