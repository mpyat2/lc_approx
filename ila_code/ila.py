import numpy as np
from scipy.optimize import curve_fit

# Andrych, Kateryna D.; Andronov, Ivan L.; Chinarova, Lidia L.
# MAVKA: Program of Statistically Optimal Determination of Phenomenological 
# Parameters of Extrema. Parabolic Spline Algorithm And Analysis of
# Variability of The Semi-Regular Star Z UMa
# Journal of Physical Studies, Vol. 24, No. 1, Article 1902 [10 pages] (2020)
# Bibcode: 2020JPhSt..24.1902A
# DOI: 10.30970/jps.24.1902, 10.48550/arXiv.1912.07677

def f_AP(t, C1, C2, C3, C4, C5):
    D = (C5 - C4) / 2; v = t - (C5 + C4) / 2
    if D <= 0.0:
        return np.inf
    if t < C4:
        return C1 + C2 * (-2 * v - D) * D + C3 * v
    elif C4 <= t <= C5:
        return C1 + C2 * v * v + C3 * v
    else:
        return C1 + C2 * (2 * v - D) * D + C3 * v
        
def f_AP_a(t_a, C1, C2, C3, C4, C5):
    return list(map(lambda t: f_AP(t, C1, C2, C3, C4, C5), t_a))

###############################################################################

def f_WSAP(t, C1, C2, C3, C4, C5):
    D = (C5 - C4) / 2; v = t - (C5 + C4) / 2
    if D <= 0.0:
        return np.inf
    if t < C4:
        return C1 + C2 * (-2 * v - D) * D + C3 * abs(t - C4) ** 1.5
    if C4 <= t <= C5:
        return C1 + C2 * v * v
    else:
        return C1 + C2 * (2 * v - D) * D + C3 * abs(t - C5) ** 1.5
        
def f_WSAP_a(t_a, C1, C2, C3, C4, C5):
    return list(map(lambda t: f_WSAP(t, C1, C2, C3, C4, C5), t_a))

###############################################################################

def f_WSL(t, C1, C2, C3, C4, C5):
    if C5 <= C4:
        return np.inf
    if C4 <= t <= C5:
        return C1
    else:
        if t < C4:
            x = C4 - t
        else:
            x = t - C5
        return C1 + C2 * abs(x) ** 1.5 + C3 * abs(x) ** 3.5

def f_WSL_a(t_a, C1, C2, C3, C4, C5):
    return list(map(lambda t: f_WSL(t, C1, C2, C3, C4, C5), t_a))

###############################################################################

def f_A(t, C1, C2, C3, C4):
    if t < C4:
        return C1 + C2 * (t - C4)
    else:
        return C1 - C3 * (t - C4)
        
def f_A_a(t_a, C1, C2, C3, C4):
    return list(map(lambda t: f_A(t, C1, C2, C3, C4), t_a))

###############################################################################

#FTOL=1e-8
#XTOL=1e-8
#GTOL=1e-8

def approx(method, t_obs, m_obs, maxfev=12000):
    if method != "AP" and method != "WSAP" and method != "WSL" and method != "A":
        raise Exception("Only AP, WSAP, WSL, and A methods are supported.")

    param_warning = None
    
    mean_t = np.mean(t_obs)
    t_obs = t_obs - mean_t

    t_min = min(t_obs)
    t_max = max(t_obs)

    #Initial values (AP, WSAP)
    C1 = np.mean(m_obs)
    C2 = 0.0
    C3 = 0.0
    C4 = t_min + (t_max - t_min) / 3.0
    C5 = t_max - (t_max - t_min) / 3.0
    
    if method == "AP" or method == "WSAP":
        if method == "AP":
            func = f_AP_a
        else:
            func = f_WSAP_a
            
        params_opt, params_cov = curve_fit(func, t_obs, m_obs, p0=[C1, C2, C3, C4, C5],
                                           maxfev=maxfev, 
                                           #ftol=FTOL, xtol=XTOL, gtol=GTOL
                                           )
        #print(params_cov)
        #print(params_opt)
        C1, C2, C3, C4, C5 = params_opt
        if C4 < t_min or C4 > t_max or C5 < t_min or C5 > t_max:
            param_warning = "Bad C4 or C5 or both. Trying again using the previous values as the starting point."
            params_opt, params_cov = curve_fit(func, t_obs, m_obs, p0=[C1, C2, C3, C4, C5],
                                               maxfev=maxfev, 
                                               #ftol=FTOL, xtol=XTOL, gtol=GTOL
                                               )
            #print(params_cov)
            if C4 < t_min or C4 > t_max or C5 < t_min or C5 > t_max:
                param_warning = "Second run: Bad C4 or C5 or both. Try another method."
            
        params_opt[3] = params_opt[3] + mean_t #C4
        params_opt[4] = params_opt[4] + mean_t #C5
    elif method == "WSL":
        params_opt, params_cov = curve_fit(f_WSL_a, t_obs, m_obs, p0=[C1, C2, C3, C4, C5],
                                           maxfev=maxfev, 
                                           #ftol=FTOL, xtol=XTOL, gtol=GTOL
                                           )
        params_opt[3] = params_opt[3] + mean_t #C4
        params_opt[4] = params_opt[4] + mean_t #C5
    elif method == "A":
        C4  = (t_max + t_min) / 2.0
        params_opt, params_cov = curve_fit(f_A_a, t_obs, m_obs, p0=[C1, C2, C3, C4],
                                           maxfev=maxfev, 
                                           #ftol=FTOL, xtol=XTOL, gtol=GTOL
                                           )
        params_opt[3] = params_opt[3] + mean_t #C4
    else:
        raise Exception(f"Unknown mapproximation ethod: {method}")
    return params_opt, params_cov, param_warning

def method_result(method, params_opt, params_cov, t_min, t_max):
    warning = None
    time_of_extremum = np.nan
    time_extr_sig = np.nan
    mag_of_extremum = np.nan
    mag_extr_sig = np.nan
    eclipse_duration = np.nan
    eclipse_sig = np.nan
    if method == "AP" or method == "WSAP":
        C1, C2, C3, C4, C5 = params_opt
        cov_matrix = params_cov.copy()
        if method == "AP":
            time_of_extremum = (C4 + C5) / 2.0 - C3 / C2 / 2.0
        else:
            time_of_extremum = (C4 + C5) / 2.0
        # we suppose that the extremun must be in the parabolic part    
        if C4 <= time_of_extremum <= C5:
            if method == "AP":
                time_of_extremum = (C4 + C5) / 2.0 - C3 / C2 / 2.0
                mag_of_extremum = C1 - (C3 * C3) / (4 * C2)
                J_t = np.array([0.0, 0.5 * C3 / (C2 * C2), -0.5 / C2, 0.5, 0.5])
                J_m = np.array([1.0, (C3 * C3) / (4 * C2 * C2), -C3 / (2 * C2), 0.0, 0.0])
                time_extr_var = J_t @ cov_matrix @ J_t.T
                time_extr_sig = np.sqrt(time_extr_var)
                mag_extr_var = J_m @ cov_matrix @ J_m.T
                mag_extr_sig = np.sqrt(mag_extr_var)
            else: #WSAP
                time_of_extremum = (C4 + C5) / 2.0
                mag_of_extremum = C1
                J_t = np.array([0.0, 0.0, 0.0, 0.5, 0.5])
                J_m = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
                time_extr_var = J_t @ cov_matrix @ J_t.T
                time_extr_sig = np.sqrt(time_extr_var)
                mag_extr_var = J_m @ cov_matrix @ J_m.T
                mag_extr_sig = np.sqrt(mag_extr_var)
            if abs(C5 - C4) < time_extr_sig:
                # Parabolic part is shorter than the uncertainty.
                # It seems the method is not suitable.
                warning = "The parabolic part is shorter than the uncertainty! Try another method."
                #time_extr_sig = np.nan
                #mag_extr_sig = np.nan
        else:
            warning = "The extremum is out of the parabolic part! Try another method."
            time_of_extremum = np.nan
            time_extr_sig = np.nan
            mag_of_extremum = np.nan
            mag_extr_sig = np.nan
    elif method == "WSL":
        C1, C2, C3, C4, C5 = params_opt
        cov_matrix = params_cov
        J_t = np.array([0.0, 0.0, 0.0, 0.5, 0.5])
        time_of_extremum = (C4 + C5) / 2.0
        time_extr_var = J_t @ cov_matrix @ J_t.T
        time_extr_sig = np.sqrt(time_extr_var)
        mag_of_extremum = C1
        mag_extr_sig = np.sqrt(cov_matrix[0, 0])  #Uncertainty of C1
        eclipse_duration = C5 - C4
        J_t = np.array([0.0, 0.0, 0.0, -1.0, 1.0])
        eclipse_sig = np.sqrt(J_t @ cov_matrix @ J_t.T)
        if abs(C5 - C4) < time_extr_sig:
            # Flat part is shorter than the uncertainty.
            # It seems the method is not suitable.
            warning = "The flat part is shorter than the uncertainty! Try another method."
    elif method == "A":
        C1, C2, C3, C4 = params_opt
        cov_matrix = params_cov
        time_of_extremum = C4
        time_extr_sig = np.sqrt(cov_matrix[3, 3]) #Uncertainty of C4
        mag_of_extremum = C1
        mag_extr_sig = np.sqrt(cov_matrix[0, 0])  #Uncertainty of C1
    else:
        raise Exception(f"Unsupported method: {method}")

    return time_of_extremum, time_extr_sig, mag_of_extremum, mag_extr_sig, eclipse_duration, eclipse_sig, warning

