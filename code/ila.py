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
    if t < C4:
        return C1 + C2 * (-2 * v - D) * D + C3 * v
    elif C4 <= t <= C5:
        return C1 + C2 * v * v + C3 * v
    else:
        return C1 + C2 * (2 * v - D) * D + C3 * v
        
def f_AP_a(t_a, C1, C2, C3, C4, C5):
    return list(map(lambda t: f_AP(t, C1, C2, C3, C4, C5), t_a))

###############################################################################

def f_WSAP(t, C1, C2, C3, C4, C5, C6):
    D = (C5 - C4) / 2; v = t - (C5 + C4) / 2
    if t < C4:
        return C1 + C2 * (-2 * v - D) * D + C3 * v + C6 * abs(t - C4) ** 1.5
    elif C4 <= t <= C5:
        return C1 + C2 * v * v + C3 * v
    else:
        return C1 + C2 * (2 * v - D) * D + C3 * v + C6 * abs(t - C5) ** 1.5
        
def f_WSAP_a(t_a, C1, C2, C3, C4, C5, C6):
    return list(map(lambda t: f_WSAP(t, C1, C2, C3, C4, C5, C6), t_a))

###############################################################################

def approx(method, t_obs, m_obs, maxfev=12000):
    
    if method != "AP" and method != "WSAP":
        raise Exception("Only AP and WSAP methods are supported.")
    
    mean_t = np.mean(t_obs)
    t_obs = t_obs - mean_t

    t_min = min(t_obs)
    t_max = max(t_obs)

    #Initial values
    C1 = np.mean(m_obs)
    C2 = 0.0
    C3 = 0.0
    C4 = t_min + (t_max - t_min) / 3.0
    C5 = t_max - (t_max - t_min) / 3.0
    C6 = 0.0

    if method == "AP":
        params_opt, params_cov = curve_fit(f_AP_a, t_obs, m_obs, p0=[C1, C2, C3, C4, C5], maxfev=maxfev)
    else:
        params_opt, params_cov = curve_fit(f_WSAP_a, t_obs, m_obs, p0=[C1, C2, C3, C4, C5, C6], maxfev=maxfev)
    
    params_opt[3] = params_opt[3] + mean_t #C4
    params_opt[4] = params_opt[4] + mean_t #C5

    return params_opt, params_cov

def method_result(params_opt, params_cov):

    if len(params_opt) == 6:
        # WSAP
        C1, C2, C3, C4, C5, C6 = params_opt
        cov_matrix = params_cov[:5, :5]
    else:
        # AP
        C1, C2, C3, C4, C5 = params_opt
        cov_matrix = params_cov.copy()

    time_of_extremum = (C4 + C5) / 2.0 - C3 / C2 / 2.0

    # we suppose that the extremun must be in the parabolic part    
    if C4 <= time_of_extremum <= C5:
        J_t = np.array([0.0, 0.5 * C3 / (C2 * C2), -0.5 / C2, 0.5, 0.5])
        time_extr_var = J_t @ cov_matrix @ J_t.T
        time_extr_sig = np.sqrt(time_extr_var)

        mag_of_extremum = C1 - (C3**2) / (4*C2)
        J_m = np.array([1.0, (C3 * C3) / (4 * C2 * C2), -C3 / (2 * C2), 0.0, 0.0])

        mag_extr_var = J_m @ cov_matrix @ J_m.T
        mag_extr_sig = np.sqrt(mag_extr_var)
    else:
        time_of_extremum = np.nan
        time_extr_sig = np.nan
        mag_of_extremum = np.nan
        mag_extr_sig = np.nan

    return time_of_extremum, time_extr_sig, mag_of_extremum, mag_extr_sig

