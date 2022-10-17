import json
import py21cmfast as p21c
import copy
import numpy as np

# read UV luminosity function data
with open('/Users/hovavlazare/PycharmProjects/21CM Project/UV_LU_data.json', 'r') as openfile:
    # Reading from json file
    UV_LU_data = json.load(openfile)

user_params = {"DIM": 512, "HII_DIM": 128, "BOX_LEN": 256, "N_THREADS": 1, 'USE_RELATIVE_VELOCITIES': False,
               "POWER_SPECTRUM": 4, "DO_VCB_FIT": True}
flag_options = {"USE_MASS_DEPENDENT_ZETA": True, "USE_CMB_HEATING": False, "USE_LYA_HEATING": False,
                'USE_TS_FLUCT': True,
                'USE_MINI_HALOS': False, "INHOMO_RECO": True}


def predict_luminosity(theta):
    tmp = copy.deepcopy(theta)
    F_STAR10, F_ESC10, L_X, M_TURN, NU_X_THRESH = tmp
    astro_params = {'F_STAR10': [F_STAR10], 'F_ESC10': [F_ESC10], 'L_X': [L_X], 'M_TURN': [M_TURN],
                    'NU_X_THRESH': [NU_X_THRESH], 'ALPHA_STAR': 0.08, 'ALPHA_ESC': -0.2, 't_STAR': 0.5}
    redshifts = [4, 5, 6, 7, 8, 10]
    Lf = p21c.compute_luminosity_function(
        redshifts=redshifts,
        user_params=user_params,
        astro_params=astro_params,
        flag_options=flag_options,
        nbins=30,
        component=0
    )
    interp_lum = []
    for i, m_uv in Lf[0]:
        fp = np.power(Lf[2][i], 10)
        x_eval = UV_LU_data[redshifts[i]]['M']
        y = np.interp(x_eval , m_uv , fp)
        interp_lum += y
    return interp_lum






