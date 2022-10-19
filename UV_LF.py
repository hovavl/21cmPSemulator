import json
import py21cmfast as p21c
import copy
import numpy as np
import matplotlib.pyplot as plt

# read UV luminosity function data
with open('/gpfs0/elyk/users/hovavl/jobs/21cm_mcmc_job/UV_LU_data.json', 'r') as openfile:
    # Reading from json file
    UV_LU_data = json.load(openfile)

user_params = {"DIM": 512, "HII_DIM": 128, "BOX_LEN": 256, "N_THREADS": 1, 'USE_RELATIVE_VELOCITIES': False,
               "POWER_SPECTRUM": 0}
flag_options = {"USE_MASS_DEPENDENT_ZETA": True, "USE_CMB_HEATING": False, "USE_LYA_HEATING": False,
                'USE_TS_FLUCT': True,
                'USE_MINI_HALOS': False, "INHOMO_RECO": True}


def predict_luminosity(theta):
    tmp = copy.deepcopy(theta)
    F_STAR10,ALPHA_STAR, F_ESC10,ALPHA_ESC, M_TURN,t_STAR,L_X , NU_X_THRESH, X_RAY_SPEC_INDEX = tmp
    
    astro_params = {'F_STAR10': F_STAR10, 'F_ESC10': F_ESC10, 'L_X': L_X, 'M_TURN': M_TURN,
              'NU_X_THRESH': NU_X_THRESH, 'ALPHA_STAR':ALPHA_STAR, 'ALPHA_ESC':ALPHA_ESC,
               't_STAR':t_STAR, 'X_RAY_SPEC_INDEX':X_RAY_SPEC_INDEX}
    redshifts = [4, 5, 6, 7, 8, 10]
    Lf = p21c.compute_luminosity_function(
        redshifts=redshifts,
        user_params=user_params,
        astro_params=astro_params,
        flag_options=flag_options,
        nbins=50,
        component=0
    )
    interp_lum = []
    for i, m_uv in enumerate(Lf[0]):
        fp = np.power(10, np.array(Lf[2][i]))
        log_fp = np.isnan(fp)
        
        fp =  np.flip(fp[~log_fp])
        m_uv = np.flip(m_uv[~log_fp])
        x_eval = UV_LU_data[str(redshifts[i])]['M']
        
        y = np.interp(x_eval , m_uv , fp)
        interp_lum += [y]
        #plt.plot(x_eval,y, color = 'b', ls = '--', label = 'interpolated')
        #plt.plot(m_uv[30:52], fp[30:52] , color = 'r', label = 'real')
        #plt.legend()
        #plt.savefig('uv_luminosity_function')
        #break
    return interp_lum






