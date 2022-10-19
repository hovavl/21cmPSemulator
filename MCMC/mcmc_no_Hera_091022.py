import matplotlib.pyplot as plt
import emcee
from cosmopower import cosmopower_NN
import numpy as np
from schwimmbad import MPIPool
import mpi4py
from scipy import interpolate
from scipy import special
import pickle
import warnings
import copy
import hera_pspec as hp
import model_for_xh
import model_for_tau
import corner
import json
import UV_LF
warnings.filterwarnings('ignore', module='hera_sim')
warnings.filterwarnings('ignore', category=RuntimeWarning)

# HERA-Stack

warnings.filterwarnings('ignore')


def interp_Wcdf(W, k, lower_perc=0.16, upper_perc=0.84):
    """
    Construct CDF from normalized window function and interpolate
    to get k at window func's 16, 50 & 84 percentile.

    Parameters
    ----------
    W : ndarray
        Normalized window function of shape (Nbandpowers, Nk)
    k : ndarray
        vector of k modes of shape (Nk,)

    Returns
    -------
    ndarray
        k of WF's 50th percentile
    ndarray
        dk of WF's 16th (default) percentile from median
    ndarray
        dk of WF's 84th (default) percentile from median
    """
    # get cdf: take sum of only abs(W)
    W = np.abs(W)
    Wcdf = np.array([np.sum(W[:, :i + 1].real, axis=1) for i in range(W.shape[1] - 1)]).T

    # get shifted k such that a symmetric window has 50th perc at max value
    kshift = k[:-1] + np.diff(k) / 2

    # interpolate each mode (xvalues are different for each mode!)
    med, low_err, hi_err = [], [], []
    for i, w in enumerate(Wcdf):
        interp = interpolate.interp1d(w, kshift, kind='linear', fill_value='extrapolate')
        m = interp(0.5)  # 50th percentile
        # m = k[np.argmax(W[i])]  # mode
        med.append(m)
        low_err.append(m - interp(lower_perc))
        hi_err.append(interp(upper_perc) - m)

    return np.array(med), np.array(low_err), np.array(hi_err)


# each field is an independent file, containing Band 1 and Band 2
# just load field 1 for now
uvp = hp.UVPSpec()
field = 1
uvp.read_hdf5('/gpfs0/elyk/users/hovavl/jobs/21cm_mcmc_job/data_for_mcmc/ps_files/pspec_h1c_idr2_field1.h5')

# print the two available keys
band1_key, band2_key = uvp.get_all_keys()
keys = [band1_key, band2_key]

# get data
band2_dsq = uvp.get_data(band2_key)
band2_cov = uvp.get_cov(band2_key)
band2_wfn = uvp.get_window_function(band2_key)

band1_dsq = uvp.get_data(band1_key)
band1_cov = uvp.get_cov(band1_key)
band1_wfn = uvp.get_window_function(band1_key)

# extract data
spw = 0
kbins = uvp.get_kparas(spw)  # after spherical binning, k_perp=0 so k_mag = k_para

ks = slice(2, None)
xlim = (0, 2.0)
band2_err = np.sqrt(band2_cov[0].diagonal())
band1_err = np.sqrt(band1_cov[0].diagonal())
#data
y = band2_dsq[0, ks].real  # omit two first values (zeros)
y104 = band1_dsq[0, ks].real


yerr = band2_err[ks].real
yerr104 = band1_err[ks].real

x, xerr_low, xerr_hi = interp_Wcdf(band2_wfn[0], kbins, lower_perc=0.16, upper_perc=0.84)
xerr = np.array([xerr_low, xerr_hi]).T[ks]

mcmc_k_modes = kbins[ks]

smaller_2 = (mcmc_k_modes < 2)
# mcmc_k_modes[~smaller_2] = 0 # omit greater than 2
y[~smaller_2] = 0
ypos = y > 0  # only positive values
ypos104 = y104 > 0

ps_data = y[ypos]
ps_data104 = y104[ypos104]

yerr = yerr[ypos]
yerr104 = yerr104[ypos104]

xerr = xerr[ypos].T

# model constants:
TAU_MEAN = 0.0569
TAU_STD_HIGH = 0.0081  # not True
TAU_STD_LOW = 0.0066
XH_MEAN = 0.06
XH_STD = 0.05

# data points
emulator_k_modes = [0.0307812, 0.04617179, 0.06925769, 0.10388654, 0.1558298, 0.23374471,
                    0.35061706, 0.52592559, 0.78888838, 1.18333258, 1.77499887]
# luminosity function real data
with open('/gpfs0/elyk/users/hovavl/jobs/21cm_mcmc_job/UV_LU_data.json', 'r') as openfile:
    # Reading from json file
    UV_LU_data = json.load(openfile)

# restore NN

nn_ps = cosmopower_NN(restore=True,
                      restore_filename='/gpfs0/elyk/users/hovavl/jobs/21cm_mcmc_job/data_for_mcmc/models/21cm_PS_NN_z=7.9_081022_all_params',
                      )
nn_ps104 = cosmopower_NN(restore=True,
                         restore_filename='/gpfs0/elyk/users/hovavl/jobs/21cm_mcmc_job/data_for_mcmc/models/21cm_PS_NN_z=10.4_081022_all_params',
                         )

nn_tau = model_for_tau.cosmopower_NN(restore=True,
                                     restore_filename='/gpfs0/elyk/users/hovavl/jobs/21cm_mcmc_job/data_for_mcmc/models/21cm_tau_NN_081022_all_params')
nn_xH = model_for_xh.cosmopower_NN(restore=True,
                                   restore_filename='/gpfs0/elyk/users/hovavl/jobs/21cm_mcmc_job/data_for_mcmc/models/21cm_xH_NN_081022_all_params')


def culcPS(theta):
    tmp = copy.deepcopy(theta)
    F_STAR10,ALPHA_STAR, F_ESC10,ALPHA_ESC, M_TURN,t_STAR,L_X, NU_X_THRESH, X_RAY_SPEC_INDEX = tmp
    params = {'F_STAR10': [F_STAR10], 'F_ESC10': [F_ESC10], 'L_X': [L_X], 'M_TURN': [M_TURN],
              'NU_X_THRESH': [NU_X_THRESH], 'ALPHA_STAR':[ALPHA_STAR], 'ALPHA_ESC':[ALPHA_ESC],
               't_STAR':[t_STAR], 'X_RAY_SPEC_INDEX':[X_RAY_SPEC_INDEX]}
    predicted_testing_spectra = nn_ps.ten_to_predictions_np(params)

    tck = interpolate.splrep(emulator_k_modes, predicted_testing_spectra[0])
    model_ps = interpolate.splev(mcmc_k_modes, tck)

    w_mat = band2_wfn[0, 2:, 2:]
    model_ps = np.dot(w_mat, model_ps)
    return model_ps[ypos]


# calculate the power spectrum at z = 10.4
def culcPS2(theta):
    tmp = copy.deepcopy(theta)
    F_STAR10,ALPHA_STAR, F_ESC10,ALPHA_ESC, M_TURN,t_STAR,L_X ,NU_X_THRESH, X_RAY_SPEC_INDEX = tmp
    params = {'F_STAR10': [F_STAR10], 'F_ESC10': [F_ESC10], 'L_X': [L_X], 'M_TURN': [M_TURN],
              'NU_X_THRESH': [NU_X_THRESH], 'ALPHA_STAR':[ALPHA_STAR], 'ALPHA_ESC':[ALPHA_ESC],
               't_STAR':[t_STAR], 'X_RAY_SPEC_INDEX':[X_RAY_SPEC_INDEX]}
    predicted_testing_spectra = nn_ps104.ten_to_predictions_np(params)

    tck = interpolate.splrep(emulator_k_modes, predicted_testing_spectra[0])
    model_ps = interpolate.splev(mcmc_k_modes, tck)

    w_mat = band1_wfn[0, 2:, 2:]
    model_ps = np.dot(w_mat, model_ps)
    return model_ps[ypos104]


""" 
parameters for 21cm simulation
same as the simulations for the training set
"""
user_params = {"DIM": 512, "HII_DIM": 128, "BOX_LEN": 256, "N_THREADS": 1, 'USE_RELATIVE_VELOCITIES': False,
               "POWER_SPECTRUM": 4, "DO_VCB_FIT": True}
flag_options = {"USE_MASS_DEPENDENT_ZETA": True, "USE_CMB_HEATING": False,
                "USE_LYA_HEATING": False, 'USE_TS_FLUCT': True, 'USE_MINI_HALOS': False, "INHOMO_RECO": True}


# predict luminosity function
def predict_luminosity(theta):
    return UV_LF.predict_luminosity(theta)


# calc the luminosity function likelihood
def luminosity_func_lnlike(luminosity_func):
    tot_lnlike = 0
    redshifts = [4, 5, 6, 7, 8, 10]
    for i, func in enumerate(luminosity_func):
        lum_data = UV_LU_data[str(redshifts[i])]['phi_k']
        lum_err = UV_LU_data[str(redshifts[i])]['err']
        # print(f'data size: {len(lum_data)} err size: {len(lum_err)} func size: {len(func)}')

        for j, val in enumerate(lum_data):
            if val is None:
                like = 0 if func[j] <= lum_err[j] else -100
                tot_lnlike += like
            else:
                like = -(1 / 2) * (((val - func[j]) / lum_err[j]) ** 2 + np.log(2 * np.pi * lum_err[j] ** 2))
                tot_lnlike += like
    return tot_lnlike


# predict tau
def predict_tau(theta):
    tmp = copy.deepcopy(theta)
    F_STAR10,ALPHA_STAR, F_ESC10,ALPHA_ESC, M_TURN,t_STAR,L_X ,NU_X_THRESH, X_RAY_SPEC_INDEX = tmp
    params = {'F_STAR10': [F_STAR10], 'F_ESC10': [F_ESC10], 'L_X': [L_X], 'M_TURN': [M_TURN],
              'NU_X_THRESH': [NU_X_THRESH], 'ALPHA_STAR':[ALPHA_STAR], 'ALPHA_ESC':[ALPHA_ESC],
               't_STAR':[t_STAR], 'X_RAY_SPEC_INDEX':[X_RAY_SPEC_INDEX]}
    predicted_tau = nn_tau.predictions_np(params)[0]
    return predicted_tau


# predict xH
def predict_xH(theta):
    tmp = copy.deepcopy(theta)
    F_STAR10,ALPHA_STAR, F_ESC10,ALPHA_ESC, M_TURN,t_STAR,L_X ,NU_X_THRESH, X_RAY_SPEC_INDEX = tmp
    params = {'F_STAR10': [F_STAR10], 'F_ESC10': [F_ESC10], 'L_X': [L_X], 'M_TURN': [M_TURN],
              'NU_X_THRESH': [NU_X_THRESH], 'ALPHA_STAR':[ALPHA_STAR], 'ALPHA_ESC':[ALPHA_ESC],
               't_STAR':[t_STAR], 'X_RAY_SPEC_INDEX':[X_RAY_SPEC_INDEX]}
    predicted_xH = nn_xH.predictions_np(params)[0]
    return max(predicted_xH, 0)


# the model
def model(theta, k_modes=emulator_k_modes):
    return culcPS(theta)


# calculate the power spectrum at z = 10.4

def model2(theta, k_modes=emulator_k_modes):
    return culcPS2(theta)


def lnlike(theta, k_modes=emulator_k_modes, y_data=ps_data, data_err=yerr):
    # tmp = -(1 / 2) * np.sum(((y_data - model(theta)) / (data_err)) ** 2) - np.sum(
    #     np.log(np.sqrt(2 * np.pi) * data_err))
    # return tmp

    #ps_lnLike = np.sum(np.log((1 / 2) * (1 + special.erf((y_data - model(theta)) / (data_err * np.sqrt(2))))))

    #ps104_lnLike = np.sum(np.log((1 / 2) * (1 + special.erf((ps_data104 - model2(theta)) / (yerr104 * np.sqrt(2))))))

    tau = predict_tau(theta)
    if tau > TAU_MEAN:
        tau_lnLike = (-1 / 2) * (((tau - TAU_MEAN) / TAU_STD_HIGH) ** 2 + np.log(2 * np.pi * TAU_STD_HIGH ** 2))
    else:
        tau_lnLike = (-1 / 2) * (((tau - TAU_MEAN) / TAU_STD_LOW) ** 2 + np.log(2 * np.pi * TAU_STD_LOW ** 2))
    xH = predict_xH(theta)
    if xH < 0.06:
        xH_lnLike = 0
    else:
        xH_lnLike = (-1 / 2) * (((xH - XH_MEAN) / XH_STD) ** 2 + np.log(2 * np.pi * XH_STD ** 2))

    UV_lum = predict_luminosity(theta)
    luminosity_lnlike = luminosity_func_lnlike(UV_lum)
    return  tau_lnLike + xH_lnLike + luminosity_lnlike # + ps_lnLike + ps104_lnLike 
    # return -0.5 * np.sum(((y_data - model(theta)) / data_err) ** 2)


def lnprior(theta):
    # n = np.random.rand()
    # if n > 0.9:
    #     print('theta: ', theta)
    F_STAR10,ALPHA_STAR, F_ESC10,ALPHA_ESC, M_TURN,t_STAR,L_X ,NU_X_THRESH, X_RAY_SPEC_INDEX = theta
    
    if (-3.0 <= F_STAR10 <= 0.0 and -3.0 <= F_ESC10 <= 0.0 and 38 <= L_X <= 42 and 8 <= M_TURN <= 10 
    and -0.5 <=ALPHA_STAR <= 1 and -1 <= ALPHA_ESC <= 0.5 and 0 <= t_STAR <= 1 and -1 <= X_RAY_SPEC_INDEX <= 3
            and 100 <= NU_X_THRESH <= 1500):
        return 0.0
    # if (-1.62 <= F_STAR10 <= -1.04 and -1.47 <= F_ESC10 <= -0.52 and 39.29 <= L_X <= 41.52 and 8.2 <= M_TURN <= 9.17 and 300 <= NU_X_THRESH <=1210):
    #     return 0.0
    return -np.inf


def lnprob(theta, k_modes=emulator_k_modes, y_data=ps_data, data_err=yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    val = lnlike(theta, k_modes, y_data, data_err)
    return lp + val  # recall if lp not -inf, its 0, so this just returns likelihood


def GRforParameter(sampMatrix):
    s = np.array(sampMatrix)
    meanArr = []
    varArr = []
    n = s.shape[0]
    for samp in s:
        meanArr += [np.mean(samp)]
        varArr += [np.std(samp) ** 2]
    a = np.std(meanArr) ** 2
    b = np.mean(varArr)
    return np.sqrt((1 - 1 / n) + a / (b * n))


def main(p0, nwalkers, niter, ndim, lnprob, data):
    with MPIPool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)

        print("Running burn-in...")
        p0, _, _,_ = sampler.run_mcmc(p0, 5000, progress=True)
        sampler.reset()
        flag = True
        count = 0
        while (flag):
            print("Running production...")
            pos, prob, state,_ = sampler.run_mcmc(p0, niter, progress=True)
            samples = sampler.get_chain()
            print('shape of samples: ', samples.shape)

            GR = []
            for i in range(samples.shape[2]):
                GR += [GRforParameter(samples[:, :, i])]
                tmp = np.abs((1 - np.array(GR) < 10 ** (-5)))
            count += 4000
            print('position: ', pos, 'GR: ', GR, '\nnum of iterations: ', count)
            break
            if np.all(tmp):
                flag = False
            else:
                p0 = pos
        return sampler, pos, prob, state


data = (mcmc_k_modes, ps_data, yerr)
nwalkers = 24
niter = 60000
initial = np.array([-1.24,0.5,-1.11, 0.02,8.59, 0.64, 40.64, 720, 0.8])  # best guesses
ndim = len(initial)
p0 = [np.array(initial) + 1e-1 * np.random.randn(ndim) for i in range(nwalkers)]
sampler, pos, prob, state = main(p0, nwalkers, niter, ndim, lnprob, data)
samples = sampler.get_chain()


flat_samples = sampler.chain[:, :, :].reshape((-1, ndim))
pickle.dump(flat_samples, open('MCMC_results_091022_no_hera.pk', 'wb'))

print(flat_samples.shape)
plt.ion()
labels = [r'$\log_{10}f_{\ast,10}$',r'$\alpha_{\ast}$', r'$\log_{10}f_{{\rm esc},10}$',r'$\alpha_{\rm esc}$',
          r'$\log_{10}[M_{\rm turn}/M_{\odot}]$',r'$t_{\ast}$', 
          r'$\log_{10}\frac{L_{\rm X<2 , keV/SFR}}{\rm erg\, s^{-1}\,M_{\odot}^{-1}\,yr}$',
          r'$E_0/{\rm keV}$', r'$\alpha_{X}$']
fig = corner.corner(flat_samples, show_titles=True, labels=labels, plot_datapoints=True,
                    quantiles=[0.16, 0.5, 0.84])
plt.savefig('mcmc_no_hera_091022.png')
