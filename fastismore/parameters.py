import numpy as np
import scipy

__all__ = ['not_param', 'add_extra', 'param_to_label', 'param_to_latex']

not_param = [
    'like',
    'old_like',
    'delta_loglike',
    'new_like',
    'new_prior',
    'new_post',
    'prior',
    'post',
    '2pt_like',
    'old_weight',
    'weight',
    'old_post',
    'log_weight',
    'chi2',
    'chi2_joint'
]

label_dict = {
    'cosmological_parameters--tau':  r'\tau',
    'cosmological_parameters--w':  r'w_0',
    'cosmological_parameters--wa':  r'w_a',
    'cosmological_parameters--wp':  r'w_p',
    'cosmological_parameters--w0_fld':  r'w_{GDM}',
    'cosmological_parameters--cs2_fld': r'c_s^2',
    'cosmological_parameters--log_cs2': r'log(c_s^2)',
    'cosmological_parameters--omega_m': r'\Omega_{\rm m}',
    'cosmological_parameters--omega_c': r'\Omega_c',
    'cosmological_parameters--ommh2': r'\Omega_m h^2',
    'cosmological_parameters--ombh2': r'\Omega_b h^2',
    'cosmological_parameters--omch2': r'\Omega_c h^2',
    'cosmological_parameters--h0':    r'h',
    'cosmological_parameters--omega_b': r'\Omega_b',
    'cosmological_parameters--n_s':  r'n_s',
    'cosmological_parameters--a_s':  r'A_s',
    'cosmological_parameters--omnuh2':  r'\Omega_{\nu}',
    'cosmological_parameters--sigma_8': r'\sigma_8',
    'cosmological_parameters--sigma_12': r'\sigma_12',
    'cosmological_parameters--s8': r'S_8',
    'cosmological_parameters--massive_nu': r'\nu_\text{massive}',
    'cosmological_parameters--massless_nu': r'\nu_\text{massless}',
    'cosmological_parameters--omega_k': r'\Omega_k',
    'cosmological_parameters--yhe': r'Y_\text{He}',
    'cosmological_parameters--mnu': r'm_\nu',
    'cosmological_parameters--meffsterile': r'm_{\rm eff}',
    'cosmological_parameters--s8_07': r'\sigma_8(\Omega_{\rm m}/0.3)^{0.7}',
    'cosmological_parameters--xi_interaction': r'\xi',
    'halo_model_parameters--detg_beta': r'\beta',
    'rescale_pk_fz--alpha_cmb':r'$\alpha_{\rm CMB}$',\
    'rescale_pk_fz--sigma_8_0':r'$\sigma_8^0$',\
    'rescale_pk_fz--sigma_8_1':r'$\sigma_8^1$',\
    'rescale_pk_fz--sigma_8_2':r'$\sigma_8^2$',\
    'rescale_pk_fz--sigma_8_3':r'$\sigma_8^3$',\
    'rescale_pk_fz--sigma_8_4':r'$\sigma_8^4$',\
    'rescale_pk_fz--sigma_8_5':r'$\sigma_8^5$',\
    'rescale_pk_fz--sigma_8_6':r'$\sigma_8^6$',\
    'rescale_pk_fz--sigma_8_7':r'$\sigma_8^7$',\
    'rescale_pk_fz--sigma_8_8':r'$\sigma_8^8$',\
    'rescale_pk_fz--sigma_8_9':r'$\sigma_8^9$',\
    'rescale_pk_fz--sigma_8_10':r'$\sigma_8^{10}$',\
    'rescale_pk_fz--sigma_8_11':r'$\sigma_8^{11}$',\
    'rescale_pk_fz--sigma_8_12':r'$\sigma_8^{12}$',\
    'rescale_pk_fz--sigma_8_13':r'$\sigma_8^{13}$',\
    'rescale_pk_fz--sigma_8_14':r'$\sigma_8^{14}$',\
    'rescale_pk_fz--sigma_8_15':r'$\sigma_8^{15}$',\
    'rescale_pk_fz--sigma_8_16':r'$\sigma_8^{16}$',\
    
    'intrinsic_alignment_parameters--a': r'A_{IA}',
    'intrinsic_alignment_parameters--alpha': r'\alpha_{IA}',
    'bin_bias--b1': r'b_1',
    'bin_bias--b2': r'b_2',
    'bin_bias--b3': r'b_3',
    'bin_bias--b4': r'b_4',
    'bin_bias--b5': r'b_5',
    'shear_calibration_parameters--m1': r'm_1',
    'shear_calibration_parameters--m2': r'm_2',
    'shear_calibration_parameters--m3': r'm_3',
    'shear_calibration_parameters--m4': r'm_4',
    'lens_photoz_errors--bias_1': r'z^l_1',
    'lens_photoz_errors--bias_2': r'z^l_2',
    'lens_photoz_errors--bias_3': r'z^l_3',
    'lens_photoz_errors--bias_4': r'z^l_4',
    'lens_photoz_errors--bias_5': r'z^l_5',
    'wl_photoz_errors--bias_1': r'z^s_1',
    'wl_photoz_errors--bias_2': r'z^s_2',
    'wl_photoz_errors--bias_3': r'z^s_3',
    'wl_photoz_errors--bias_4': r'z^s_4',

    'bias_lens--b1': r'b_1',
    'bias_lens--b2': r'b_2',
    'bias_lens--b3': r'b_3',
    'bias_lens--b4': r'b_4',
    'bias_lens--b5': r'b_5',
    'intrinsic_alignment_parameters--a1': r'A_1',
    'intrinsic_alignment_parameters--a2': r'A_2',
    'intrinsic_alignment_parameters--alpha1': r'\alpha_1',
    'intrinsic_alignment_parameters--alpha2': r'\alpha_2',
    'intrinsic_alignment_parameters--bias_ta': r'b_ta',
    'intrinsic_alignment_parameters--z_piv': r'z_\text{piv}',
    # IA free amplitude per bin
    'intrinsic_alignment_parameters--a_1': r'A_{z_1}',
    'intrinsic_alignment_parameters--a_2': r'A_{z_2}',
    'intrinsic_alignment_parameters--a_3': r'A_{z_3}',
    'intrinsic_alignment_parameters--a_4': r'A_{z_4}',

    'mag_alpha_lens--alpha_1': r'\alpha_\text{lens}^1',
    'mag_alpha_lens--alpha_2': r'\alpha_\text{lens}^2',
    'mag_alpha_lens--alpha_3': r'\alpha_\text{lens}^3',
    'mag_alpha_lens--alpha_4': r'\alpha_\text{lens}^4',
    'mag_alpha_lens--alpha_5': r'\alpha_\text{lens}^5',
    
    'npg_parameters--a1': 'A_1',
    'npg_parameters--a2': 'A_2',
    'npg_parameters--a3': 'A_3',
    'npg_parameters--a4': 'A_4',
    'modified_gravity--sigma0': r"\Sigma_0",
    'modified_gravity--mu0': r"\mu_0",
    'modified_gravity--p1': "p_1",
    'modified_gravity--s8sigma0': r"S_8 \Sigma_0",
}

def calc_rd(h, Omega_m, obh2):
    h_fid = h
    hubble_to_Mpc = 2997.92458
    hubble_distance_fid = hubble_to_Mpc / h_fid # in Mpc
    
    Tcmb = 2.7260 * 8.6173303e-5 # in eV
    Tnu = (4./11.)**(1./3.) * Tcmb
    H0 = h_fid * 2.13311968e-33   # in eV
    Mplanck = 1.22089007e28      # in eV
    crit_dens = 3./8./np.pi * (H0 * Mplanck)**2 # in eV^4
    Omega_g = np.pi**2 / 15 * Tcmb**4 / crit_dens
    Neff = 3.044
    Omega_nu_rel = 7./8. * (4./11.)**(4./3.) * Neff * Omega_g # relativistic neutrinos
    
    Omega_de = 1 - Omega_m - Omega_g - Omega_nu_rel
    inv_efunc = lambda z: 1./np.sqrt(Omega_m*(1+z)**3 + (Omega_g + Omega_nu_rel)*(1+z)**4 + Omega_de)
    
    cs = lambda obh2: np.vectorize(lambda z: 1/np.sqrt(3 * (1 + 3./4. * obh2 / h_fid**2 / Omega_g / (1+z))))

    zdrag = 1060
    
    r_dec = lambda z_dec: (lambda obh2, Omega_m: hubble_distance_fid*scipy.integrate.quad(lambda a: inv_efunc(1/a - 1) * cs(obh2)(1/a - 1) / a**2, 0, 1/(1+z_dec))[0])
    rdrag = r_dec(zdrag)
    return rdrag(obh2,Omega_m)

def add_s8(data, extra=None, weights=None):
    """
    Add S8 to the data dictionary if it exists.
    """
    if extra is not None:
        data.update(extra)
    
    if 'cosmological_parameters--sigma_8' in data.keys():
        data['cosmological_parameters--s8'] = \
            data['cosmological_parameters--sigma_8']*(data['cosmological_parameters--omega_m']/0.3)**0.5
    return data
def add_extra(data, extra=None, weights=None):
    if extra is not None:
        data.update(extra)
    if 'cosmological_parameters--sigma_8' in data.keys():
        data['cosmological_parameters--s8'] = \
            data['cosmological_parameters--sigma_8']*(data['cosmological_parameters--omega_m']/0.3)**0.5
        data['cosmological_parameters--s8_07'] = \
            data['cosmological_parameters--sigma_8']*(data['cosmological_parameters--omega_m']/0.3)**0.7

    if 'cosmological_parameters--h0' in data.keys():
        data['cosmological_parameters--ommh2'] = \
            data['cosmological_parameters--omega_m']*data['cosmological_parameters--h0']**2
        

    if 'cosmological_parameters--omega_b' in data.keys():
        data['cosmological_parameters--ombh2'] = \
            data['cosmological_parameters--omega_b']*data['cosmological_parameters--h0']**2
        data['cosmological_parameters--rdh'] = np.array([calc_rd(h, obh2, om) for h,obh2,om in zip(
            data['cosmological_parameters--h0'],
            data['cosmological_parameters--ombh2'],
            data['cosmological_parameters--omega_m']
        )])*data['cosmological_parameters--h0']

    if 'cosmological_parameters--ommh2' in data.keys() and 'cosmological_parameters--ombh2' in data.keys():
        data['cosmological_parameters--omch2'] = \
            data['cosmological_parameters--ommh2'] - data['cosmological_parameters--ombh2']

    if 'cosmological_parameters--omega_b' in data.keys():
        data['cosmological_parameters--omega_c'] = \
            data['cosmological_parameters--omega_m'] - data['cosmological_parameters--omega_b']
        
    if 'modified_gravity--sigma0' in data.keys():
        data['modified_gravity--s8sigma0'] = data['modified_gravity--sigma0']*data['cosmological_parameters--s8']
    
    if 'cosmological_parameters--w' in data.keys() and 'cosmological_parameters--wa' in data.keys() and weights is not None:
        w0, wa = data['cosmological_parameters--w'], data['cosmological_parameters--wa']
        w0wa_cov = np.cov(w0, wa, aweights=weights)
        ap = 1. + w0wa_cov[0,1]/w0wa_cov[1,1]
        data['cosmological_parameters--ap'] = ap
        data['cosmological_parameters--wp'] = w0 + wa*(1. - ap)

    
    if 'rescale_pk_fz--sigma_8_0' in data.keys():
        for i in range(9):
            data[f'rescale_pk_fz--s8_{i}'] = \
                data[f'rescale_pk_fz--sigma_8_{i}']*(data['cosmological_parameters--omega_m']/0.3)**0.5


    return data

param_to_label = np.vectorize(lambda param: label_dict[param] if param in label_dict else param)
param_to_latex = np.vectorize(lambda param: '${}$'.format(label_dict[param]) if param in label_dict else param)