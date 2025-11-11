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
    'cosmological_parameters--sigma_12': r'\sigma_{12}',
    'cosmological_parameters--s8': r'S_8',
    'cosmological_parameters--massive_nu': r'\nu_\text{massive}',
    'cosmological_parameters--massless_nu': r'\nu_\text{massless}',
    'cosmological_parameters--omega_k': r'\Omega_k',
    'cosmological_parameters--yhe': r'Y_\text{He}',
    'cosmological_parameters--mnu': r'm_\nu',
    'cosmological_parameters--s8_07': r'\sigma_8(\Omega_{\rm m}/0.3)^{0.7}',
    'cosmological_parameters--xi_interaction': r'\xi',
     'cosmological_parameters--a_s_1e9': '10^9 A_s',
     'cosmological_parameters--s8_07': 'S_8^{0.7}',
     'cosmological_parameters--rdh': 'r_d h',

    'intrinsic_alignment_parameters--a': r'A_{IA}',
    'intrinsic_alignment_parameters--alpha': r'\alpha_{IA}',
    'bin_bias--b1': r'b_1',
    'bin_bias--b2': r'b_2',
    'bin_bias--b3': r'b_3',
    'bin_bias--b4': r'b_4',
    'bin_bias--b5': r'b_5',
    'bin_bias--b6': r'b_6',
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
    'bias_lens--b6': r'b_6',
    'intrinsic_alignment_parameters--a1': r'A_1',
    'intrinsic_alignment_parameters--a2': r'A_2',
    'intrinsic_alignment_parameters--alpha1': r'\alpha_1',
    'intrinsic_alignment_parameters--alpha2': r'\alpha_2',
    'intrinsic_alignment_parameters--bias_ta': r'b_ta',
    'intrinsic_alignment_parameters--z_piv': r'z_\text{piv}',

    'mag_alpha_lens--alpha_1': r'\alpha_\text{lens}^1',
    'mag_alpha_lens--alpha_2': r'\alpha_\text{lens}^2',
    'mag_alpha_lens--alpha_3': r'\alpha_\text{lens}^3',
    'mag_alpha_lens--alpha_4': r'\alpha_\text{lens}^4',
    'mag_alpha_lens--alpha_5': r'\alpha_\text{lens}^5',
    'mag_alpha_lens--alpha_6': r'\alpha_\text{lens}^6',
    'shear_calibration_parameters--m1_uncorr': 'm_1',
    'shear_calibration_parameters--m2_uncorr': 'm_2',
    'shear_calibration_parameters--m3_uncorr': 'm_3',
    'shear_calibration_parameters--m4_uncorr': 'm_4',
    
    'mag_alpha_lens--alpha_1': r'\alpha^\ell_1',
    'mag_alpha_lens--alpha_3': r'\alpha^\ell_3',
    'mag_alpha_lens--alpha_4': r'\alpha^\ell_4', 
    'mag_alpha_lens--alpha_5': r'\alpha^\ell_5', 
    'mag_alpha_lens--alpha_6': r'\alpha^\ell_6',
    
    'source_photoz_u--u_0_uncorr': 'u^s_{0}',
    'source_photoz_u--u_1_uncorr': 'u^s_{1}',
    'source_photoz_u--u_2_uncorr': 'u^s_{2}',
    'source_photoz_u--u_3_uncorr': 'u^s_{3}',
    'source_photoz_u--u_4_uncorr': 'u^s_{4}',
    'source_photoz_u--u_5_uncorr': 'u^s_{5}',
    'source_photoz_u--u_6_uncorr': 'u^s_{6}',
    
    'lens_photoz_u--u_0_0': r'u^\ell_{00}',
    'lens_photoz_u--u_0_1': r'u^\ell_{01}',
    'lens_photoz_u--u_0_2': r'u^\ell_{02}',
    'lens_photoz_u--u_2_0': r'u^\ell_{20}',
    'lens_photoz_u--u_2_1': r'u^\ell_{21}',
    'lens_photoz_u--u_2_2': r'u^\ell_{22}',
    'lens_photoz_u--u_3_0': r'u^\ell_{30}',
    'lens_photoz_u--u_3_1': r'u^\ell_{31}',
    'lens_photoz_u--u_3_2': r'u^\ell_{32}',
    'lens_photoz_u--u_4_0': r'u^\ell_{40}',
    'lens_photoz_u--u_4_1': r'u^\ell_{41}',
    'lens_photoz_u--u_4_2': r'u^\ell_{42}',
    'lens_photoz_u--u_5_0': r'u^\ell_{50}',
    'lens_photoz_u--u_5_1': r'u^\ell_{51}',
    'lens_photoz_u--u_5_2': r'u^\ell_{52}',
    
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
        data['cosmological_parameters--w0wa'] = w0 + wa
        

    
    if 'rescale_pk_fz--sigma_8_0' in data.keys():
        for i in range(9):
            data[f'rescale_pk_fz--s8_{i}'] = \
                data[f'rescale_pk_fz--sigma_8_{i}']*(data['cosmological_parameters--omega_m']/0.3)**0.5


    return data

param_to_label = np.vectorize(lambda param: label_dict[param] if param in label_dict else param)
param_to_latex = np.vectorize(lambda param: '${}$'.format(label_dict[param]) if param in label_dict else param)