import numpy as np

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
]

label_dict = {
    'cosmological_parameters--tau':  r'\tau',
    'cosmological_parameters--w':  r'w',
    'cosmological_parameters--wa':  r'w_a',
    'cosmological_parameters--wp':  r'w_p',
    'cosmological_parameters--w0_fld':  r'w_{GDM}',
    'cosmological_parameters--cs2_fld': r'c_s^2',
    'cosmological_parameters--log_cs2': r'log(c_s^2)',
    'cosmological_parameters--omega_m': r'\Omega_m',
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
}

def add_extra(data, extra=None, weights=None):
    if extra is not None:
        data.update(extra)

    data['cosmological_parameters--s8'] = \
        data['cosmological_parameters--sigma_8']*(data['cosmological_parameters--omega_m']/0.3)**0.5

    data['cosmological_parameters--ommh2'] = \
        data['cosmological_parameters--omega_m']*data['cosmological_parameters--h0']**2

    data['cosmological_parameters--ombh2'] = \
        data['cosmological_parameters--omega_b']*data['cosmological_parameters--h0']**2

    data['cosmological_parameters--omch2'] = \
        data['cosmological_parameters--ommh2'] - data['cosmological_parameters--ombh2']

    data['cosmological_parameters--omega_c'] = \
        data['cosmological_parameters--omega_m'] - data['cosmological_parameters--omega_b']
    
    if 'cosmological_parameters--w' in data.keys() and 'cosmological_parameters--wa' in data.keys() and weights is not None:
        w0, wa = data['cosmological_parameters--w'], data['cosmological_parameters--wa']
        w0wa_cov = np.cov(w0, wa, aweights=weights)
        ap = 1. + w0wa_cov[0,1]/w0wa_cov[1,1]
        data['cosmological_parameters--ap'] = ap
        data['cosmological_parameters--wp'] = w0 + wa*(1. - ap)

    return data

param_to_label = np.vectorize(lambda param: label_dict[param] if param in label_dict else param)
param_to_latex = np.vectorize(lambda param: '${}$'.format(label_dict[param]) if param in label_dict else param)