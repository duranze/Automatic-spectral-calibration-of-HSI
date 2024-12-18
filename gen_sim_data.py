import pickle
import numpy as np
with open('ill_sim.pkl', 'rb') as file:
    # load 10 illuminations
    data = pickle.load(file)
    print(data.keys())
# Before generate extended data, please download the original data from the link in the github page
# https://github.com/duranze/Automatic-spectral-calibration-of-HSI/tree/main
img_id = '' # eg:1246
gt_ref_path = 'your_data_root/gt_files/gtRef_'+img_id+'.npy'
save_root = './extended/'
ref_data = np.load(gt_ref_path)
for ill in data.keys():
    sc_data = ref_data * data[ill]
    np.save(save_root+ill+'_'+img_id+'.npy',sc_data.astype(np.float16))

# we also provide a function hyper_resample for 31 subset generation


def hyper_resample(datacube):
    import numpy as np
    from scipy.interpolate import interp1d
    from scipy.io import loadmat,savemat
    """
    Resample a hyperspectral data cube to new specified wavelengths.
    
    Args:
    datacube (np.array): The input hyperspectral data cube with shape (height, width, num_bands).
    wavelens (np.array): The original wavelengths corresponding to the third dimension of the datacube.
    samp_waves (np.array): The target wavelengths for resampling.
    
    Returns:
    np.array: The resampled hyperspectral data cube.
    """
    # Initialize the output data cube with the new shape
    samp_waves = np.linspace(400, 700, 31)
    wavelens = np.array([
    397.32,
    400.20,
    403.09,
    405.97,
    408.85,
    411.74,
    414.63,
    417.52,
    420.40,
    423.29,
    426.19,
    429.08,
    431.97,
    434.87,
    437.76,
    440.66,
    443.56,
    446.45,
    449.35,
    452.25,
    455.16,
    458.06,
    460.96,
    463.87,
    466.77,
    469.68,
    472.59,
    475.50,
    478.41,
    481.32,
    484.23,
    487.14,
    490.06,
    492.97,
    495.89,
    498.80,
    501.72,
    504.64,
    507.56,
    510.48,
    513.40,
    516.33,
    519.25,
    522.18,
    525.10,
    528.03,
    530.96,
    533.89,
    536.82,
    539.75,
    542.68,
    545.62,
    548.55,
    551.49,
    554.43,
    557.36,
    560.30,
    563.24,
    566.18,
    569.12,
    572.07,
    575.01,
    577.96,
    580.90,
    583.85,
    586.80,
    589.75,
    592.70,
    595.65,
    598.60,
    601.55,
    604.51,
    607.46,
    610.42,
    613.38,
    616.34,
    619.30,
    622.26,
    625.22,
    628.18,
    631.15,
    634.11,
    637.08,
    640.04,
    643.01,
    645.98,
    648.95,
    651.92,
    654.89,
    657.87,
    660.84,
    663.81,
    666.79,
    669.77,
    672.75,
    675.73,
    678.71,
    681.69,
    684.67,
    687.65,
    690.64,
    693.62,
    696.61,
    699.60,
    702.58,
    705.57,
    708.57,
    711.56,
    714.55,
    717.54,
    720.54,
    723.53,
    726.53,
    729.53,
    732.53,
    735.53,
    738.53,
    741.53,
    744.53,
    747.54,
    750.54,
    753.55,
    756.56,
    759.56,
    762.57,
    765.58,
    768.60,
    771.61,
    774.62,
    777.64,
    780.65,
    783.67,
    786.68,
    789.70,
    792.72,
    795.74,
    798.77,
    801.79,
    804.81,
    807.84,
    810.86,
    813.89,
    816.92,
    819.95,
    822.98,
    826.01,
    829.04,
    832.07,
    835.11,
    838.14,
    841.18,
    844.22,
    847.25,
    850.29,
    853.33,
    856.37,
    859.42,
    862.46,
    865.50,
    868.55,
    871.60,
    874.64,
    877.69,
    880.74,
    883.79,
    886.84,
    889.90,
    892.95,
    896.01,
    899.06,
    902.12,
    905.18,
    908.24,
    911.30,
    914.36,
    917.42,
    920.48,
    923.55,
    926.61,
    929.68,
    932.74,
    935.81,
    938.88,
    941.95,
    945.02,
    948.10,
    951.17,
    954.24,
    957.32,
    960.40,
    963.47,
    966.55,
    969.63,
    972.71,
    975.79,
    978.88,
    981.96,
    985.05,
    988.13,
    991.22,
    994.31,
    997.40,
   1000.49,
   1003.58
])
    output_cube = np.zeros((datacube.shape[0], datacube.shape[1], len(samp_waves)))
    
    # Iterate through each pixel in the spatial dimensions
    for i in range(datacube.shape[0]):
        for j in range(datacube.shape[1]):
            # Extract the spectrum at this pixel
            spectrum = datacube[i, j, :]
            
            # Create an interpolation function for the current spectrum
            interp_func = interp1d(wavelens, spectrum, kind='linear', bounds_error=False, fill_value="extrapolate")
            
            # Use the interpolation function to resample the spectrum
            resampled_spectrum = interp_func(samp_waves)
            
            # Place the resampled spectrum in the output data cube
            output_cube[i, j, :] = resampled_spectrum
    return output_cube