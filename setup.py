import multiprocessing
from utils import *
from skimage.filters.thresholding import threshold_otsu
from skimage import morphology as morp
from scipy.spatial.distance import dice
def setValuetoNan(data, defaultVal = 0):
    return np.where(data == defaultVal, np.nan, data)

def setNantoVal(data, defaultVal = 0):
    return np.where(np.isnan(data), defaultVal, data)

def generate4PEDsGhosts(refImg, R, Omega = [1,1,1,1]):
    LRRL = syntheticGhost(refImg,1,R)
    APPA = syntheticGhost(refImg, 0,R)
    LR = LRRL*Omega[0]
    RL = LRRL*Omega[1]
    AP = APPA*Omega[2]
    PA = APPA*Omega[3]
    return np.squeeze(np.stack([ LR, RL, AP, PA], axis =-1))

def compute_dice(arr1, arr2):
    return 1- dice(arr1.flatten(), arr2.flatten())

def compute_otsu_img(dat):
    if len(dat.shape)<3:
        th = threshold_otsu(dat)
        return (dat>th)*1.0
    else:
        bin_dat = np.zeros(dat.shape)
        for i in range(dat.shape[2]):
            th = threshold_otsu(dat[...,i])
            bin_dat[...,i] = (dat[...,i] > th) * 1.0
        return bin_dat

def compute_ssw(dat): # Computing the sum-of-square-within
    return np.sum(np.square(dat - np.mean(dat)))

def compute_ssb(dat, mean_dat): # Compute the sum-of-square between
    return np.square(np.mean(dat) - mean_dat)

def patches_vba(*args):
    total = 0
    total_size = 0
    vba_arr = np.zeros(len(args))
    for arg in args:
        sz = arg.size
        total_size += sz
        total += np.sum(arg)
    mean_val = total / total_size
    for a, arg in enumerate(args):
        sz = arg.size
        vba_arr[a] = compute_ssw(arg) + sz * compute_ssb(arg, mean_val)
    return vba_arr / np.sum(vba_arr)


def compute_vba_arr(dat, mask_arr = None, wd_size = 3):
    vba_ = np.zeros([dat.shape[0], dat.shape[1], 4])
    wdsize_x, wdsize_y = wd_size, wd_size
    hw_x, hw_y = wdsize_x // 2, wdsize_y // 2
    if mask_arr is None:
        mask_arr = dat
    for i in range(hw_x, dat.shape[0] - hw_x):
        for j in range(hw_y, dat.shape[1] - hw_y):
                if np.sum(mask_arr[i, j]) == 0:
                    continue
                patch_data = dat[i - hw_x: i + hw_x + 1,
                             j - hw_y: j + hw_y + 1, :]
                vba_[i, j, :] = patches_vba(patch_data[..., 0],
                                                 patch_data[..., 1],
                                                 patch_data[..., 2],
                                                 patch_data[..., 3])
    vba_[np.isnan(vba_)] = 0
    return vba_
def generate_patch_from_ijk(main_dat, ijk, wd_size):
    i, j, k = ijk[0], ijk[1], ijk[2]
    wd_x, wd_y, wd_z = wd_size[0], wd_size[1], wd_size[2]
    hw_x, hw_y, hw_z = wd_x // 2, wd_y // 2, wd_z // 2
    dat = main_dat[i - hw_x:i + hw_x + 1,
          j - hw_y:j + hw_y + 1,
          k - hw_z:k + hw_z + 1,:]
    return dat

def aov_function_mp(data):
    n_samples = 9
    data = (data- np.min(data))/ (np.max(data)-np.min(data)) # DELETE AFTER TEST
    mean_all = np.mean(data)
    mean_grps = [np.mean(data[..., i]) for i in range(4)]
    var_grps = [np.var(data[..., i]) * n_samples for i in range(4)] + n_samples * np.square(mean_grps - mean_all)
    total_var = np.var(data)*data.size
    return var_grps / total_var

def compute_vba_mp(imgdata, wd_size=3, mask_data = None):
    wdsize_x, wdsize_y, wdsize_z = wd_size, wd_size, 1
    hw_x, hw_y, hw_z = wdsize_x // 2, wdsize_y // 2, wdsize_z // 2
    if mask_data is None:
        mask_data = np.mean(imgdata, axis = -1)
    with multiprocessing.Pool() as pool:
        args = [generate_patch_from_ijk(imgdata, [i, j, k], [3, 3, 1]) for i in range(hw_x, imgdata.shape[0] - hw_x) for j in
                range(hw_y, imgdata.shape[1] - hw_y) for k in range(imgdata.shape[2])]
        results = pool.map(aov_function_mp, args)
    results = np.array(results)
    aov_result = np.zeros(imgdata.shape)
    for i in range(hw_x, imgdata.shape[0] - hw_x):
        for j in range(hw_y, imgdata.shape[1] - hw_y):
            for k in range(imgdata.shape[2]):
                if mask_data[i,j,k] == 0:
                    continue
                try:
                    indices = i * (imgdata.shape[1] - hw_y*2) * (imgdata.shape[2]) + j * (imgdata.shape[2]) + k
                    dat = results[indices, :]
                except IndexError:
                    # Catch the empty indices
                    print(i, j, k, indices)
                dat[np.isnan(dat)] = 0
                aov_result[i, j, k, :] = dat
    return aov_result

def morp_closing(img, disksize = 3):
    n_dim = len(img.shape)
    if n_dim <3:
        output = morp.closing(img, footprint=morp.square(disksize))
        return output
    else:
        output = np.zeros(img.shape)
        for i in range(img.shape[-1]):
            output[...,i] = morp.closing(img[...,i], footprint=morp.square(disksize))
        return output

def correctV1(img_arr, vba_arr, th = 0.25):
    vba_temp = np.abs( 1 - vba_arr > th)
    img_art = img_arr*vba_temp
    img_art[img_art == 0] = np.nan
    corrV1 = np.nanmean(img_art, axis = -1)
    corrV1[np.isnan(corrV1)] = 0
    return corrV1

def correctV2(img_arr, vba_arr, syn_arr):
    vba_syn = vba_arr * syn_arr
    w_i = 1/vba_syn
    w_i[np.isinf(w_i)] = 0
    w_i[np.isnan(w_i)] = 0
    w = np.sum(w_i, axis = -1)
    corrV2_i = w_i * img_arr/w[..., np.newaxis]
    corrV2_i[corrV2_i == 0] = np.nan
    corrV2 = np.sum(corrV2_i, axis = -1)
    return corrV2