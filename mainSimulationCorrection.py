from default_setup import *
# Version 1
def artifact_correction_mode(MyCase = 1, R_setup = 1, th = 0.27, Omega = None):
    if Omega is None:
        temp_omega = [np.random.random() + 0.7 for i in range(MyCase)]
    else:
        if len(Omega) != MyCase:
            prt_yellow('Set Omegas constant across {} PEDs'.format(MyCase))
            const_omega = np.random.random() +0.7
            temp_omega = [const_omega for i in range(MyCase)]
        else: temp_omega = Omega
    temp_omega = np.array(temp_omega)
    OmegaSign = np.array([-1, 1, 1, -1])
    Omega = np.zeros(4)
    randomIdx = np.random.permutation(4)
    Omega[randomIdx[:MyCase]] = temp_omega
    Omega = Omega*OmegaSign

    if R_setup == 1:
        ghost_func = generate_N2_ghost
        synGhost_func = generate_N2_ghost
    else:
        R_setup = 2
        ghost_func = generate_N4_ghost
        synGhost_func = generate_N4_ghost

    #
    # ped_str = ['LR', 'RL', 'AP', 'PA']
    # t2w_fn = 'subj2_Philips_s0_T2W_fatsat_midsag_acpc.nii'
    # mask_fn = 'mask_mask.nii'
    #
    # FigureFolder = 'C:/Users/thaias/OneDrive - National Institutes of Health/Figures/simulationT2W'
    # DataFolder = 'C:/Users/thaias/OneDrive - National Institutes of Health/T2W_ghost_simulation/SimulationData'
    # s_id = 90
    # data = np.fliplr(nib.load(os.path.join(DataFolder, t2w_fn)).get_fdata()[..., s_id].T)
    # mask = np.fliplr(nib.load(os.path.join(DataFolder, 'mask_mask.nii')).get_fdata()[..., s_id].T)
    # data_masked = data * mask

    GhostImgs = generate_APLR_imgs_from1img(data_masked, Omega, ghost_func)
    GhostImgsLabeled = otsu_threshold_3D(np.abs(GhostImgs))*mask[..., np.newaxis]
    GhostImgsLabeled = morp_closing(GhostImgsLabeled)
    AliasedImgs = data_masked[..., np.newaxis] + R_setup* GhostImgs * GhostImgsLabeled
    vba_arr = compute_vba_arr(AliasedImgs, mask_arr=mask)
    vba_arr_th = vba_arr > th
    SyntheticImgs = generate_APLR_imgs_from4img(AliasedImgs * mask[..., np.newaxis],
                                                [1, 1, 1, 1],
                                                synGhost_func)

    SyntheticMaps = otsu_threshold_3D(SyntheticImgs)
    SyntheticMaps = morp_closing(SyntheticMaps, 3)
    Detected_ArtifactMaps = vba_arr_th * SyntheticMaps * mask[..., np.newaxis]
    C1 = correct_method_OnlyVBA(AliasedImgs,mask, vba_arr, th = 0.27 )*mask
    C1[np.isnan(C1)] = 0
    C2 = correct_method_synGhost_withVBA(AliasedImgs,SyntheticImgs, vba_arr)*mask
    C2[np.isnan(C2)] = 0

    Average = np.mean(AliasedImgs, axis = -1)*mask

    return Omega, Average, C1, C2

def compute_ape(arr1, gt):
    out = (np.abs(arr1 - gt) / np.abs(gt)) * 100
    out[np.isnan(out)] = 0
    out[np.isinf(out)] = 0
    return out

def compute_se(arr1, gt):
    return (arr1**2 - gt**2)

def compute_meanval(arr):
    return np.mean(arr)

def compute_ssim2D(arr1, gt, nanMask = None, full = False):
    ssim_val, ssim_img = ssim(arr1, gt, full = True, data_range=gt.max() - gt.min())
    if nanMask is not None:
        ssim_img_nan  = ssim_img*nanMask
        ssim_val = np.nanmean(ssim_img_nan)
    if full:
        return ssim_val, ssim_img
    return ssim_val

def compute_mape2d(arr1,gt, nanMask = None, full = False):
    mape_img = compute_ape(arr1, gt)
    mape_val = compute_meanval(mape_img)
    if nanMask is not None:
        mape_img_nan = mape_img * nanMask
        mape_val = np.nanmean(mape_img_nan)
    if full:
        return mape_val, mape_img
    return mape_val

# def single_simulation(myCase, R_setup, heuristic_th, Omegas = None, return_imgs = False):
#     # If you want to run multiple-iterations, please set np.random.seed beforehand.
#     myplt = PlotSettup(save_folder=mySubjectScan.output_path)
#     Omega, Average, C1, C2 = artifact_correction_mode(myCase, R_setup,
#                                                     th=heuristic_th,
#                                                     Omega=Omegas
#                                                    )
#     ssim_vals = [ssim(img*mask2D, data2D*mask2D) for img in [Average, C1, C2]]
#     mape_vals = [compute_meanval(compute_ape(img*mask2D, data2D*mask2D)) for img in [Average, C1, C2]]
#     if return_imgs:
#         return Average, C1, C2
#     return ssim_vals, mape_vals
#
#
# def re_evaluate_sinlge_simulation_metrics():
#     np.random.seed(10)
#     Average, C1, C2 = single_simulation(2, 2, 0.27, return_imgs=True)
#     mask2D_nan = mask2D.copy()
#     mask2D_nan[mask2D_nan ==0] = np.nan
#     for img in [Average, C1, C2]:
#         ssim_val , ssim_img = ssim(img*mask2D, data2D*mask2D, full=True)
#         print(ssim_val,  np.mean(ssim_img))
#         ssim_img_nan = ssim_img*mask2D_nan
#         prt_cyan('SSIM with Mask: {}, and without Mask: {}'.format(ssim_val, np.nanmean(ssim_img_nan)))
#         mape_img = compute_ape(img*mask2D, data2D*mask2D)
#         mape_val = compute_meanval(mape_img)
#         mape_img_nan = mape_img.copy()*mask2D_nan
#         # mape_img_nan[mape_img_nan ==0] = np.nan
#         prt_green('MAPE with Zeroes: {}, and without Zeros: {}'.format(mape_val, np.nanmean(mape_img_nan)))



def random_iteration_test(myCase = 1, R_setup = 1,n_iters = 100,
                          heuristic_th = 0.3,
                          print_all = False, overwrite = False):
    np.random.seed(10)
    prt_cyan('MyCase {}, Rsetup: {}'.format(myCase, R_setup))
    c = 0
    MAPE_array = np.zeros((3,n_iters))
    SSIM_array = np.zeros((3,n_iters))
    mySim = SimulationSetup(isflip=True, s_id=90)

    # mySim.set_sub_output_path('SimulatedReports')
    # MAPE_fn = os.path.join(mySim.sub_output_path, 'MAPE_Case{}_R{}'.format(myCase, R_setup))
    # SSIM_fn = os.path.join(mySim.sub_output_path, 'SSIM_Case{}_R{}'.format(myCase, R_setup))

    mask2D_nan = mask.copy()
    mask2D_nan[mask2D_nan == 0] = np.nan
    gt = data_masked
    while (c<n_iters):
        Omega, Average, C1, C2 = artifact_correction_mode(myCase, R_setup,
                                                          th=heuristic_th,
                                                          Omega=None
                                                          )
        mape_vals = np.array([compute_mape2d(img*mask, gt, mask2D_nan) for img in [Average, C1, C2]])
        ssim_vals = np.array([compute_ssim2D(img*mask, gt, mask2D_nan) for img in [Average, C1, C2]])
        MAPE_array[:, c] = mape_vals
        SSIM_array[:, c] = ssim_vals
        if print_all:
            prt_cyan('Omega: \t{}'.format(Omega))
            prt_cyan('mape_vals: \t{}'.format( mape_vals))
            prt_cyan('ssim_vals: \t{}'.format( ssim_vals))
        c+=1
    # np.save(MAPE_fn, MAPE_array)
    # np.save(SSIM_fn, SSIM_array)

    prt_yellow('Average MAPE_array: {} +/- {}'.format(np.mean(MAPE_array, axis = -1),
                                                     np.std(MAPE_array, axis = -1)))
    prt_yellow('Average SSIM_array: {} +/- {}'.format(np.mean(SSIM_array, axis = -1),
                                                     np.std(SSIM_array, axis = -1)))

random_iteration_test(1,1,n_iters=100,overwrite = True)
random_iteration_test(2,1,n_iters=100,overwrite = True)
random_iteration_test(1,2,n_iters=100,overwrite = True)
random_iteration_test(2,2,n_iters=100,overwrite = True)
