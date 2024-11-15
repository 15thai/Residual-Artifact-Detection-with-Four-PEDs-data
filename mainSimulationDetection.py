import numpy as np
from default_setup import *
np.random.seed(10)
def artifact_detection_mode(MyCase = 1, R_setup = 1, th = 0.3, Omega = None, noSynthetic = False):
    if Omega is None:
        temp_omega = [np.random.random() + GLOBAL_MIN for i in range(MyCase)]
    else:
        if len(Omega) != MyCase:
            prt_yellow('Set Omegas constant across {} PEDs'.format(MyCase))
            temp_omega = [Omega for i in range(MyCase)]
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


    ped_str = ['LR', 'RL', 'AP', 'PA']
    t2w_fn = 'subj2_Philips_s0_T2W_fatsat_midsag_acpc.nii'
    mask_fn = 'mask_mask.nii'

    FigureFolder = 'C:/Users/thaias/OneDrive - National Institutes of Health/Figures/simulationT2W'
    DataFolder = 'C:/Users/thaias/OneDrive - National Institutes of Health/T2W_ghost_simulation/SimulationData'
    s_id = 90
    data = np.fliplr(nib.load(os.path.join(DataFolder, t2w_fn)).get_fdata()[...,s_id].T)
    mask = np.fliplr(nib.load(os.path.join(DataFolder, 'mask_mask.nii')).get_fdata()[..., s_id].T)
    data_masked = data*mask

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
    if noSynthetic:
        Detected_ArtifactMaps = morp_closing(vba_arr_th)
    return Omega, Detected_ArtifactMaps, GhostImgsLabeled


def random_iteration_test(myCase = 1,
                          R_setup = 1,
                          n_iters = 100,
                          print_all = False,
                          saveArry = False):
    c = 0
    np.random.seed(10)

    DICE_array = np.zeros((4,n_iters))

    while (c < n_iters):
        Omega, pred_y, true_y = artifact_detection_mode(myCase,R_setup)
        dice_scores = mySubject.compute_dice_nd(pred_y, true_y)
        if print_all:
            prt_cyan('Omega: \t{}'.format(Omega))
            prt_cyan('dice_score: \t{}'.format( dice_scores))
        DICE_array[:,c] = dice_scores
        c+=1
    DICE_array[DICE_array ==0] = np.nan
    prt_cyan('MyCase {}, Rsetup: {}'.format(myCase, R_setup))
    prt_yellow('Average DICE_score: {}'.format(np.nanmean(DICE_array)))
    prt_yellow('Stdev DICE_score: {}'.format(np.nanmean(DICE_array)))
    if saveArry:
        prt_cyan('Save the Dice score')


random_iteration_test(1,1)
random_iteration_test(1,2)
random_iteration_test(2,1)
random_iteration_test(2,2)
random_iteration_test(3,1)
random_iteration_test(3,2)
random_iteration_test(4,1)
random_iteration_test(4,2)
