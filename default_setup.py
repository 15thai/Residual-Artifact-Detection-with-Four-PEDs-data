import sys
"""This version of code, is to run detection simulation. 
For reporting in the paper. 
"""
import numpy as np

sys.path.append('C:/Users/thaias/OneDrive - National Institutes of Health/T2W_ghost_simulation')
from T4setup import *
from functions import *
from plot_functions import *

#---------------------------------
ped_str = ['LR', 'RL', 'AP', 'PA']
t2w_fn = 'subj2_Philips_s0_T2W_fatsat_midsag_acpc.nii'
mask_fn  = 'mask_mask.nii'

FigureFolder = 'C:/Users/thaias/OneDrive - National Institutes of Health/Figures/simulationT2W'
DataFolder = 'C:/Users/thaias/OneDrive - National Institutes of Health/T2W_ghost_simulation/SimulationData'
# s_id = 90
# data = np.fliplr(nib.load(os.path.join(DataFolder, t2w_fn)).get_fdata()[...,s_id].T)
# mask = np.fliplr(nib.load(os.path.join(DataFolder, 'mask_mask.nii')).get_fdata()[..., s_id].T)
# data_masked = data*mask

myPlot = PlotSettup(block_show=False, fig_size=(10,7), save_folder=FigureFolder, overwrite_all=False)
myPlot.create_save_folder()
mySubject = ScanSetup(2,0)
GLOBAL_MIN = 0.7

sub_ped_names = ['$_{LR}$',
                 '$_{RL}$',
                 '$_{AP}$',
                 '$_{PA}$']
vba_names = ['VBA$_{LR}$',
             'VBA$_{RL}$',
             'VBA$_{AP}$',
             'VBA$_{PA}$']

true_art_names = ['A{}'.format(i) for i in sub_ped_names]

ped_str = ['LR', 'RL', 'AP', 'PA']
t2w_fn = 'subj2_Philips_s0_T2W_fatsat_midsag_acpc.nii'
mask_fn = 'mask_mask.nii'

FigureFolder = 'C:/Users/thaias/OneDrive - National Institutes of Health/Figures/simulationT2W'
DataFolder = 'C:/Users/thaias/OneDrive - National Institutes of Health/T2W_ghost_simulation/SimulationData'
s_id = 90
data = np.fliplr(nib.load(os.path.join(DataFolder, t2w_fn)).get_fdata()[..., s_id].T)
mask = np.fliplr(nib.load(os.path.join(DataFolder, 'mask_mask.nii')).get_fdata()[..., s_id].T)
data_masked = data * mask






class SimulationSetup():
    def __init__(self,
                 sub_id = 2,
                 scan_id = 0,
                 project_path = DataFolder,
                 outfolder = 'SimPaper',
                 s_id = 50,
                 isflip = False,
                 overwrite_all = False):
        self.sub_id = sub_id
        self.scan_id = scan_id
        self.project_path = project_path
        self.outfolder = outfolder
        self.outpath = os.path.join(DataFolder, outfolder)
        self.datapath = DataFolder
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)
        self.s_id = s_id
        # isflip = True is use for afni, mipav, and other software
        self.isflip = isflip
        self.overwrite_all = overwrite_all
        self.CorrectV1 = correct_method_OnlyVBA
        self.CorrectV2 = correct_method_synGhost_withVBA
        self.figpath = FigureFolder

    def set_value_tonan(self,data,  defaultVal = 0):
        data = np.where(data == defaultVal, np.nan, data)
        return data

    def set_nan_toval(self, data, defaultVal = 0):
        data = np.where(np.isnan(data), defaultVal, data)
        return data

    def get_default_structural_fn(self):
        self.structure_fn = os.path.join(self.datapath, t2w_fn)
        return self.structure_fn

    def get_default_maskfn(self):
        self.mask_fn = os.path.join(self.datapath, mask_fn)
        return self.mask_fn
    def get_RefData(self, imgfn = None, s_id = None):
        if imgfn  is None:
            imgfn = self.get_default_structural_fn()
        prt_cyan('Get default data from img: {}'.format(imgfn))
        RefData = nib.load(imgfn).get_fdata()
        if s_id is None:
            s_id = self.s_id
        if s_id:
            RefData = RefData[..., [s_id]]
        if self.isflip:
            RefData = np.swapaxes(RefData, 1,0)
            RefData = np.flip(RefData, 1)
        self.RefData= RefData
        return self.RefData
    def get_RefMask(self, maskfn = None, s_id = None):
        if maskfn is None:
            maskfn = self.get_default_maskfn()
        prt_cyan('Get default mask from img: {}'.format(maskfn))
        RefMask = nib.load(maskfn).get_fdata()
        if s_id is None:
            s_id = self.s_id
        if s_id:
            RefMask = RefMask[..., [s_id]]
        if self.isflip:
            RefMask = np.swapaxes(RefMask, 1, 0)
            RefMask = np.flip(RefMask, 1)
        self.RefMask = RefMask
        return self.RefMask

    def get_RefData_and_RefMask(self, imgfn = None,maskfn = None, s_id = None):
        RefMask = self.get_RefMask(maskfn, s_id)
        RefData = self.get_RefData(imgfn, s_id)
        return RefData, RefMask

    def set_simulationFuction(self, R_setup = 1):
        if R_setup == 1:
            synGhost_func = generate_N2_ghost
        else:
            synGhost_func = generate_N4_ghost
        self.ghostFunc = synGhost_func
        return synGhost_func

    def compute_dice(self, arr_1, arr_2):
        return 1 - dice(arr_1.flatten(), arr_2.flatten())

    def set_Omegas(self, myCase= 1, Omega = None, minVal = 0.7):
        if Omega is None:
            temp_omega = [np.random.random() + minVal for i in range(myCase)]
        else:
            if len(Omega) == 4:
                self.Omega = Omega
                return Omega
            elif len(Omega) != myCase:
                prt_yellow('Set Omegas constant across {} PEDs '
                           'Special case.'.format(myCase))
                const_omega = np.random.random() + 0.7
                temp_omega = [const_omega for i in range(myCase)]
            else:
                temp_omega = Omega
        temp_omega = np.array(temp_omega)
        OmegaSign = np.array([-1, 1, 1, -1])
        Omega = np.zeros(4)
        randomIdx = np.random.permutation(4)
        Omega[randomIdx[:myCase]] = temp_omega
        Omega = Omega * OmegaSign
        self.Omega = Omega
        return Omega

    def compute_dice_nd(self, arr_1, arr_2):
        if len(arr_1.shape) >2:
            results = []
            for i in range(arr_1.shape[2]):
                result = self.compute_dice(arr_1[...,i], arr_2[...,i])
                results.append(result)
            return results
        else:
            return self.compute_dice(arr_1, arr_2)

    def compute_detectionMetrics(self, pred_y, true_y):
        dice_score = self.compute_dice_nd(pred_y, true_y)
        # dice_score[np.isnan(dice_score)] = 0
        recall_score = metrics.recall_score(pred_y.ravel(), true_y.ravel())
        accuracy_score = metrics.accuracy_score(pred_y.ravel(), true_y.ravel())
        precision_score = metrics.precision_score(pred_y.ravel(), true_y.ravel())
        return dice_score, recall_score, accuracy_score, precision_score

    def SyntheticImages(self, AliaseImages):
        ghostFunc = self.ghostFunc
        if len(AliaseImages.shape) < 4:
            nz = 1
        else:
            nz = AliaseImages.shape[-2]

        synImgs = generate_APLR_imgs_from4img(AliaseImages,
                                                [1,1,1,1],
                                                ghostFunc)
        synMask = otsu_threshold_3D(synImgs)*self.RefMask
        synMask = morp_closing(synMask)
        return synImgs,  synMask

    def ArtifactDetection(self, AliaseImages,
                          SyntheticMap = None,
                          MaskDat = None,
                          wd = 3,
                          heu_th = 0.3
                          ):
        if SyntheticMap is None:
            SyntheticMap = 1

        vba_arr = compute_vba_arr(AliaseImages, mask_arr= MaskDat, wd_size=wd)
        vba_arr_map = (vba_arr > heu_th)
        self.vba_arr = vba_arr
        return vba_arr, vba_arr_map

    def generate_AliasedImgs(self, R_setup=1, MyCase=1, Omega=None, noMask=False):
        RefData, RefMask = self.get_RefData_and_RefMask()
        Omega = self.set_Omegas(MyCase, Omega)
        if len(RefData.shape) < 3:
            RefData = RefData[..., np.newaxis]
            nz = 1
            output = np.zeros([RefData.shape[0], RefData.shape[1], 4])
        else:
            nz = RefData.shape[-1]
            output = np.zeros([RefData.shape[0], RefData.shape[1], nz, 4])

        ghostFunc = self.set_simulationFuction(R_setup)
        RefData = RefData * RefMask
        AliaseImgs = output
        GhostImgs = np.zeros(AliaseImgs.shape)
        GhostMaps = np.zeros(AliaseImgs.shape)
        if noMask:
            RefMask = np.ones(RefData.shape)
        for z in range(nz):
            ghostPart = generate_APLR_imgs_from1img(RefData[..., z],
                                                    Omega,
                                                    ghostFunc)
            ghostMask = otsu_threshold_3D(np.abs(ghostPart)) * RefMask[..., z][..., np.newaxis]
            ghostMask = morp_closing(ghostMask)
            AliaseImgs[..., z, :] = RefData[..., z][..., np.newaxis] +ghostPart * ghostMask
            GhostMaps[..., z, :] = ghostMask * RefMask
            GhostImgs[..., z, :] = ghostPart * RefMask

        return np.squeeze(AliaseImgs), np.squeeze(GhostMaps), np.squeeze(GhostImgs)

    def compute_MAPE(self, gt_dat, pred_dat):
        abs_erro = np.abs(pred_dat - gt_dat)/gt_dat
        mape = np.mean(abs_erro[gt_dat!=0])
        return mape

    def compute_MSE(self, gt_dat, pred_dat):
        square_error = (pred_dat - gt_dat)**2
        mse = np.sum(square_error) / gt_dat.size
        return mse
