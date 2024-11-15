import numpy as np

from default_setup import *

imgStr = ['LR', 'RL', 'AP', 'PA']
pedSubStr = ['$_{LR}$', '$_{RL}$',
             '$_{AP}$', '$_{PA}$']
prt_yellow('Pause here')

"""This version is based on the linux version."""
mySim = SimulationSetup(isflip=True, s_id = 90)
myPlot.block_show = False
data, mask = mySim.get_RefData_and_RefMask()
# myPlot.show_imgs_list([data, mask])

# Change between R, and Case to reproduce the report in the paper.
R_setup = 2
Case = 2
if Case == 1:
    Omega = mySim.set_Omegas(1, [0, 1.3, 0, 0])
    tag = "R{}_Case{}_{}".format(R_setup, Case,"1.3")

if Case == 2:
    # Omega = mySim.set_Omegas(1, [0, 0.8, -0.9, 0])
    # tag = "R{}_Case{}_{}".format(R_setup,Case,"0.8_-0.9")
    # Omega = mySim.set_Omegas(1, [0, 0.9, -0.8, 0])
    # tag = "R{}_Case{}_{}".format(R_setup,Case,"0.9_-0.8")

    Omega = mySim.set_Omegas(1, [0, 0, 2.0, -0.7])
    tag = "R{}_Case{}_{}".format(R_setup,Case,"LR_-1.3_RL_1.2")

mySim.set_simulationFuction(R_setup=R_setup)
noMask = False
APLR_img, APLR_gmaps, APLR_gimgs = mySim.generate_AliasedImgs(R_setup,1,Omega)
APLR_img_noMask, APLR_gmaps_noMask, APLR_gimgs_noMask = mySim.generate_AliasedImgs(R_setup,1,Omega, noMask = True)
APLR_vba_arr, APLR_vba_map = mySim.ArtifactDetection(APLR_img, MaskDat=mask)
APLR_syn, APLR_synMaps = mySim.SyntheticImages(APLR_img)
pred_img = APLR_vba_map*APLR_synMaps

true_img = APLR_gmaps
dc_score,recall_score, accuracy_score, precision_score = mySim.compute_detectionMetrics(pred_img, true_img)
prt_lightpurple('Accuracy score R = {}, Case = {}: {}'.format(R_setup, Case,accuracy_score))
prt_lightpurple('Recall score R = {}, Case = {}: {}'.format(R_setup, Case,recall_score))
prt_lightpurple('Precision score R = {}, Case = {}: {}'.format(R_setup, Case,precision_score))



for i in range(4):
    prt_cyan('Omega:{}, dc_score = {}'.format(Omega[i], dc_score[i]))

# prt_cyan(dc_score)
# prt_yellow(mySim.Omega)

"""Correction section"""
# This is the correction part. Please be advice that I will test for all available cases, with constant omegas.
# However, in the paper, only case 2, R=2 is reported
C1 = mySim.CorrectV1(APLR_img, mySim.RefMask, APLR_vba_arr, th = 0.3)
C1[np.isnan(C1)] = 0
# C2 = mySim.CorrectV2(APLR_img, mySim.RefMask, APLR_syn)
C2 = mySim.CorrectV2(APLR_img, APLR_syn, APLR_vba_arr)
# C1 = mySim.CorrectV1(APLR_img, mySim.RefMask, APLR_vba_arr, th = 0.3)
Cmean = np.mean(APLR_img, axis = -1)
DataMasked = mySim.RefData * mySim.RefMask

Cs = np.squeeze(np.stack([DataMasked[...,0], Cmean, C1, C2], axis = -1))
Cs = Cs*mySim.RefMask
myPlot.show_imgs_array(Cs)
"""=>> Image Quality Metrics"""
Cdifs = Cs - DataMasked
Cdifs_max = np.max(Cdifs)
Cdifs_min = np.min(Cdifs)

prt_cyan('Max and Min of Cdif without absolute: {} ,  {}'.format(Cdifs_max, Cdifs_min))
prt_cyan('While max and min of true signals: {},  {}'.format(np.max(mySim.RefData), np.min(mySim.RefData)))

myPlot.show_imgs_array(Cdifs, vmin = np.min(Cdifs), vmax = np.max(Cdifs))

#The Differences between Corrected images is 'Cdifs'
MSE = np.mean(Cdifs**2, axis = (0,1))
mape_temp = np.abs(Cdifs)/np.abs(DataMasked)
mape_temp[np.isnan(mape_temp)] = 0
mape_temp[np.isinf(mape_temp)] = 0
MAPE = np.mean(mape_temp, axis = (0,1))
imgs_list = ['gt', 'mean', 'c1', 'c2']
prt_lightpurple("This part compute within the whole image")
for i in range(4):
    prt_cyan('{} MSE: {} , MAPE: {}'.format(imgs_list[i], MSE[i], MAPE[i]*100))
# Calculate only within the brain-voxels
n_voxels_in_mask = len(mySim.RefMask[mySim.RefMask!=0])
prt_yellow(n_voxels_in_mask)
data_masked = DataMasked[...,0]
mask_arr = mySim.RefMask[...,0]
prt_cyan('This part only compute within the brain maks. ')
for i in range(4):
    # Cdifs is already the difference calculated.

    img_diff = Cdifs[...,i]
    img_diff_square = img_diff**2
    img_diff_abs = np.abs(img_diff)
    img_diff_percentage_abs = np.abs(img_diff)/np.abs(data_masked)
    img_diff_percentage_abs[np.isinf(img_diff_percentage_abs)] = 0
    img_diff_percentage_abs[np.isnan(img_diff_percentage_abs)] = 0

    sum_diff_square = np.sum(img_diff_square[mask_arr == 1])
    mean_diff_square = sum_diff_square/n_voxels_in_mask
    sum_diff_abs = np.sum(img_diff_abs[mask_arr == 1])
    mean_diff_abs = sum_diff_abs/n_voxels_in_mask

    sum_diff_percentage_abs = np.sum(img_diff_percentage_abs[mask_arr == 1])
    mean_diff_percentage_abs = sum_diff_percentage_abs/n_voxels_in_mask
    prt_yellow('img {}, mse = {} \t; mae = {} \t; mape = {} '.format(imgs_list[i],
                                                                     mean_diff_square,
                                                                     mean_diff_abs,
                                                                     mean_diff_percentage_abs))

saveImages = False
if saveImages:
    myPlot.overwrite_all = False
    f = myPlot.show_imgs_array(APLR_img,list_of_names=False,
                               crop_x=[21, 200], crop_y=[21,172],
                               save_figname='Imgs_{}'.format(tag))
    f = myPlot.show_imgs_array(Cs, list_of_names=['gt', 'mean', 'c1', 'c2'],
                               crop_x=[21, 200], crop_y=[21,172],
                               save_figname='CorrectedImages_{}'.format(tag))
    f = myPlot.show_imgs_array(Cdifs, list_of_names=['gt', 'mean', 'c1', 'c2'],
                               crop_x=[21, 200], crop_y=[21, 172],
                               save_figname='CorrectedImages_{}'.format(tag))
    f = myPlot.show_imgs_array(mape_temp, list_of_names=['gt', 'mean', 'c1', 'c2'],
                               crop_x=[21, 200], crop_y=[21, 172],
                               vmax = 0.2, vmin = 0.0001,
                               mycolor = 'hot',
                               save_figname='AbsPerErrImgs_{}'.format(tag))

    f = myPlot.show_imgs_array(APLR_img_noMask,list_of_names=False,
                               crop_x=[21, 200], crop_y=[21,172],
                               save_figname='Imgs_noMask_{}'.format(tag))

    f = myPlot.show_imgs_array(APLR_gmaps_noMask,list_of_names=False,
                               crop_x=[21, 200], crop_y=[21,172],
                               save_figname='TrueGhostMap_noMask_{}'.format(tag))
    f = myPlot.show_imgs_array(APLR_gimgs_noMask,list_of_names=False, crop_x=[21, 200], crop_y=[21,172],
                               save_figname='TrueGhostImg_noMask_{}'.format(tag))

    f = myPlot.show_imgs_array(APLR_vba_arr,list_of_names=False,crop_x=[21, 200], crop_y=[21,172], vmin = 0, vmax = 1.0,
                               save_figname='VBAarr_{}'.format(tag))
    # myPlot.save_figure(f, 'VBAarr_{}'.format(tag))
    f = myPlot.show_imgs_array(APLR_vba_map,list_of_names=False,crop_x=[21, 200], crop_y=[21,172],
                               save_figname='VBAthr_{}'.format(tag))
    # myPlot.save_figure(f, 'VBAthr_{}'.format(tag))
    f = myPlot.show_imgs_array(APLR_syn,list_of_names=False,crop_x=[21, 200], crop_y=[21,172],
                               save_figname='synghostImg_{}'.format(tag))
    # myPlot.save_figure(f, 'synGhostImg_{}'.format(tag))
    f = myPlot.show_imgs_array(APLR_synMaps,list_of_names=False,crop_x=[21, 200], crop_y=[21,172],
                               save_figname='synGhostMask_{}'.format(tag))
    # myPlot.save_figure(f, 'synGhostMask_{}'.format(tag))

    f = myPlot.show_imgs_array(pred_img,list_of_names=False,crop_x=[21, 200], crop_y=[21,172],
                               save_figname='detectArt_{}'.format(tag))
    # myPlot.save_figure(f, 'detectArt_{}'.format(tag))
