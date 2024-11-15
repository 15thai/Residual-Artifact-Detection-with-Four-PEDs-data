#"This simulation is based on Okan change. It's stupid but what can I do?"


from default_setup import *
"""This version is based on the linux version."""
mySim = SimulationSetup(isflip=True, s_id = 90)
myPlot.block_show = False
data, mask = mySim.get_RefData_and_RefMask()
# myPlot.show_imgs_list([data, mask])


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