from setup import *
from sklearn import metrics

data_dir = 'data'
R = 1
vba_th = 0.3
imgPlot = myPlot()

img2D = np.load(os.path.join(data_dir, 'gt_s90.npy'))
mask2D = np.load(os.path.join(data_dir, 'ma_s90.npy'))
peds = [ 'LR', 'RL','AP', 'PA']

def demonstrate_examples (R = 1, Omegas = [1,1,1,1]):
    ghost4PEDFsCase = generate4PEDsGhosts(img2D, Omega=Omegas, R=R)  # set default [1,1,1,1]
    artOtsuCase = np.squeeze(np.stack([compute_otsu_img(np.abs(ghost4PEDFsCase[..., i])) for i in range(4)], axis=-1)) * \
                  mask2D[..., np.newaxis]
    artOtsuCase = morp_closing(artOtsuCase)
    imgWArtifactsCase = img2D[..., np.newaxis] + ghost4PEDFsCase * artOtsuCase
    vbaCase = compute_vba_arr(imgWArtifactsCase)

    # Base on the acquisition parameter we already know R = 1
    AP_synCase = syntheticGhost(imgWArtifactsCase[..., 0], 0, R)  # Omega=1
    PA_synCase = syntheticGhost(imgWArtifactsCase[..., 0], 0, R)  # Omega=1
    LR_synCase = syntheticGhost(imgWArtifactsCase[..., 0], 1, R)  # Omega=1
    RL_synCase = syntheticGhost(imgWArtifactsCase[..., 0], 1, R)  # Omega=1
    synOtsuCase = [compute_otsu_img(img) for img in [LR_synCase, RL_synCase, AP_synCase, PA_synCase]]
    detectedArtifactsCase = np.squeeze(np.stack(synOtsuCase, axis=-1)) * vbaCase > vba_th
    detectedArtifactsCase = morp_closing(detectedArtifactsCase)
    for i in range(4):
        print('======================Detection metric R = {}, of PED {}'.format(R ,peds[i]))
        recall_score = metrics.recall_score(detectedArtifactsCase[..., i].ravel(), artOtsuCase[..., i].ravel())
        acc_score = metrics.accuracy_score(detectedArtifactsCase[..., i].ravel(), artOtsuCase[..., i].ravel())
        precision_score = metrics.precision_score(detectedArtifactsCase[..., i].ravel(), artOtsuCase[..., i].ravel())
        dice_score = compute_dice(detectedArtifactsCase[..., i], artOtsuCase[..., i])
        print('recall score: {}'.format(recall_score))
        print('acc score: {}'.format(acc_score))
        print('precision score: {}'.format(precision_score))
        print('dice score: {}'.format(dice_score))

Case1 = demonstrate_examples(1, [-1.3, 0,0,0])
Case2 = demonstrate_examples(1, [-1.3, 1.2,0,0])
Case3 = demonstrate_examples(1, [-1.3, 1.2,1.5,0])
Case4 = demonstrate_examples(1, [-1.3, 1.2,1.5,-0.9])

Case1 = demonstrate_examples(2, [-1.3, 0,0,0])
Case2 = demonstrate_examples(2, [-1.3, 1.2,0,0])
Case3 = demonstrate_examples(2, [-1.3, 1.2,1.5,0])
Case4 = demonstrate_examples(2, [-1.3, 1.2,1.5,-0.9])