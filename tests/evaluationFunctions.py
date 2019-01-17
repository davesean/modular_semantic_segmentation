import numpy as np
import cv2

def computePRvalues(simMat, mskMat):
    thresholds = [0.2,0.4,0.6,0.8]

    precision = np.zeros((len(thresholds),1))
    recall = np.zeros((len(thresholds),1))

    dx = int(mskMat.shape[1]/simMat.shape[1])
    ppd = simMat.shape[1]
    for t,thresh in enumerate(thresholds):
        true_pos_counter = 0
        true_neg_counter = 0
        false_pos_counter = 0
        false_neg_counter = 0
        for k in range(simMat.shape[0]):
            mask_gt = mskMat[k,:,:]
            mask_sim = simMat[k,:,:]

            if np.sum(mask_gt) == 0:
                mask_gt = (~mask_gt.astype(bool)).astype(int)
                mask_sim = 1-mask_sim
                thresholds.reverse()

            for j in range(ppd):
                for i in range(ppd):
                        patch_gt = mask_gt[j*dx:(j+1)*dx,i*dx:(i+1)*dx]
                        if (np.sum(patch_gt)/patch_gt.size > 0.5):
                            true_label = 1
                        else:
                            true_label = 0
                        if  mask_sim[j,i] < thresh:
                            class_label = 0
                        else:
                            class_label = 1

                        if (true_label == 1 and class_label == 1):
                            true_pos_counter+=1
                        elif (true_label == 1 and class_label == 0):
                            false_neg_counter+=1
                        elif (true_label == 0 and class_label == 1):
                            false_pos_counter+=1
                        else:
                            true_neg_counter+=1

        if (true_pos_counter+false_pos_counter) == 0:
            precision[t] = 1
        else:
            precision[t] = true_pos_counter/(true_pos_counter+false_pos_counter)
        if (true_pos_counter+false_neg_counter == 0):
            recall[t] = 1
        else:
            recall[t] = true_pos_counter/(true_pos_counter+false_neg_counter)

    return thresholds, precision, recall

def computeIOU(simMat, mskMat):
    thresholds = [0.2,0.4,0.6,0.8]
    iou = np.zeros((len(thresholds),1))
    OoD = np.zeros((len(thresholds),1))

    for t,thresh in enumerate(thresholds):
        avgIOU = 0
        avgOoD = 0
        for k in range(simMat.shape[0]):
            mask_gt = mskMat[k,:,:]
            mask_sim = simMat[k,:,:]
            if np.sum(mask_gt) == 0:
                mask_gt = (~mask_gt.astype(bool)).astype(int)
                mask_sim = 1-mask_sim
                thresholds.reverse()

            avgOoD += np.sum(mask_gt)/(mskMat.shape[1]*mskMat.shape[2])
            sim = cv2.resize(mask_sim,(mskMat.shape[1],mskMat.shape[2]),interpolation=cv2.INTER_NEAREST)
            simMask = (sim>thresh).astype(int)

            # inter = np.sum(simMask[mask_gt.astype(bool)])
            inter = np.sum(mask_gt[simMask.astype(bool)])
            union = np.sum(((simMask+mask_gt) > 0).astype(int))
            if union > 0:
                avgIOU +=inter/union
            else:
                avgIOU += 1 # TODO check about this
        iou[t] = avgIOU/(simMat.shape[0])
        OoD[t] = avgOoD/(simMat.shape[0])
    return thresholds, iou, OoD