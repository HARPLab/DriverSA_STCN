import torch
import torch.nn as nn

class object_level_Accuracy(nn.Module):
    __name__ = "object_level_accuracy"

    def __init__(self, threshold=0.5, remove_small_objects = True, activation=None, ignore_channels=None):
        self.__name__ = "object_level_accuracy"
        self.remove_small_objects = remove_small_objects

    def forward(self, y_pr_raw, y_gt, y_inst):
        
        # get object_ids
        # get_mask for each vehicle id
        # get prediction for each object
        # calculate accuracy
        y_pr = torch.argmax(y_pr_raw[:, :2, ...], dim=1)
        y_gt = torch.argmax(y_gt, dim=1)
        # allowed_inds = y_inst[:, 0, :, :] == 10 or y_inst[:, 0, :, :] == 4 or y_inst[:, 0, :, :] == 23
        ids_tensor = y_inst[:, 1, :, :] + y_inst[:, 2, :, :]*256
        accs = []
        preds = []
        raw_preds = []
        gts = []
        obj_ids = []

        # iterate over the batch
        for b in range(ids_tensor.shape[0]):
            ids = ids_tensor[b].unique()
            # iterate over the objects in each sample
            for id in ids:
                # ignore if the mask is zero for this object
                inds = ids_tensor[b] == id
                if (self.remove_small_objects == True) and torch.sum(inds) < 10:
                    continue
                y_pr_obj = y_pr[b][inds]
                y_gt_obj = y_gt[b][inds]
                val = torch.mode(y_pr_obj)
                obj_pr = val[0].item()
                # obj_pr_ind = val[1].item()
                obj_gt = torch.mode(y_gt_obj)[0].item()
                if obj_gt == 2:
                    continue
                # obj_gt = y_gt_obj[0].item()
                # print(id, obj_pr, obj_gt)
                accs.append(obj_pr == obj_gt)
                preds.append(obj_pr)
                renorm_y_pr_raw = y_pr_raw[:, :2, ...]/torch.sum(y_pr_raw[:, :2, ...], dim=1, keepdim=True) 
                raw_preds.append((renorm_y_pr_raw[:, :2, ...][b][1][inds] - renorm_y_pr_raw[:, :2, ...][b][0][inds]).cpu().detach().numpy())
                gts.append(obj_gt)
                obj_ids.append(id)
        if len(accs) == 0:
            return 0, preds, gts, raw_preds, obj_ids
        return sum(accs)/len(accs), preds, gts, raw_preds, obj_ids