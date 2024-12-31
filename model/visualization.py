import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def get_viz(b, Ms, Qs, mask, input_image):
    gt_mask = Ms[b].cpu().squeeze()
    gt_fin  = np.zeros([gt_mask.shape[0], gt_mask.shape[1], 3])
    gt_fin[gt_mask == 0] = np.array([0, 255, 0])
    gt_fin[gt_mask == 1] = np.array([255, 0, 0])
    gt_fin[gt_mask == 2] = np.array([0, 0, 0])

    pr_mask = mask[b].cpu().squeeze().detach().numpy()
    pr_mask = np.argmax(pr_mask, axis=0)
    #print(np.unique(pr_mask))
    pr_fin = np.zeros([pr_mask.shape[0], pr_mask.shape[1], 3])                
    pr_fin[pr_mask == 0] = np.array([0, 255, 0]) # green for aware
    pr_fin[pr_mask == 1] = np.array([255, 0, 0]) # red for aware
    #mask_b = mask[b].squeeze()
    gt_image = Image.fromarray(np.uint8(gt_fin))
    #logits_image = F.to_pil_image(logits_b)
    mask_image = Image.fromarray(np.uint8(pr_fin))

    # image == data['input']
    # viz_inputs_with_gaze_overlaid([im0, im1, im2], rgb_image, gt_im, pr_im))
    # im0 = Image.fromarray(np.uint8(image.numpy()[1]*255)) # take G channel instead of R for input segmentation
    # im1 = Image.fromarray(np.uint8(mask.numpy()[0]*255)) 
    # im2 = Image.fromarray(np.uint8(image.numpy()[2]*255))
    # image_inputs = [im0, im1, im2]
    # gaze_heatmap = np.array(img_inputs[-1])[4:-4, :]*255
    
    # I think the heatmaps themselves are not being created properly???/

    im2 = Image.fromarray(np.uint8(input_image[b].cpu().numpy()[1]*255))
    gaze_heatmap = np.array(im2)[4:-4, :]*255

    #heatmap_image = Qs[b].cpu().squeeze()

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes[0].imshow(gt_image)
    axes[0].set_title('Ground Truth')
    axes[1].imshow(mask_image)
    axes[1].set_title('Prediction')
    axes[2].imshow(gaze_heatmap)
    axes[2].set_title('Gaze Heatmap')
    plt.tight_layout()
    figure = fig

    plt.close('all')

    return figure
    #return gt_image, mask_image, gaze_heatmap
