import torch
import carla
from torch.utils.data import Dataset, Sampler, BatchSampler
from torchvision import transforms
from pathlib import Path
from PIL import Image, ImageDraw
import configparser
import numpy as np
import pandas as pd
import os
from io import BytesIO
import torchvision.transforms.functional as TF

def ptsWorld2Cam(focus_hit_pt, world2camMatrix, K):
    tick_focus_hitpt_homog = np.hstack((focus_hit_pt,1))    
    sensor_points = np.dot(world2camMatrix, tick_focus_hitpt_homog)
    
    # Now we must change from UE4's coordinate system to an "standard" camera coordinate system
    point_in_camera_coords = np.array([
        sensor_points[1],
        sensor_points[2] * -1,
        sensor_points[0]])

    # Finally we can use our K matrix to do the actual 3D -> 2D.
    points_2d = np.dot(K, point_in_camera_coords)

    # Remember to normalize the x, y values by the 3rd value.
    points_2d /= points_2d[2]

    u_coord = points_2d[0].astype(np.int)
    v_coord = points_2d[1].astype(np.int)
    return (u_coord, v_coord)

def world2pixels(focus_hit_pt, vehicle_transform, K, sensor_config):
    '''
    takes in the dataframe row with all the information of where the world is currently 
    '''        
    vehicleP = vehicle_transform.get_matrix()
    
    # center image
    camera_loc_offset = carla.Location(x=float(sensor_config['rgb']['x']), y=float(sensor_config['rgb']['y']), z=float(sensor_config['rgb']['z']))    
    camera_rot_offset = carla.Rotation(pitch=float(sensor_config['rgb']['pitch']), yaw=float(sensor_config['rgb']['yaw']), roll=float(sensor_config['rgb']['roll']))
    cam_transform = carla.Transform(location=camera_loc_offset, rotation=camera_rot_offset)
    world2cam = np.matmul(cam_transform.get_inverse_matrix(), vehicle_transform.get_inverse_matrix())    
    
    u,v = ptsWorld2Cam(focus_hit_pt, world2cam, K)
    pts_mid = (u,v)
        
    # left image  
    camera_loc_offset = carla.Location(x=float(sensor_config['rgb_left']['x']), y=float(sensor_config['rgb_left']['y']), z=float(sensor_config['rgb_left']['z']))    
    camera_rot_offset = carla.Rotation(pitch=float(sensor_config['rgb_left']['pitch']), yaw=float(sensor_config['rgb_left']['yaw']), roll=float(sensor_config['rgb_left']['roll']))
    cam_transform = carla.Transform(location=camera_loc_offset, rotation=camera_rot_offset)    
    world2cam = np.matmul(cam_transform.get_inverse_matrix(), vehicle_transform.get_inverse_matrix())
        
    u,v = ptsWorld2Cam(focus_hit_pt, world2cam, K)
    pts_left = (u,v)
    
    # right image  
    camera_loc_offset = carla.Location(x=float(sensor_config['rgb_right']['x']), y=float(sensor_config['rgb_right']['y']), z=float(sensor_config['rgb_right']['z']))    
    camera_rot_offset = carla.Rotation(pitch=float(sensor_config['rgb_right']['pitch']), yaw=float(sensor_config['rgb_right']['yaw']), roll=float(sensor_config['rgb_right']['roll']))
    cam_transform = carla.Transform(location=camera_loc_offset, rotation=camera_rot_offset)    
    world2cam = np.matmul(cam_transform.get_inverse_matrix(), vehicle_transform.get_inverse_matrix())
        
    u,v = ptsWorld2Cam(focus_hit_pt, world2cam, K)
    pts_right = (u,v)    
    
    return pts_mid, pts_left, pts_right

def gaze_dot_plot(rgb_image, gaze_points, sigma=10, kernel_size = 61, gaze_fade=False):
    # Create a grid of coordinates
    height, width = rgb_image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask = Image.fromarray(mask)
    mask_draw = ImageDraw.Draw(mask)
    gaze_dot_radius = 5

    gaze_fade_min = 10    
    N = len(gaze_points)
    gaze_fade_step = (255 - gaze_fade_min) / N
    
    if not gaze_fade:
        for center_pixel in gaze_points:

            mask_draw.ellipse((center_pixel[0]-gaze_dot_radius, center_pixel[1]-gaze_dot_radius,
                            center_pixel[0]+gaze_dot_radius, center_pixel[1]+gaze_dot_radius),
                            fill=255)
    else:
        for i, center_pixel in enumerate(gaze_points):
            if center_pixel[0] < -1 or center_pixel[1] < -1:
                continue
            # fade gaze from 255 to gaze_fade_min linearly based on time
            # most recent gaze is gaze_points[0]
            fill_value = int(gaze_fade_min + gaze_fade_step*(N-i))
            mask_draw.ellipse((center_pixel[0]-gaze_dot_radius, center_pixel[1]-gaze_dot_radius,
                center_pixel[0]+gaze_dot_radius, center_pixel[1]+gaze_dot_radius),
                fill=fill_value)

    # if gaze_blur:
    #     dotmap = TF.gaussian_blur(mask, kernel_size, sigma)
    dotmap = transforms.functional.to_tensor(mask)
    
    return dotmap

def gaussian_contour_plot(rgb_image, gaze_points, sigma=10, kernel_size = 61, gaze_fade=False):
    # Create a grid of coordinates
    height, width = rgb_image.shape[:2]
    mask = torch.zeros((1, height, width), dtype=torch.float32)

    gaze_fade_min = 10    
    N = len(gaze_points)
    gaze_fade_step = (255 - gaze_fade_min) / N
    
    if not gaze_fade:
        for center_pixel in gaze_points:
            mask[0, center_pixel[1], center_pixel[0]] = 255
    else:
        for i, center_pixel in enumerate(gaze_points):
            if center_pixel[0] < -1 or center_pixel[1] < -1:
                continue
            # fade gaze from 255 to gaze_fade_min linearly based on time
            # most recent gaze is gaze_points[0]
            mask[0, center_pixel[1], center_pixel[0]] = int(gaze_fade_min + gaze_fade_step*(N-i))

    # convolve the mask with a gaussian kernel    
    # this also normalizes it
    heatmap = TF.gaussian_blur(mask, kernel_size, sigma)
    if not gaze_fade:
        heatmap[heatmap > 0] = 1

    # heatmap_image = Image.fromarray(heatmap.numpy().squeeze())
    # heatmap_image.save('heatmap.png')
    
    return heatmap

def frame_filter(frame_num, awareness_df):
     ego_vel = awareness_df["EgoVariables_VehicleVel"][frame_num]
     visible_list = awareness_df["AwarenessData_Visible"][frame_num]
     if ego_vel == 0 and visible_list == []:
          return 0
     else:
          return 1

def read_corrected_csv(path):
    corrected_labels_df = pd.read_csv(path)
    # Extract columns into numpy arrays
    frame_no = corrected_labels_df['frame_no'].to_numpy()
    visible_total = corrected_labels_df['visible_total'].apply(eval).apply(np.array).to_numpy()
    visible_is = corrected_labels_df['visible_is'].apply(eval).apply(np.array).to_numpy()
    awareness_label = corrected_labels_df['awareness_label'].apply(eval).apply(np.array).to_numpy()
    
    
    data = {'frame_no': frame_no,
    'visible_total': visible_total,
    'awareness_label': awareness_label,
    'visible_is': visible_is}

    return pd.DataFrame(data)

def get_clicked_frame_dict(corrected_labels_df):
    clicked_frame_dict = {}
    for i, row in corrected_labels_df.iterrows():
        visible_total = row['visible_total']
        awareness_label = row['awareness_label']
        # print(visible_total, awareness_label)
        for ind, actor_id in enumerate(visible_total):
            # check if actor has been clicked before
            if actor_id not in clicked_frame_dict:
                # if not, and the actor is aware, put it in the dict and set the visible + clicked status to 1
                if awareness_label[ind] == 1:    
                    clicked_frame_dict[actor_id] = [[row['frame_no']], 1]
                # if not, and the actor is unaware, do not put it in the dict
            else:
                # if the actor is in the dict, check if the actor is aware
                if awareness_label[ind] == 1:
                    # if the actor is aware, check if the actor's visible and clicked status is 0
                    if clicked_frame_dict[actor_id][1] == 0:
                        # if it is, append the frame number to the list of click times and set the visible + clicked status to 1
                        clicked_frame_dict[actor_id][0].append(row['frame_no'])
                        clicked_frame_dict[actor_id][1] = 1
                else:
                    # if the actor is unaware, check if the actor's visible and clicked status is 1
                    if clicked_frame_dict[actor_id][1] == 1:
                        # if it is, set the visible + clicked status to 0
                        clicked_frame_dict[actor_id][1] = 0

    # for actors in the dict that are not in visible_is, set visible + clicked status to 0    
    for clicked_actor in clicked_frame_dict:
        if clicked_actor not in visible_total:
            clicked_frame_dict[clicked_actor][1] = 0

    return clicked_frame_dict

def extract_relevant_objects(instance_seg_image, args):
    # only include vehicles, pedestrians, and 2 wheelers in the instseg mask
    relevant_object_values = [4,  # vehicles 
                              10, # pedestrians + pedestrian part of 2 wheelers
                              23] # vehicle part of 2 wheelers
    width, height = instance_seg_image.size
    mask = np.zeros((height, width, 1), dtype=np.uint8)
    instance_seg_image_r = np.array(instance_seg_image)[:, :, 0]
    for obj_val in relevant_object_values:
        mask[instance_seg_image_r == obj_val] = 255
    return mask

class SABalancedSampler(BatchSampler):
    """
    Samples elements from our imbalanced SA dataset
    We sample the same number of frames with unaware objects as with only unaware objects
    Not currently supported
    """
    def __init__(self, source_dataset, batch_size):
        raise NotImplementedError
        self.dataset = source_dataset
        self.unaware_indices = self.dataset.obj_unaware_dict.keys()
        self.aware_indices = set(self.dataset.obj_aware_dict.keys())
        self.batch_size = batch_size

    def __iter__(self):
        unaware_batch_size = self.batch_size // 2
        aware_batch_size = self.batch_size - unaware_batch_size

        # Oversample unaware class
        while len(self.aware_indices) >= aware_batch_size:
            # Sample remaining aware indices
            aware_idcs_batch = np.randperm.choice(list(self.aware_indices))
            self.aware_indices -= set(aware_idcs_batch)            

            # get matching num of unaware idcs
            unaware_idcs_batch = np.random.choice(self.unaware_indices, size=unaware_batch_size, replace=False)        
            batch = aware_idcs_batch + unaware_idcs_batch            
            # np.random.shuffle(batch)
            yield batch

        # if we are in the remainder of batches:
        batch = []

    def __len__(self):
        return len(self.data_source.index_mapping)

class SituationalAwarenessDataset(Dataset):
    #def __init__(self, images_dir, awareness_df, sensor_config, secs_of_history = 5, sample_rate = 4.0, gaussian_sigma = 10.0):
    def __init__(self, raw_data, sensor_config, episode, args):        
        self.raw_data_dir = Path(raw_data)
        self.episode = episode
        self.rgb_frame_delay = 3
        self.use_rgb = args.use_rgb
        self.middle_andsides = False
        self.instseg_channels = args.instseg_channels
        self.images_dir = Path(os.path.join(self.raw_data_dir, self.episode, "images"))
        aw_df_file_name = os.path.join(self.raw_data_dir, self.episode, 'rec_parse-awdata.json')
        
        self.awareness_df = pd.read_json(aw_df_file_name, orient='index')
        self.args = args

        self.sensor_config = configparser.ConfigParser()
        self.sensor_config.read(sensor_config)
        self.secs_of_history = args.secs_of_history 
        self.sample_rate = args.history_sample_rate
        self.gaussian_sigma = args.gaze_gaussian_sigma

        if self.args.gaze_format == 'blob':
            self.get_gaze_map_tensor = gaussian_contour_plot
        elif self.args.gaze_format == 'dot':
            self.get_gaze_map_tensor = gaze_dot_plot
        
        corrected_labels_df_filename = os.path.join(self.raw_data_dir, self.episode, 'corrected-awlabels.csv')
        self.corrected_labels_df = read_corrected_csv(corrected_labels_df_filename)

        self.clicked_frame_dict = get_clicked_frame_dict(self.corrected_labels_df)
        
        # read visible ids per image view (middle, left, right)
        # this is in RGB/instseg frame of reference
        # need to subtract rgb_frame_delay to match awareness_df frame of reference
        vis_per_view_filename = os.path.join(self.raw_data_dir, self.episode, 'visible_is_per_view.npy')        
        vis_per_view = np.load(vis_per_view_filename, allow_pickle=True)[0]
        vis_per_view_df = pd.DataFrame.from_dict(vis_per_view)
        vis_per_view_df['frame_no'] = vis_per_view_df['frame_no'].astype(int) - self.rgb_frame_delay # convert to awareness_df frame of reference
        vis_per_view_df = vis_per_view_df.set_index('frame_no') 
        self.vis_per_view_df = vis_per_view_df

        print(self.episode)

        if self.args.sample_clicks:
            # max time allowed since last click for valid sample
            self.sample_clicks_min_time = self.args.secs_of_history                
            # skip samples just before click (to account for reaction time)
            self.pre_clicks_excl_time = self.args.pre_clicks_excl_time

            # precompute userinput frame nos
            user_input_arr = np.array(self.awareness_df.AwarenessData_UserInput)
            user_input_locs = np.nonzero(user_input_arr)[0] # returning 
            user_input_diffs = np.diff(user_input_locs)
            user_input_diffs = np.insert(user_input_diffs, 0, user_input_locs[0])
            self.user_input_locs = user_input_locs[user_input_diffs > 7] # 6 clicks occur together for one press
            self.prev_click_idx = 0
            self.prev_click_time = self.awareness_df.iloc[self.user_input_locs[self.prev_click_idx]]['TimeElapsed']
            self.next_click_idx = 0
            self.next_click_time = self.awareness_df.iloc[self.user_input_locs[self.next_click_idx]]['TimeElapsed']

        index_mapping = {}
        self.class_counts_dict = {}        
        self.obj_aware_dict = {}
        self.obj_unaware_dict = {}

        idx = 0
            
        instance_seg_dir = Path(os.path.join(self.raw_data_dir, self.episode, "images", "instance_segmentation_output"))
        label_correction_end_buffer = -50
        instance_seg_imgs = sorted(os.listdir(instance_seg_dir), key = lambda x: int(x.split('.')[0]))[:label_correction_end_buffer]
        
        first_valid_idx = self.awareness_df[self.awareness_df["TimeElapsed"] > self.secs_of_history].index[0]
        class_distribution = np.array([0,0])
        skip_ctr = 0

        # start from above index here and create index mapping to frames
        for i in range(first_valid_idx, len(instance_seg_imgs)):
            file_name = instance_seg_imgs[i] # '000001.png' for example where 1 is the frame num
            frame_num = int(file_name.split('.')[0])        

            # check that target (label mask) is not empty
            # corrected df idcs 1 removed from aw df
            cur_corrected_row = self.corrected_labels_df.iloc[frame_num - self.rgb_frame_delay - 1]            
            actor_IDs = cur_corrected_row['visible_is']
            valid_vis = actor_IDs != 0
            num_objs_vis = np.sum(valid_vis)
            if num_objs_vis == 0:
                skip_ctr += 1
                continue

            # check if middle image has objects otherwise skip
            if not self.middle_andsides:
                vis_per_view_df_row = self.vis_per_view_df.loc[frame_num]
                all_vis_mid = np.concatenate((vis_per_view_df_row['iv_vis_mid'],
                                            vis_per_view_df_row['ip_vis_mid']))
                if len(all_vis_mid) == 0:
                    skip_ctr += 1
                    continue
                elif all_vis_mid[0] == 0 and len(all_vis_mid) == 1: # sometimes the first vehicle is 0 but these are bg vehicles
                    skip_ctr += 1
                    continue

            # if we want to sample only around clicks, 
            if self.args.sample_clicks:                
                # find previous and next click
                # idx = np.searchsorted(self.user_input_locs, frame_num - self.rgb_frame_delay)
                # returned index i satisfies a[i-1] < v <= a[i]

                cur_click_time = self.awareness_df.iloc[frame_num - self.rgb_frame_delay]['TimeElapsed']
                
                if cur_click_time > self.next_click_time:
                    self.prev_click_idx = self.next_click_idx
                    self.prev_click_time = self.next_click_time
                    
                    if self.next_click_idx == len(self.user_input_locs)-1:
                        pass
                    else:
                        self.next_click_idx += 1
                        self.next_click_time = self.awareness_df.iloc[self.user_input_locs[self.next_click_idx]]['TimeElapsed']                    

                if self.args.sample_clicks == 'both':
                    if self.next_click_time - cur_click_time < self.pre_clicks_excl_time:
                        if cur_click_time - self.prev_click_time < self.sample_clicks_min_time: # okay even if this is negative
                            # sample
                            pass
                        else:
                            skip_ctr +=1
                            continue
                    elif cur_click_time - self.prev_click_time >= self.sample_clicks_min_time:
                        # sample
                        skip_ctr +=1
                        continue

                elif self.args.sample_clicks == 'post_click':
                    if cur_click_time - self.prev_click_time < self.sample_clicks_min_time: # okay even if this is negative
                        # sample
                        pass
                    else:
                        skip_ctr +=1
                        continue

                elif self.args.sample_clicks == 'pre_excl':
                    if self.next_click_time - cur_click_time < self.pre_clicks_excl_time:                
                        skip_ctr +=1
                        continue
                    else:
                        # sample
                        pass                

            # calculate aware/unaware distribution
            valid_actors = actor_IDs[valid_vis]
            vis_total = cur_corrected_row['visible_total']
            vis_idcs_forlabel = np.in1d(vis_total, valid_actors)
            cur_awlabels = cur_corrected_row['awareness_label'][vis_idcs_forlabel]            
            num_vis_aware = np.count_nonzero(cur_awlabels)
            num_vis_unaware = cur_awlabels.size - num_vis_aware
            class_distribution += [num_vis_unaware, num_vis_aware]
        
            if num_vis_unaware > 0:
                self.obj_unaware_dict[idx] = num_vis_unaware 
            else:
                self.obj_aware_dict[idx] = num_vis_aware
            self.class_counts_dict[idx] = [num_vis_unaware, num_vis_aware]

            # add valid frame to index mapping
            index_mapping[idx] = frame_num            
            idx += 1
        
        self.index_mapping = index_mapping

        self.sample_weights = self._calc_sample_weights()

        print("Class distribution for episode %s: %s" % (self.episode, class_distribution))
        print("Num frames with atleast 1 unaware: %d, fully aware: %d" % (len(self.obj_unaware_dict), len(self.obj_aware_dict)))
        print("Number of frames skipped: %d" % skip_ctr)
    
    def _calc_sample_weights(self, data_balance_type='object'):
        """
        We use databalancing at the object or pixel level to ensure good representation of unaware data
        """
        if data_balance_type=="object":
            unaware_batch_split = 0.5
            aware_batch_split = 1 - unaware_batch_split
            unaware_images_weight = unaware_batch_split / len(self.obj_unaware_dict)
            aware_images_weight = aware_batch_split / len(self.obj_aware_dict)
            weights = []
            for idx in self.index_mapping.keys():
                num_vis_unaware, num_vis_aware =  self.class_counts_dict[idx]
                if num_vis_unaware > 0:
                    weights +=[unaware_images_weight]
                else:
                    weights +=[aware_images_weight]
            
            assert np.isclose(sum(weights), 1, atol= 1e-5)
            
            return weights
        else:
            raise NotImplementedError        

    def get_sample_weights(self):
        return self.sample_weights

    
    def get_ignore_mask(self, inst_img, visible_ids, awareness_labels, offset, frame_num):
        width, height = inst_img.size
        num_channels = 2 # aware, not aware
        mask = np.ones((height, width, num_channels), dtype=np.uint8)
        raw_img = np.array(inst_img)
        b = raw_img[:, :, 2]
        g = raw_img[:, :, 1]
        # Calculate the sum of b*256 + g
        sum_bg = (b * 256) + g
        
        current_time = self.awareness_df["TimeElapsed"][frame_num - self.rgb_frame_delay]
        end_time = current_time - self.secs_of_history
        frame_histsecs_ago = frame_num
        while frame_histsecs_ago > 0 and self.awareness_df["TimeElapsed"][frame_histsecs_ago - self.rgb_frame_delay] > end_time:
            frame_histsecs_ago -= 1
        
        for id_idx, id in enumerate(visible_ids):
            run_id = id - offset
            if awareness_labels[id_idx] == 1:
                if id in self.clicked_frame_dict:
                    clicked_frame_history = self.clicked_frame_dict[id][0] # clicked frame dict is in the awareness_df frame of reference
                    temp_arr = frame_num - np.array(clicked_frame_history)
                    if len(temp_arr[temp_arr >= 0]) > 0:
                        most_recent_click = np.min(temp_arr[temp_arr >= 0])    
                        if most_recent_click > frame_num - frame_histsecs_ago:
                        # Create a mask where sum_bg is equal to target_value
                            mask[sum_bg == int(run_id), 0] = 0
                            mask[sum_bg == int(run_id), 1] = 0
        if self.args.seg_mode == 'multiclass':           
            mask = mask[...,0]
            mask = mask[..., None]

        return mask

    def get_corrected_label_mask(self, inst_img, visible_ids, awareness_labels, offset):
        width, height = inst_img.size
        if self.args.seg_mode == 'multiclass':
            num_channels = 3 # aware, not aware, background/ignore
        elif self.args.seg_mode == 'multilabel':
            num_channels = 2 # aware, not aware
        mask = np.zeros((height, width, num_channels), dtype=np.uint8)
        #mask = np.zeros((width, height))
        raw_img = np.array(inst_img)
        b = raw_img[:, :, 2]
        g = raw_img[:, :, 1]
        # Calculate the sum of b*256 + g
        sum_bg = (b * 256) + g
        
        for id_idx, id in enumerate(visible_ids):
            run_id = id - offset
            if awareness_labels[id_idx] == 1:
                # Create a mask where sum_bg is equal to target_value
                mask[sum_bg == int(run_id), 0] = 255
            else:
                mask[sum_bg == int(run_id), 1] = 255
        
        if self.args.seg_mode == 'multiclass': # background channel
            mask[np.sum(mask[...,:2], axis=2) == 0, 2] = 255
            mask = np.argmax(mask, axis=2) #mask shape [600, 800, 3] --> [600, 800]
        
        return mask
    
    def get_full_label_mask(self, inst_img, id_list, offset, aw_visible, aw_answer, user_input, type_bit=16):
        width, height = inst_img.size
        mask = np.zeros((height, width), dtype=np.uint8)
        raw_img = np.array(inst_img)
        b = raw_img[:, :, 2]
        g = raw_img[:, :, 1]
        # Calculate the sum of b*256 + g
        sum_bg = (b * 256) + g
        
        for id in id_list:
            run_id = id - offset
            id_idx = aw_visible.index(id)
            label = False
            if (user_input & type_bit == aw_answer[id_idx] & type_bit) and (user_input & aw_answer[id_idx]):
                label = True
            if label == True:
                # Create a mask where sum_bg is equal to target_value
                mask[sum_bg==run_id] = 100
            else:
                mask[sum_bg==run_id] = 200
        
        return mask

    def get_gaze_point(self, focus_hit_pt, loc, rot):
        FOV = int(self.sensor_config['rgb']['fov'])
        w = int(self.sensor_config['rgb']['width'])
        h = int(self.sensor_config['rgb']['height'])
        F = w / (2 * np.tan(FOV * np.pi / 360))

        cam_info = {
            'w' : w,
            'h' : h,
            'fy' : F,
            'fx' : 1.0 * F,
        }

        K = np.array([
        [cam_info['fx'], 0, cam_info['w']/2],
        [0, cam_info['fy'], cam_info['h']/2],
        [0, 0, 1]])
        
        focus_hit_pt_scaled = np.array(focus_hit_pt.squeeze())/100 # conversion cm to m 
        vehicle_loc = carla.Location(*(loc.squeeze()))/100
        vehicle_rot = carla.Rotation(*(rot.squeeze()))
        vehicle_transform = carla.Transform(location=vehicle_loc, rotation=vehicle_rot)

        return world2pixels(focus_hit_pt_scaled, vehicle_transform, K, self.sensor_config)


    def __getitem__(self, idx):
        # Return rgb_img, instance_segmentation_img, gaze_heatmap 
        # (read in raw gaze and construct heatmap in get_item itself), label mask image
        frame_num = self.index_mapping[idx]

        # init data paths
        if self.use_rgb:        
            rgb_image = Image.open(self.images_dir / 'rgb_output' / ('%.6d.png' % frame_num)).convert('RGB')
             
        instance_seg_image = Image.open(self.images_dir / 'instance_segmentation_output' / ('%.6d.png' % frame_num)).convert('RGB')

        visible_total = self.corrected_labels_df['visible_total'][frame_num-self.rgb_frame_delay - 1] # -1 because corrected df is one frame shifted from awareness df
        awareness_label = self.corrected_labels_df['awareness_label'][frame_num-self.rgb_frame_delay - 1] # -1 because corrected df is one frame shifted from awareness df

        with open("%s/offset.txt" % self.images_dir, 'r') as file:
            offset = int(file.read()) 
        
        full_label_mask = self.get_corrected_label_mask(instance_seg_image, visible_total, awareness_label, offset=offset)
        # print("mask unique values")
        # print(np.unique(full_label_mask))

        if self.args.ignore_oldclicks:
            ignore_mask = self.get_ignore_mask(instance_seg_image, visible_total, awareness_label, offset, frame_num)
        else:
            ignore_mask = np.ones_like(full_label_mask)
            if ignore_mask.ndim == 2: #  add a channel dimension even if this is a single channel mask
                ignore_mask = ignore_mask[..., np.newaxis]
        
        # Construct gaze heatmap
        raw_gaze_mid= []
        
        current_time = self.awareness_df["TimeElapsed"][frame_num - self.rgb_frame_delay]
        end_time = current_time - self.secs_of_history
        history_frames = []
        history_frame_times = []
        i = 0
        frame = frame_num - i - 1
        while frame > 0 and self.awareness_df["TimeElapsed"][frame - self.rgb_frame_delay] > end_time:
            history_frames.append(frame)
            history_frame_times.append(self.awareness_df["TimeElapsed"][frame - self.rgb_frame_delay])
            frame = frame_num - i - 1
            i+=1
        
        assert self.secs_of_history*self.sample_rate == int(self.secs_of_history*self.sample_rate), "sample_rate x secs_of_history must be an integer"
        total_history_frames = int(self.secs_of_history*self.sample_rate)
        frame_time = 1/self.sample_rate
        step = 0

        frame = frame_num - 1
        raw_gaze_mid = np.zeros((total_history_frames+1, 2), dtype=np.int32)-1
        
        # use the vehicle (and hence cameras') current location and rotation 
        loc = self.awareness_df["EgoVariables_VehicleLoc"][frame_num - self.rgb_frame_delay]
        loc = np.asarray([loc[0], loc[1], loc[2]])
        rot = self.awareness_df["EgoVariables_VehicleRot"][frame_num - self.rgb_frame_delay]
        rot = np.asarray([rot[0], rot[1], rot[2]])


        while frame - self.rgb_frame_delay > 0 and step != total_history_frames:

            target_time = self.awareness_df["TimeElapsed"][frame - self.rgb_frame_delay] - frame_time*step
            closest_frame_indices_ranking = np.argsort(abs(np.array(history_frame_times)-target_time))
            closest_frame_ranking = [history_frames[i] for i in closest_frame_indices_ranking]
            
            closest_frame = 0
            for f in range(len(closest_frame_ranking)):
                if os.path.exists(self.images_dir / 'instance_segmentation_output' / \
                                  ('%.6d.png' % (closest_frame_ranking[f] + self.rgb_frame_delay))):
                    closest_frame = closest_frame_ranking[f]
                    break
            step += 1

            focus_hit_pt_i = np.asarray(self.awareness_df["FocusInfo_HitPoint"][closest_frame])
            
            pts_mid, pts_left, pts_right = self.get_gaze_point(focus_hit_pt_i, loc, rot)
            

            # don't append gaze if its outside the image            
            if pts_mid[0] >= 0 and pts_mid[0] < instance_seg_image.width and \
                pts_mid[1] >= 0 and pts_mid[1] < instance_seg_image.height:
                raw_gaze_mid[step,:] = pts_mid
            
        gaze_heatmap = self.get_gaze_map_tensor(np.array(instance_seg_image), raw_gaze_mid, sigma=self.gaussian_sigma, gaze_fade=self.args.gaze_fade)
        
        
        # Convert all images to tensors
        if self.use_rgb:
            rgb_image = transforms.functional.to_tensor(rgb_image) # only use this function to convert PIL images to tensors, normalizes between 0 and 1
        
        inst_img_metric_transform = transforms.Compose([transforms.PILToTensor()])
        instance_seg_image_metrics = inst_img_metric_transform(instance_seg_image).float()
        
        # change instance segmentation images
        if self.instseg_channels == 1:
            instance_seg_image = extract_relevant_objects(instance_seg_image, self.args)
            instance_seg_image = transforms.functional.to_tensor(instance_seg_image)
        if self.instseg_channels > 1:
            instance_seg_image = transforms.functional.to_tensor(instance_seg_image)[:self.instseg_channels, :, :]
        
        label_mask_image = transforms.functional.to_tensor(full_label_mask)
        #print(label_mask_image.shape)
        ignore_mask = torch.from_numpy(ignore_mask)

        if self.use_rgb:
            input_images = (rgb_image, instance_seg_image, gaze_heatmap)
        else:
            input_images = (instance_seg_image, gaze_heatmap)
        final_input_image = torch.cat(input_images)
 
        final_label_mask_image = label_mask_image
        
        final_ignore_mask = ignore_mask

        final_inst_images_for_metrics = instance_seg_image_metrics

        # padded_tensor = torch.nn.functional.pad(final_input_image, (0, 0, 4, 4), mode='constant', value=0)
        # padded_label_mask_image_tensor = torch.nn.functional.pad(final_label_mask_image, (0, 0, 4, 4), mode='constant', value=0)
        # padded_final_ignore_mask = torch.nn.functional.pad(final_ignore_mask.permute(2, 0, 1), (0, 0, 4, 4), mode='constant', value=0)
        
        # if self.use_rgb:
        #     padded_rgb_image = torch.nn.functional.pad(rgb_image, (0, 0, 4, 4), mode='constant', value=0)

        # padded_instance_seg_image = torch.nn.functional.pad(instance_seg_image, (0, 0, 4, 4), mode='constant', value=0)
        # padded_gaze_heatmap = torch.nn.functional.pad(gaze_heatmap, (0, 0, 4, 4), mode='constant', value=0)

        # padded_inst_images_for_metrics = torch.nn.functional.pad(final_inst_images_for_metrics, (0, 0, 4, 4), mode='constant', value=0)

        # data = {
        #     'input': padded_tensor,
        #     'label': padded_label_mask_image_tensor,
        #     'ignore_mask': padded_final_ignore_mask,
        #     'gaze_heatmap': padded_gaze_heatmap,
        #     'instance_seg': padded_instance_seg_image,
        #     'inst_metrics': padded_inst_images_for_metrics
        # }

        # if self.use_rgb:
        #     data['rgb'] = padded_rgb_image
        
    
        data = {
            'input': final_input_image,
            'label': final_label_mask_image,
            'ignore_mask': final_ignore_mask.permute(2, 0, 1),
            'gaze_heatmap': gaze_heatmap,
            'instance_seg': instance_seg_image,
            'inst_metrics': final_inst_images_for_metrics
        }

        return data


        #return padded_tensor, padded_label_mask_image_tensor, padded_final_ignore_mask

    def __len__(self):        
        return len(self.index_mapping)

