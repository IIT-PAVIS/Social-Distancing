# evaluation code

import json
import os
import numpy as np
from socialDistance_mask import main as compute_sd
from sklearn.metrics import precision_recall_fscore_support as measures
import sys

min_distance = 2
#horizontal_ratio = 0.5
#vertical_ratio = 0.8
Homography_option = 'rectified' #'GT', 'rectified', 'vh_params'
is_ablation = False
## put your base dir here
DATASET_DIR = '../dataset/' 
# gather available datasets
DATASETS = [os.path.join(DATASET_DIR, d) for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
print("Available Datasets list:")
for d in DATASETS:
	print("\t\t\t {}".format(d))
print("\n")
for horizontal_ratio in [0.6]:#[0.5, 0.6, 0.7, 0.8]:
	for vertical_ratio in [0.6]:#[0.5, 0.6, 0.7, 0.8]:

		for dataset in DATASETS:
			imgs_path = os.path.join(dataset, 'Images')
			joints_path = os.path.join(dataset, 'HumanJoints')
			distances_path = os.path.join(dataset, 'Annotations')
			calibrations_path = os.path.join(dataset, 'Calibrations')
			
			SEQUENCES = sorted([d for d in os.listdir(imgs_path) if os.path.isdir(os.path.join(imgs_path, d))])
			assert len(SEQUENCES) >= 1
			dataset_precision = []
			dataset_recall = []
			dataset_f1 = []
			for sequence in SEQUENCES:
				imgs_path_seq = os.path.join(imgs_path, sequence)
				joints_path_seq = os.path.join(joints_path, sequence)
				distances_path_seq = os.path.join(distances_path, sequence) + '_joints.json'
				calibration_path_seq = os.path.join(calibrations_path, sequence) + '.json'
			
				IMAGES = sorted([d for d in os.listdir(imgs_path_seq)])
				JOINTS = sorted([d for d in os.listdir(joints_path_seq)])
				assert len(IMAGES) == len(JOINTS)
				
				print("Reading {}".format(distances_path_seq))
				dist_file = open(distances_path_seq)
				dist_data = json.load(dist_file)
				#~ assert len(IMAGES) == len(dist_data.keys())  ## # this can fail since there are no annotations for frames with no people
				
				print("Reading {}".format(calibration_path_seq))
				calib_file = open(calibration_path_seq)
				calib_data = json.load(calib_file)

				gt_homography_matrix = np.asarray(calib_data['Homography'])
				gt_scale_factor = (calib_data['Scale'])
				assert gt_homography_matrix.shape == (3,3)

				sequence_precision = []
				sequence_recall = []
				sequence_f1 = []
				for ii, (img_name, joint_name) in enumerate(zip(IMAGES,JOINTS)): 
					img_full_path = os.path.join(imgs_path_seq, img_name)

					#print('image name is',img_name.split('.')[0])
					dataset_name = dataset.split('/')[-1]
					if dataset_name == 'Epfl-wildtrack':
						ii_name = img_name.split('.')[0]
					else:
						ii_name = '{:06}'.format(ii)
					
					### if no people in the image, there is no annotation from frame ii in the .json
					try:
						person_ind_list = dist_data[str(ii_name)]['PersonList']
					except:
						continue
									
					dist_matrix = dist_data[str(ii_name)]['DistanceMatrix']
					
					joints_file = open(os.path.join(joints_path_seq,joint_name))
					joints_data = json.load(joints_file)

					skeletal_joints = []
					for valid_index in person_ind_list:
					#for i in range(len(joints_data['people'])):
						joints = joints_data['people'][int(valid_index)]['pose_keypoints_2d']
						del joints[2::3]
						skeletal_joints.append(joints)
						#print('person in is:',len(person_ind_list))
						#print('all skeletal is:', len(skeletal_joints))
					
					dt_dist_map_bin = compute_sd(img_full_path,skeletal_joints,horizontal_ratio, vertical_ratio, Homography_option, gt_homography_matrix, gt_scale_factor,is_ablation)

					gt_dist_map_bin = np.zeros(len(dt_dist_map_bin))  
					
					for ind1 in range(len(person_ind_list)-1):
						for ind2 in range(ind1+1, len(person_ind_list)):
							
							val = dist_matrix[ind1][ind2]
							if val <= min_distance:
								gt_dist_map_bin[ind1] = 1
								gt_dist_map_bin[ind2] = 1

					frame_precision, frame_recall, frame_f1, _ = measures(gt_dist_map_bin,dt_dist_map_bin, average = 'binary')

					sequence_precision.append(frame_precision)
					sequence_recall.append(frame_recall)
					sequence_f1.append(frame_f1)
					dataset_precision.append(frame_precision)
					dataset_recall.append(frame_recall)
					dataset_f1.append(frame_f1)
				print('horizontal_ratio is:', horizontal_ratio)
				print('vertical_ratio is:', vertical_ratio)
				print('sequence %s precision is: %f' % (sequence, (sum(sequence_precision)/len(sequence_precision))))
				print('sequence %s recall is: %f' % (sequence , (sum(sequence_recall)/len(sequence_recall))))
				print('sequence %s f1 is: %f' % (sequence, (sum(sequence_f1)/len(sequence_f1))))

			print('dataset %s precision is: %f' % (dataset, (sum(dataset_precision)/len(dataset_precision))))
			print('dataset %s recall is: %f' % (dataset, (sum(dataset_recall)/len(dataset_recall))))
			print('dataset %s f1 is: %f' % (dataset, (sum(dataset_f1)/len(dataset_f1))))

print('done!')

