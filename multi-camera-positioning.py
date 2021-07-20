import sys
import math
import time
import os

import numpy as np
from numpy import linalg as LA

import cv2
from scipy.spatial import distance
from openvino.inference_engine import IECore
from munkres import Munkres               # Hungarian algorithm for ID assignment

omz_root = os.environ['OMZ_ROOT'] if 'OMZ_ROOT' in os.environ else '.'

'''
# Pedestrian detection
#model_det  = '{0}/intel/{1}/FP16/{1}'.format(omz_root, 'pedestrian-detection-adas-0002')
#model_reid = '{0}/intel/{1}/FP16/{1}'.format(omz_root, 'person-reidentification-retail-0277')
'''

#'''
# Face detection
model_det  = '{0}/intel/{1}/FP16/{1}'.format(omz_root, 'face-detection-adas-0001')
model_reid = '{0}/intel/{1}/FP16/{1}'.format(omz_root, 'face-reidentification-retail-0095')
#'''

print(model_det, model_reid)

_N = 0
_C = 1
_H = 2
_W = 3

num_cameras = 2
cam_attr = [ [(10,10), 180-60, 60], [(640-10,10), 180+60, 60] ]    # (x,y),angle,fov
_dist = 1200

def calc_angle(camera_base_angle, fov, position_x, image_width):
	return (position_x / image_width - 0.5) * fov + camera_base_angle

def TBangle0(x):   # Event handler for track bar 0
	global cam_attr
	cam_attr[0][1] = x

def TBangle1(x):   # Event handler for track bar 1
	global cam_attr
	cam_attr[1][1] = x

def mouseEvent(event, x, y, flags, param):  # Mouse button event handler
	global cam_attr
	if event == cv2.EVENT_LBUTTONDOWN:
		cam_attr[0][0] = (x,y)
	if event == cv2.EVENT_RBUTTONDOWN:
		cam_attr[1][0] = (x,y)

def calc_radial_pos(pos, angle, length):
	return (int(pos[0]+math.sin(math.radians(angle))*length), int(pos[1]-math.cos(math.radians(angle))*length))

def draw_radial_line(img, pos, angle, length, color, width=2):
	ipos=(int(pos[0]), int(pos[1]))
	cv2.line(img, ipos, calc_radial_pos(ipos, angle, length), color, width)

def draw_camera(img, pos, angle, id):
	cv2.circle(img, pos, 30, (0,255,0),-1)
	draw_radial_line(img, pos, angle, 30, (0,0,0), 8)
	cv2.putText(img, str(id), (pos[0], pos[1]+15), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,0,255), 1) 



def line(p1, p2):
	A = (p1[1] - p2[1])
	B = (p2[0] - p1[0])
	C = (p1[0]*p2[1] - p2[0]*p1[1])
	return A, B, -C

def intersection(L1, L2):
	D  = L1[0] * L2[1] - L1[1] * L2[0]
	Dx = L1[2] * L2[1] - L1[1] * L2[2]
	Dy = L1[0] * L2[2] - L1[2] * L2[0]
	x = Dx / D
	y = Dy / D
	return x,y

def intersection_check(p1, p2, p3, p4):
	tc1 = (p1[0] - p2[0]) * (p3[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p3[0])
	tc2 = (p1[0] - p2[0]) * (p4[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p4[0])
	td1 = (p3[0] - p4[0]) * (p1[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p1[0])
	td2 = (p3[0] - p4[0]) * (p2[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p2[0])
	return tc1*tc2<0 and td1*td2<0

# line(point1)-(point2)
# convert a line to a vector
def line_vectorize(point1, point2):
	a = point2[0]-point1[0]
	b = point2[1]-point1[1]
	return [a,b]

# point = (x,y)
# line1(point1)-(point2), line2(point3)-(point4)
# calculate the angle between line1 and line2
def calc_vector_angle( point1, point2, point3, point4 ):
	u = np.array(line_vectorize(point1, point2))
	v = np.array(line_vectorize(point3, point4))
	i = np.inner(u, v)
	n = LA.norm(u) * LA.norm(v)
	c = i / n
	a = np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))
	if u[0]*v[1]-u[1]*v[0]<0:
		return a
	else:
		return 360-a



def main():
		global _dist

		cv2.namedWindow('pos')
		cv2.createTrackbar('cam0_angle', 'pos', cam_attr[0][1], 360, TBangle0)
		cv2.createTrackbar('cam1_angle', 'pos', cam_attr[1][1], 360, TBangle1)
		cv2.setMouseCallback('pos', mouseEvent)
		posimg=np.zeros((480,640,3), np.uint8)

		ie = IECore()

		# Prep for face/pedestrian detection
		net_det  = ie.read_network(model=model_det+'.xml', weights=model_det+'.bin')
		#                                                           model=pedestrian-detection-adas-0002
		input_name_det  = next(iter(net_det.input_info))            # Input blob name "data"
		input_shape_det = net_det.input_info[input_name_det].tensor_desc.dims      # [1,3,384,672]
		out_name_det    = next(iter(net_det.outputs))               # Output blob name "detection_out"
		out_shape_det   = net_det.outputs[out_name_det].shape       # [ image_id, label, conf, xmin, ymin, xmax, ymax ]
		exec_net_det    = ie.load_network(network=net_det, device_name='CPU', num_requests=1)

		# Preparation for face/pedestrian re-identification
		net_reid = ie.read_network(model=model_reid+".xml", weights=model_reid+".bin")
		#                                                             person-reidentificaton-retail-0079
		input_name_reid = next(iter(net_reid.input_info))           # Input blob name "data"
		input_shape_reid = net_reid.input_info[input_name_reid].tensor_desc.dims   # [1,3,160,64]
		out_name_reid    = next(iter(net_reid.outputs))             # Output blob name "embd/dim_red/conv"
		out_shape_reid   = net_reid.outputs[out_name_reid].shape    # [1,256,1,1]
		exec_net_reid    = ie.load_network(network=net_reid, device_name='CPU', num_requests=1)

		# Open USB webcams
		#webcam = [cv2.VideoCapture(i) for i in range(1, num_cameras+1)]
		webcam = [ cv2.VideoCapture(i) for i in [0,1] ]

		feature_repo_db = []

		prev_feature = [ 0 for i in range(256) ]
		objid=0

		while cv2.waitKey(1)!=27:                               # 27 == ESC
			images = [cam.read()[1] for cam in webcam]            # cv2.VideoCapture.read() returns [ ret, image]. Take only the image.
			feature_vec = [[] for i in range(num_cameras)]        # [cam[obj[{feature,coord,id}]]]
			for camera in range(num_cameras):
				in_frame = cv2.resize(images[camera], (input_shape_det[_W], input_shape_det[_H]))
				in_frame = in_frame.transpose((2, 0, 1))                                                  # Change data layout from HWC to CHW
				in_frame = in_frame.reshape(input_shape_det)
				res_det = exec_net_det.infer(inputs={input_name_det: in_frame})
				for obj in res_det[out_name_det][0][0]:             # obj = [ image_id, label, conf, xmin, ymin, xmax, ymax ]
					if obj[2] > 0.75:                                 # Confidence > 75% 
						frame = images[camera]
						xmin = abs(int(obj[3] * frame.shape[1]))
						ymin = abs(int(obj[4] * frame.shape[0]))
						xmax = abs(int(obj[5] * frame.shape[1]))
						ymax = abs(int(obj[6] * frame.shape[0]))
						class_id = int(obj[1])

						# Obtain feature vector of the detected object using re-identification model
						obj_img=frame[ymin:ymax,xmin:xmax].copy()                                    # Crop the found object
						obj_img=cv2.resize(obj_img, (input_shape_reid[_W], input_shape_reid[_H]))
						cv2.imshow('person', obj_img)
						obj_img=obj_img.transpose((2,0,1))
						obj_img=obj_img.reshape(input_shape_reid)
						res_reid = exec_net_reid.infer(inputs={input_name_reid: obj_img})            # Run re-identification model to generate feature vectors (256 elem)
						vec=np.array(res_reid[out_name_reid]).reshape((256))                         # Convert the feature vector to numpy array
						feature_vec[camera].append({'coord': [xmin,ymin, xmax,ymax]})
						feature_vec[camera][-1]['feature'] = vec
						feature_vec[camera][-1]['id'] = -1         # -1 = ID unassigned

			if len(feature_vec[0])==0 or len(feature_vec[1])==0:                               # skip if no face is found in either picture (or both)
				cv2.imshow('pos', posimg)
				for i in range(2):
					cv2.imshow('cam'+str(camera), images[camera])
				continue

			# create cosine-similarity matrix for Hungarian assignment algorithm
			cos_sim_matrix=[ [ distance.cosine(feature_vec[1][j]["feature"], feature_vec[0][i]["feature"]) 
													 for j in range(len(feature_vec[1]))] for i in range(len(feature_vec[0])) ]
			# solve ID assignment matrix problem by Hungarian algorithm
			hangarian = Munkres()
			combination = hangarian.compute(cos_sim_matrix)

			# assign ID to objects found based on assignment matrix
			objects_found=[]          # [id, (cam0x, cam0y), (cam1x, cam1y)]   coordinates=center of the detected object
			for obj0,obj1 in combination:
				tmpid = -1
				for feature_repo in feature_repo_db:
					dist = distance.cosine(feature_vec[0][obj0]['feature'], feature_repo['feature'])
					if dist<0.6:  # a similar object is found in the feature_repo_db
						tmpid = feature_repo['id']
						feature_repo['feature'] = feature_vec[0][obj0]['feature']  # update feature vector
						feature_repo['time']    = time.monotonic()                 # last spotted time in second
				if tmpid == -1:
					feature_repo_db.append({'id':objid, 'feature':feature_vec[0][obj0]['feature'], 'time':time.monotonic()})
					tmpid = objid
					objid +=1

				feature_vec[0][obj0]['id']=tmpid
				feature_vec[1][obj1]['id']=tmpid
				x11,y11,x12,y12 = feature_vec[0][obj0]['coord']
				x21,y21,x22,y22 = feature_vec[1][obj1]['coord']
				objects_found.append([feature_vec[0][obj0]['id'], ( (x12-x11)/2+x11, (y12-y11)/2+y11 ),
																													( (x22-x21)/2+x21, (y22-y21)/2+y21 )])  #calculate center of found objects

			# discard time out feature_repo
			now = time.monotonic()
			for feature_repo in feature_repo_db:
				if feature_repo['time']+10 < now:
					print("Discarded id {}".format(feature_repo['id']))
					feature_repo_db.remove(feature_repo)         # discard feature vector from DB

			# Draw bounding boxes
			for camera in range(num_cameras):
				for obj in feature_vec[camera]:
					id = obj['id']
					color = ( (((~id)<<6) & 0x100)-1, (((~id)<<7) & 0x0100)-1, (((~id)<<8) & 0x0100)-1 )
					xmin, ymin, xmax, ymax = obj['coord']
					cv2.rectangle(images[camera], (xmin, ymin), (xmax, ymax), color, 2)
					cv2.putText(images[camera], str(id), (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 1.0, color, 1)
				cv2.imshow('cam'+str(camera), images[camera])

			# Draw position
			cv2.rectangle(posimg,(0,0),(640-1,480*2-1),(0,0,0),-1)         # clear frame buffer
			for i in range(2):
				draw_camera(posimg, pos=cam_attr[i][0], angle=cam_attr[i][1], id=i)

			id=0
			for obj in objects_found:
				angle=[]
				angle.append(calc_angle(camera_base_angle=cam_attr[0][1], fov=cam_attr[0][2], position_x=obj[1][0], image_width=images[0].shape[1]))
				angle.append(calc_angle(camera_base_angle=cam_attr[1][1], fov=cam_attr[1][2], position_x=obj[2][0], image_width=images[1].shape[1]))
				draw_radial_line(img=posimg, pos=cam_attr[0][0], angle=angle[0], length=_dist, color=(127,127,127))
				draw_radial_line(img=posimg, pos=cam_attr[1][0], angle=angle[1], length=_dist, color=(127,127,127))

				# Check whether the lines intersect or not, and then draw lines and circles
				points = [ cam_attr[0][0], calc_radial_pos(pos=cam_attr[0][0],angle=angle[0],length=_dist),
									 cam_attr[1][0], calc_radial_pos(pos=cam_attr[1][0],angle=angle[1],length=_dist) ]
				if intersection_check(points[0], points[1], points[2], points[3]) == True:
					l1 = line([points[0][0], points[0][1]], [points[1][0], points[1][1]])
					l2 = line([points[2][0], points[2][1]], [points[3][0], points[3][1]]) 
					x, y = intersection( l1, l2 )
					cv2.circle(posimg, (int(x), int(y)), 10, (255,255,0))
					cv2.putText(posimg, str(obj[0]), (int(x)-20, int(y)-20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255,255,255), 1) 

					a=calc_vector_angle(points[0], points[1],  points[2], points[3]);   # calculate intersection angle of 2 lines.   line(points[0], points[1]) line(points[2], points[3])
				id+=1

			cv2.imshow('pos', posimg)

		cv2.destroyAllWindows()

if __name__ == '__main__':
		sys.exit(main() or 0)
