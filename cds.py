import numpy as np
import cv2,argparse,time,os
import tensorflow as tf

class video_process:
	
	def __init__(self,intensity,sensitivity,det_dir,verbose,graph,label):
		self.intensity= intensity
		self.sensitivity=sensitivity
		self.det_dir=det_dir
		self.verbose=verbose
		self.graph=graph
		self.label=label
		
		try:  
			os.mkdir(det_dir)
		except OSError:  
		
		    print "Creation of the directory %s failed" % det_dir
		else:  
		    print "Successfully created the directory %s" % det_dir

	def details(self):

		width= cap.get(3)	#CV_CAP_PROP_FRAME_WIDTH
		height= cap.get(4)	#CV_CAP_PROP_FRAME_HEIGHT
		self.fps= cap.get(5)	#CV_CAP_PROP_FPS
		self.frames= cap.get(7)	#CV_CAP_PROP_FRAME_COUNT
		cap.set(2,1)		#CAP_PROP_POS_AVI_RATIO,END
		self.length= cap.get(0)	#CV_CAP_PROP_POS_MSEC
		cap.set(2,0)		#reset pointer to initial position
		if self.verbose>0:
			print "Width:		",width
			print "Height:		",height
			print "FPS:		",self.fps
			print "Total Frames:	",self.frames
			print "Length:		",self.length/1000, "sec"

	def process(self):
		global cap
		score_array={}
		for ids in self.label_lines:
			score_array[ids]=0.0 
		#print score_array
		begin=time.time()
		with tf.Session() as sess:
			for frame_count in range(0,int(self.frames),self.intensity):
				if self.verbose>1:
					print "current frame",frame_count
				
				cap.set(1,frame_count-1)
				#print "Frame - ",cap.get(0)/self.fps
				retval, im = cap.read()
				image=im
				#cv2.imwrite("pics.jpeg",im)
				#time.sleep(0.1)
				#image_path = "pics.jpeg"
				start=time.time()
				#im= cv2.resize(im,dsize=(299,299), interpolation = cv2.INTER_CUBIC)
				#np_image_data = np.asarray(im)	#Numpy array
				#np_image_data=cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
				#np_final = np.expand_dims(np_image_data,axis=0)
				image_data = tf.gfile.FastGFile("pics.jpeg", 'rb').read()
				softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
				predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
				#predictions = sess.run(softmax_tensor,{'Mul:0': np_final})
	
							   
				# Sort to show labels of first prediction in order of confidence
				top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
				
				for node_id in top_k:
					human_string = self.label_lines[node_id]
					score = predictions[0][node_id]
					score_array[human_string]=score_array[human_string]+score
					net=(('%s (score = %.5f)' % (human_string, score)))
					if score>sensitivity:
						
						cv2.imwrite(det_dir+human_string+"- "+str(score)+".jpeg",image)
						if self.verbose==3:
							print "Image saved to",det_dir,"as",human_string+"- "+str(score)+".jpeg"
					#print net
				if self.verbose>2:
					print "Time taken to scan this frame",(time.time()-start)
			if self.verbose>1:
				print "....................................................."
				for ids in score_array:
					print ids,":", score_array[ids]/(self.frames/self.intensity)*100,"%"
				print "-----------------------------------------------------"
				print "Total time taken:	",time.time()-begin
			
			
	def loader(self):
		self.label_lines = [line.rstrip() for line in tf.gfile.GFile(self.label)]
		# Unpersists graph from file
		with tf.gfile.FastGFile(self.graph, 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			_ = tf.import_graph_def(graph_def, name='')
			
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--video',type=str,default='', help='Path to video file')
	parser.add_argument('--graph',type=str,default='output_graph.pb', help='Path to graph')
	parser.add_argument('--labels',type=str,default='output_labels.txt', help='Path to labels')
	parser.add_argument('--verbosity',type=int,default=1, help='Add verbosity. Range from 0-3. 0 being least verbose and 3 being the most.')
	parser.add_argument('--sensitivity',type=float,default=0.4, help='Set the threshold. Values range from 0.1 to 1.')
	parser.add_argument('--det_dir',type=str,default='blacklist/', help='Path to the directory where the blacklisted frames need to be stored')
	parser.add_argument('--intensity',type=int,default=4, help="Intesity with which the video needs to be scanned, Ranges from 1-100. 1 being the most intense, scans every frame. 100 being the lease intense,scans every 100th frame")
	
	intensity= parser.parse_args().intensity
	sensitivity=parser.parse_args().sensitivity
	det_dir=parser.parse_args().det_dir
	verbose=parser.parse_args().verbosity
	graph=parser.parse_args().graph
	label=parser.parse_args().labels
	vp = video_process(intensity,sensitivity,det_dir,verbose,graph,label)
	path= parser.parse_args().video

cap = cv2.VideoCapture(path)
vp.loader()
vp.details()
vp.process()
