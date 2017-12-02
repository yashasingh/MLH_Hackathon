import cam_image

# script for data-set creation
a = cam_image.frames()
temp = a.check()
count = 0
while(temp):
	temp = a.check()
	img = a.capture_frames()
	proc = a.process()
	a.show_image(img, 'frame')
	a.show_image(proc, 'proc_frame')
	if(temp==2):
		print('saving!')
		a.save_image(count)
		count+=1

a.close()
"""