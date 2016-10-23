import cv2 
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold='nan')


def closewindows():
	k = cv2.waitKey(0)
	if k & 0xFF == ord('s'):
		comment = input("Comment:-\n ")
		cv2.imwrite('./data/test_result/'+comment+'_thres'+'.jpg',final_thr)
		cv2.imwrite('./data/test_result/'+comment+'_src'+'.jpg',src_img)
		cv2.imwrite('./data/test_result/'+comment+'_contr'+'.jpg',final_contr)
		print("Completed")
	elif k & 0xFF == int(27):
		cv2.destroyAllWindows()
	else:
		closewindows()

def line_array(array):
	list_x = []
	for y in range(len(array)):
		if all(i >= 3 for i in array[y:y+9]) == True:
			list_x.append(y-1)
	return list_x


#-------------Thresholding Image--------------#

src_img = cv2.imread('./data/img_2.jpg', 1)
# copy = src_img.copy()
# src_img = cv2.resize(copy, dsize =(1500, 1000), interpolation = cv2.INTER_AREA)
height = src_img.shape[0]
width = src_img.shape[1]
print("#----------------------------#")
print("Image Info:-")
print("Height =",height,"\nWidth =",width)
print("#----------------------------#")

grey_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

gud_img = cv2.adaptiveThreshold(grey_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,101,2)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
noise_remove = cv2.erode(gud_img,kernel,iterations = 2)

kernel1 = np.array([[1,0,1],[0,1,0],[1,0,1]], dtype = np.uint8)

opening = cv2.morphologyEx(gud_img, cv2.MORPH_OPEN, kernel, iterations = 2) # To remove "pepper-noise"
kernel1 = np.array([[1,0,1],[0,1,0],[1,0,1]], dtype = np.uint8)
final_thr = cv2.dilate(noise_remove,kernel1,iterations = 3)

#-------------/Thresholding Image-------------#


#-------------Line Detection------------------#

count_x = np.zeros(shape= (height))
for y in range(height):
	for x in range(width):
		if noise_remove[y][x] == 255 :
			count_x[y] = count_x[y]+1
	# print(count_x[y])

line_list = line_array(count_x)
# print(line_list) 

# t = np.arange(0,height, 1)
# plt.plot(t, count_x[t])
# plt.axis([0, height, 0, 350])


# for y in range(len(line_list)):
# 	if :
# 		main_list.append(line_list[y-1]+10)
# 		main_list.append(line_list[y]-5)

# main_list.append(line_list[-1]+10)

# print(main_list)

# for y in main_list:
# 	src_img[y][:] = (0 , 255,0)



# final_thr = cv2.erode(final_thr,kernel1,iterations = 1)


#----------------------------------------------------------#
 






#-------------------------------------------------------------#


#-------------Character segmenting------------#

chr_img = final_thr.copy()

contr_img, contours, hierarchy = cv2.findContours(chr_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# print(len(contours))
final_contr = np.zeros((final_thr.shape[0],final_thr.shape[1],3), dtype = np.uint8)
cv2.drawContours(final_contr, contours, -1, (0,255,0), 3)

for cnt in contours:
	if cv2.contourArea(cnt) > 100:
		x,y,w,h = cv2.boundingRect(cnt)
		cv2.rectangle(src_img,(x,y),(x+w,y+h),(0,255,0),2)

#-------------/Character segmenting-----------# 





#-------------Displaying Image----------------#

cv2.namedWindow('Source Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Threshold Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Contour Image', cv2.WINDOW_NORMAL)

cv2.imshow("Source Image", src_img)
cv2.imshow("Threshold Image", final_thr)
cv2.imshow("Contour Image", final_contr)

# plt.show()

#-------------/Displaying Image---------------#


#-------------Closing Windows-----------------#

closewindows()