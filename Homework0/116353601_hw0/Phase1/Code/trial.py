import numpy as np
import scipy.stats as st
import skimage.transform
import matplotlib.pyplot as plt
import cv2
import scipy
import random

def gabor_fn(sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3 # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation 
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb



def gaussian1d(sigma, mean, x, ord):
    x = np.array(x)
    x_ = x - mean
    var = sigma**2

    # Gaussian Function
    g1 = (1/np.sqrt(2*np.pi*var))*(np.exp((-1*x_*x_)/(2*var)))

    if ord == 0:
        g = g1
        return g
    elif ord == 1:
        g = -g1*((x_)/(var))
        return g
    else:
        g = g1*(((x_*x_) - var)/(var**2))
        return g

def gaussian2d(sup, scales):
    var = scales * scales
    shape = (sup,sup)
    n,m = [(i - 1)/2 for i in shape]
    x,y = np.ogrid[-m:m+1,-n:n+1]
    g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
    return g

def log2d(sup, scales):
    var = scales * scales
    shape = (sup,sup)
    n,m = [(i - 1)/2 for i in shape]
    x,y = np.ogrid[-m:m+1,-n:n+1]
    g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
    h = g*((x*x + y*y) - var)/(var**2)
    return h

def makefilter(scale, phasex, phasey, pts, sup):

    gx = gaussian1d(3*scale, 0, pts[0,...], phasex)
    gy = gaussian1d(scale,   0, pts[1,...], phasey)

    image = gx*gy

    image = np.reshape(image,(sup,sup))
    return image

def makeLMfilters():
    sup     = 49
    scalex  = np.sqrt(2) * np.array([1,2,3])
    norient = 6
    nrotinv = 12

    nbar  = len(scalex)*norient
    nedge = len(scalex)*norient
    nf    = nbar+nedge+nrotinv
    F     = np.zeros([sup,sup,nf])
    hsup  = (sup - 1)/2

    x = [np.arange(-hsup,hsup+1)]
    y = [np.arange(-hsup,hsup+1)]

    [x,y] = np.meshgrid(x,y)

    orgpts = [x.flatten(), y.flatten()]
    orgpts = np.array(orgpts)

    count = 0
    for scale in range(len(scalex)):
        for orient in range(norient):
            angle = (np.pi * orient)/norient
            c = np.cos(angle)
            s = np.sin(angle)
            rotpts = [[c+0,-s+0],[s+0,c+0]]
            rotpts = np.array(rotpts)
            rotpts = np.dot(rotpts,orgpts)
            F[:,:,count] = makefilter(scalex[scale], 0, 1, rotpts, sup)
            F[:,:,count+nedge] = makefilter(scalex[scale], 0, 2, rotpts, sup)
            count = count + 1

    count = nbar+nedge
    scales = np.sqrt(2) * np.array([1,2,3,4])

    for i in range(len(scales)):
        F[:,:,count]   = gaussian2d(sup, scales[i])
        count = count + 1

    for i in range(len(scales)):
        F[:,:,count] = log2d(sup, scales[i])
        count = count + 1

    for i in range(len(scales)):
        F[:,:,count] = log2d(sup, 3*scales[i])
        count = count + 1

    return F



def gkern(kernlen, nsig):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


def DoG(scales,orient,size):
    # scale=range(1,scales+1)
    # print(scale)
    orients=np.linspace(0,360,orient)
    # kernels=[[0 for x in range(1,scales)]for y in range(1,orient)]
    DoG_stack = list()
    for each_scale in scales:
    	for each_size in size:
	        kernel=gkern(each_size,each_scale)
	        border = cv2.borderInterpolate(0, 1, cv2.BORDER_CONSTANT)
	        sobelx64f = cv2.Sobel(kernel,cv2.CV_64F,1,0,ksize=5, borderType=border)
	        for index,eachOrient in enumerate(orients):
	            # plt.figure(figsize=(16,2))
	            image=skimage.transform.rotate(sobelx64f,eachOrient)
	            DoG_stack.append(image)

	            # plt.subplots_adjust(hspace=0.1,wspace=1.5)
	            # plt.subplot(scales,orient,index+1)
	            # plt.imshow(image,cmap='binary')
	            # plt.show()
	return DoG_stack


def single_half_disk(radius):
    two_r_plus_1 = radius*2 + 1
    half_disk = np.ones([two_r_plus_1, two_r_plus_1])

    rs = np.power(radius,2)
    for i in range(radius):
        iss = np.power((i - radius),2)
        
        for j in range(two_r_plus_1):
            if (iss+ np.power((j - radius),2) < rs):
                half_disk[i, j] = 0
    return half_disk
 #    half_disk = half_disk(20)
	# plt.imshow(half_disk, cmap = 'binary')
	# plt.show()
    


def half_disk_bank(radius_list,orient):
    orients=np.linspace(0,360,orient)
   	half_disk_bank_op = list()
    for each_rad in radius_list:
		one_half_disk=single_half_disk(each_rad)
        for eachOrient in orients:
            image=skimage.transform.rotate(one_half_disk,eachOrient,cval=1)
            half_disk_bank_op.append(image)
    return half_disk_bank_op
 #    radius_list = [7,20,35]
	# orient = 4
	# half_disk = half_disk_bank(radius_list,orient)
	# for each in half_disk:
	#     plt.imshow(each,cmap='binary')
	#     plt.show()


def gabor_filter_list(num_filters):
	# gabor_filter = list()
	# sigma_used_list = list()
	# theta_used_list = list()
	# Lambda_used_list = list()
	# psi_used_list = list()
	for each in range(1,num_filters):
		sigma = random.randint(3,6)
		theta = random.uniform(0,3.14)
		Lambda = random.randint(3,10)
		psi = random.randint(3,14)
		# gamma = random.randint(3,6)
		gamma = 1
		gabor_filter.append(gabor_fn(sigma,theta,Lambda,psi,gamma))
	return gabor_filter


def filter_bank():
	scale_list = [3,5,7,9]
	size_list = [7,11,15]
	orient = 16
	DoG_filters = DoG(scales=scale_list, orient = orient,size=size_list)
	LM_filters = makeLMfilters()
	gabor_filters = gabor_filter_list(10)
	return DoG_filters, LM_filters, gabor_filters



def texton_tensor(Img_gray):
    N_dim_tensor_dog = Img_gray
    N_dim_tensor_lm = Img_gray
    N_dim_tensor_gabor = Img_gray
    DoG_filters, LM_filters, gabor_filters = filter_bank()
    
    for each in range(len(DoG_filters)):
        # kernel_op = cv2.filter2D(Img_gray,-1,filter_bank[:,:,each])
        kernel_op = cv2.filter2D(Img_gray,-1,DoG_filters[each])
        N_dim_tensor_dog = np.dstack((N_dim_tensor_dog,kernel_op))
    return N_dim_tensor_dog



def main():
	data_path = '/home/pratique/Downloads/cmsc733/Homework0/116353601_hw0/Phase1/BSDS500/Images/'
	img = cv2.imread('/home/pratique/Downloads/cmsc733/Homework0/116353601_hw0/Phase1/BSDS500/Images/1.jpg')
	img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	N_dim_tensor_dog = texton_tensor(img_gray)
	p,q,r = np.shape(N_dim_tensor_dog)
	inp = np.reshape(N_dim_tensor_dog,((p*q),r))
	kmeans = sklearn.cluster.KMeans(n_clusters = 64, random_state = 2)
	kmeans.fit(inp)
	labels = kmeans.predict(inp)
	l = np.reshape(labels,(p,q))
	plt.imshow(l)
	plt.show()

	# brightness kmean
	p,q= np.shape(img_gray)
	inp = np.reshape(img_gray,((p*q),1))
	kmeans = sklearn.cluster.KMeans(n_clusters = 16, random_state = 2)
	kmeans.fit(inp)
	labels = kmeans.predict(inp)
	l = np.reshape(labels,(p,q))
	plt.imshow(l,cmap = 'binary')
	plt.show()

	# img_lab = cv2.cvtColor(img, CV_BGR2Lab)
	img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV);
	p,q,r = np.shape(img_hsv)
	inp = np.reshape(img_hsv,((p*q),r))
	kmeans = sklearn.cluster.KMeans(n_clusters = 16, random_state = 2)
	kmeans.fit(inp)
	labels = kmeans.predict(inp)
	l = np.reshape(labels,(p,q))
	plt.imshow(l)
	plt.show()






	kernel=gkern(21,7)
	border = cv2.borderInterpolate(0, 1, cv2.BORDER_CONSTANT)
	sobelx64f = cv2.Sobel(kernel,cv2.CV_64F,1,0,ksize=5, borderType=border)
	sobely64f = cv2.Sobel(kernel,cv2.CV_64F,0,1,ksize=5, borderType=border)
	# theta = np.radians(45)
	# c, s = np.cos(theta), np.sin(theta)
	# R = np.array(((c,-s), (s, c)))
	final=skimage.transform.rotate(sobelx64f,90)
	plt.imshow(final,cmap='binary')
	plt.show()
	# DoG(4,30,6)


	#LM

	F = makeLMfilters()
	print F.shape

	for i in range(0,18):
	    plt.subplot(3,6,i+1)
	    plt.axis('off')
	    plt.imshow(F[:,:,i], cmap = 'gray')
	    plt.show()

	for i in range(0,18):
	    plt.subplot(3,6,i+1)
	    plt.axis('off')
	    plt.imshow(F[:,:,i+18], cmap = 'gray')
    	plt.show()

	for i in range(0,12):
	    plt.subplot(4,4,i+1)
	    plt.axis('off')
	    plt.imshow(F[:,:,i+36], cmap = 'gray')
	    plt.show()



if __name__ == '__main__':
    main()
 



