import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops
import sys

'''
The gaussian function that actually computes the equation!
We only need to do one dimensional guassian computation for the most part here since in
our gaussian kernal we're going to go work with a 1D vector and then use
numpy.outer to make a gaussian 2d

But our general equation here is:
1/sqrt(2pi) * e^((x-mu)/sigma)^2/2
'''
def gaussian(sigma, x, mu):
    return (1/np.sqrt(2*np.pi)) * np.e ** (-np.power((x - mu)/sigma,2)/2)

'''
The Gaussian Kernal function actual computes our gaussian kernal based off of the 
sigma we input along with the size (which is defaulted to an integer)
'''
def gaussian_kernal(sigma, size = 5):
    #We will want a linearaly spaced vector to easily make our distribution
    a = size // 2
    k_1d = np.linspace(-a, a, size)

    #Now we have a vector which has a mean of 0 since we linearaly spaced around 0
    for i in range(size):
        k_1d[i] = gaussian(sigma, k_1d[i], 0)

    #Now we make the 1d vector into a 2d vector
    k_2d = np.outer(k_1d.T, k_1d.T)

    #But now we need to normalize our gaussian kernal
    #We can do this by taking the max value of our matrix (which should be the middle) and 
    #multiplying by the reciprical 
    max_val = k_2d.max()
    for x in range(size):
        for y in range(size):
            k_2d[x][y] *= 1 / max_val

    #Now we can return our kernal knowing it's all nice and normalized!
    return k_2d

'''
Our convolution function that uses the kernal we will make above and applies it to the image
'''
def convulation(img, kernal):

    #Here we get our dimensions for now only our image but our kernal since we can't just assume it's five
    img_y,img_x = img.shape
    k_y, k_x = kernal.shape

    #Now let's apply some padding. For simplicity's sake I am going to use zero padding!
    o = np.zeros(img.shape)

    #Getting parameters for our padded image
    pad_y = int((k_y - 1)/2)
    pad_x = int((k_x - 1)/2)

    pad_img_x = img_x + (2 * pad_x)
    pad_img_y = img_y + (2 * pad_y)

    #Making our padded image
    pad_img = np.zeros((pad_img_y, pad_img_x))
    pad_img[pad_y:pad_img.shape[0] - pad_y, pad_x:pad_img.shape[1] - pad_x] = img

    for y in range(pad_img_y):
        for x in range(pad_img_x):

            #If we are at the leftmost pixel of the image we will want to copy it to the rest of the pixels to the left
            #So the reverse algorithm is that if we are before the leftmost pixel of the image we will want that to equal the leftmost pixel
            if x < pad_x and y > pad_y and y < img_y:
                pad_img[0,x] = img[pad_y,x]

            #Now, we may have y less than pad_y, which means we're at a corner:
            if x < pad_x and y < pad_y:
                pad_img[y,x] = img[0,0]

            #What if x is greater than pad_x? Then we have to see if the y is less than pad y
            if x > pad_x and y < pad_y and x < img_x:
                pad_img[y,x] = img[pad_y,x]

            #Similar if y is greater than img_y
            if x > pad_x and y >= img_y and x < img_x:
                pad_img[y,x] = img[img_y-1, x]
            
            #Top right corner
            if x >= img_x and y < pad_y:
                pad_img[y,x] = img[0,img_x-1]

            #Bottom right corner
            if x >= img_x and y >= img_y:
                pad_img[y,x] = img[img_y-1,img_x-1]

            #Bottom left corner
            if x < pad_x and y >= img_y:
                pad_img[y,x] = img[img_y-1,0]

            #Finally, the right side
            if x >= img_x and y > pad_y and y < img_y:
                pad_img[y,x] = img[y, img_x-1]

    #Now we can implement the convolution!
    for r in range(img_y):
        for c in range(img_x):
            px = c + k_x
            py = r + k_y
            o[r,c] = np.sum(kernal * pad_img[r:py,c:px])
    
    return o

'''
0 (or anything else really) is horizontal and 1 is vertical 
'''
def sobel_kernal(dir = 0):
    kernal = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    if dir == 1:
        kernal = np.flip(kernal.T, axis = 0)
    return kernal

def guassian_filter(pgm, sigma):
    

    kernal = gaussian_kernal(sigma)
    o = convulation(pgm, kernal)
    return o

'''
For our sobel filter, we have to smoot it first. So we can easily just use
our code from our last problem!
'''
def sobel_filter(pgm, gm, sigma):
    smoothed_pgm = guassian_filter(pgm, sigma)

    #Finding it in the x direction
    kernal = sobel_kernal()
    sobel_pgm_x = convulation(smoothed_pgm, kernal)
    
    #Finding it in the y direction
    kernal = sobel_kernal(1)
    sobel_pgm_y = convulation(smoothed_pgm, kernal)

    #Now we have to combine the two with our GRADIENT MAGNITUDE!:

    g_mag = np.sqrt(np.square(sobel_pgm_x) + np.square(sobel_pgm_y))

    #Our threshold will be 255 
    g_mag *= gm / g_mag.max()

    for x in range(g_mag.shape[0]):
        for y in range(g_mag.shape[1]):
            if g_mag[x,y] <= 0.8:
                g_mag[x,y] = 0

    #Now, to help with non-max-suppression I wanna compute an angle gradient that gives us the relative angles of the gradients
    angle = np.arctan2(sobel_pgm_y, sobel_pgm_x) * 180/np.pi

    return (g_mag, angle)


'''
We cannot use interpolation here so we will have to do it by computing the direction across the gradient
'''
def non_maximum_supression(img, angles):

    img_y,img_x = img.shape
    o = np.zeros((img_y,img_x))

    y = 1
    x = 1

    #Now we can do the algorithm
    for y in range(img_y-1):
        for x in range(img_x-1):

            '''
            So the basic premise here is that we're finding the angle between the two points we're looking at. We don't really need to look at the edges because there shouldn't be any
            detectable edges really close so we can ignore those.

            But let's think of it like this:
            An angle between 22.5 and -22.5 or -157.5 and 180 is in the positive or negative x direction
            An angle between 22.5 and 67.5 or -112.5 and 157.5 is in the positive or negative x AND y direction (diaganol)
            And angle between 67.5 and 112.5 or -67.5 and -112.5 is in the positive or negative y direction
            An angle between 112.5 and 157.5 or -22.5 and -67.5 is in the positive or negative x AND y direction like the previous one

            It might make no sense just looking at it, but I drew a circle and figured it out from there. The circle really helped lmfao
            '''

            if (angles[y][x] >= -22.5 and angles[y][x] <= 22.5) or (angles[y][x] < -157.5 and angles[y][x] >= -180): 

                #So now we have to see if the point we are looking at is greater than both of the surrounding points in the direction
                #if it is, we keep it otherwise we gut it
                if (img[y,x] >= img[y,x+1]) and (img[y,x] >= img[y,x-1]):
                    o[y,x] = img[y,x]
                    
                else:
                    o[y,x] = 0
                
            #And then we just keep on doing this using the definitions I mentioned above! (I'm using elif to speed up computation, it actually doesn't matter)
            if (angles[y,x] >= 22.5 and angles[y,x] <= 67.5) or (angles[y,x] < -112.5 and angles[y,x] >= -157.5):

                if (img[y,x] >= img[y+1,x+1]) and (img[y,x] >= img[y-1,x-1]):
                    o[y,x] = img[y,x]

                else:
                    o[y,x] = 0
            
            if (angles[y,x] >= 67.5 and angles[y,x]<= 112.5) or (angles[y,x] < -67.5 and angles[y,x] >= -112.5):

                if (img[y,x] >= img[y + 1,x]) and (img[y,x] >= img[y-1,x]):
                    o[y,x] = img[y,x]

                else:
                    o[y,x] = 0
            
            if (angles[y,x] >= 112.5 and angles[y,x] <= 157.5) or (angles[y,x] < -22.5 and angles[y,x] >= -67.5):
                if(img[y,x] >= img[y+1,x-1]) or (img[y,x] >= img[y-1,x+1]):
                    o[y,x] = img[y,x]
                else:
                    o[y,x] = 0
    return o



'''
In order to see my results easier I am going to save the pgm to a png
'''
def pgm_to_png(pgm, filename):
    new_file = "{}.png".format(filename)
    
    plt.imsave(new_file, pgm)
    img = Image.open(new_file)
    img.convert("L").save(new_file)

'''
Our default args will be the red file with sigma one and threshold 100
'''
def main(filename = "burg.pgm", sigma = 1, threshold = 255):
    try:
        with open(filename, 'rb') as pgmf:
            imp = plt.imread(pgmf)
    except Exception as err:
        print("Unknown filename. Accepted filenames include: 'red.pgm' 'plane.pgm' 'kangaroo.pgm' and 'fox.pgm'")
        sys.exit()
        return

    #Computing the sobel_filter
    tup = sobel_filter(imp, threshold, sigma)

    g, angles = tup

    image = non_maximum_supression(g,angles)
    pgm_to_png(image, "result")

args = sys.argv
alength = len(sys.argv)

if alength == 1 or alength == 0:
    main()
    sys.exit()
filename = args[1]
if alength == 2:
    main(filename)
    sys.exit()
sigma = args[2]
if not sigma.isnumeric():
    raise ValueError("Sigma must be a digit. Paramenters: filename='red.pgm' sigma=1 threshold=100")
if alength == 3:
    main(filename, int(sigma))
    sys.exit()
threshold = args[3]
if not threshold.isnumeric():
    raise ValueError("Threshold must be a digit. Paramenters: filename='red.pgm' sigma=1 threshold=100")
if alength == 4:
    main(filename, int(sigma), int(threshold))
    sys.exit()
else:
    raise ValueError("Too many arguments. Must be less than 4. Paramenters: filename='red.pgm' sigma=1 threshold=100")

