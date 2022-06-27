from matplotlib import pyplot as plt
import cv2
import numpy as np

# Part 2
print("\n\n**\tImage segmentation using both pixel colour and location values as features\t**\n")

# Reading the image
img = cv2.imread("Image.jpg")

# Changing color code from BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Plotting original image
plt.imshow(img)
plt.show()

color_location_vals = []

# Reshaping the image to get the pixel color values
height, width, cols = img.shape
img = img.reshape((-1, 3))
count=0

# Adding location values as two other values
for i in range(height):
    for j in range(width):
        p = 0
        a,b,c=img[count]
        count+=1
        color_location_vals.append([a/255,b/255,c/255,i/height,j/width])
color_location_vals=np.array(color_location_vals)
print(color_location_vals)
# Pixel color match value with mean
def pixel_proxim(vec, mean):
    val = 0
    for i in range(len(vec)):
        val += (vec[i]- mean[i])**2
    
    return val

means = []

# Number of segments(color clusters) to be created
k = 15

# Declaring means with random values
for i in range(k):
    means.append([((i+1)/k)]*5)
print("\n \n",means)

# Function to provide updated means
def give_means(cluster_li, color_location_vals, m):
    count, totals, means = [0]*m, [[0]*5]*m, [0]*m
    for i, j in enumerate(cluster_li):
        totals[j] = np.add(totals[j], color_location_vals[i])
        count[j] += 1
    
    for j in range(m):
        if count[j]:
            means[j] = np.divide(totals[j], count[j])
            
    print(means)
    return means

# Function which provides index list to assign which cluster each pixel belongs to
def give_cluster_li(means, color_location_vals):
    cluster_li = []
    for i in color_location_vals:
        proxim_vals = []
        for j in means:
            proxim_vals.append(pixel_proxim(i, j))
        cluster_li.append(proxim_vals.index(min(proxim_vals)))
    
    return cluster_li


cluster_li = give_cluster_li(means, color_location_vals)

# Iterating for n iterations. This will give us the most dominant shades in RGB format in the given image as the means vector
n = 3
for i in range(n):
    means = give_means(cluster_li, color_location_vals, k)       # New means
    cluster_li = give_cluster_li(means, color_location_vals)     # New clusters

# Removing the location values after getting the clusters
i = 0
while i<len(means):
    means[i] = (means[i][:3]*255).astype(int)
    i += 1

# Segmented image using k-means
seg_img = []
for i in range(height*width):
    seg_img.append(means[cluster_li[i]])

# Re-shaping seg_img in the shape of the original image
seg_img = np.reshape(seg_img, (height, width, cols))

print(seg_img.shape)
# Plotting the segmented image
plt.imshow(seg_img)
plt.show()