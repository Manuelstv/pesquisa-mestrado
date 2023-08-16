from skimage.io import imread, imsave
from matplotlib import pyplot as plt
from scipy.stats import special_ortho_group
from scipy.spatial.transform import Rotation as R

import cv2
import numpy as np
from glob import glob

from pdb import set_trace as pause

def synthesizeRotation(image, R):
    spi = SphericalImage(image)
    spi.rotate(R)
    return spi.getEquirectangular()

def getMapping(image, R):
    spi = SphericalImage(image)
    spi.rotate(R)
    return spi.getMap()

class SphericalImage(object):

    def __init__(self, equImage):
        self.__colors = equImage
        self.__dim = equImage.shape # height and width

        phi, theta = np.meshgrid(np.linspace(0, np.pi, num = self.__dim[0], endpoint=False), np.linspace(0, 2 * np.pi, num = self.__dim[1], endpoint=False))
        self.__coordSph = np.stack([(np.sin(phi) * np.cos(theta)).T,(np.sin(phi) * np.sin(theta)).T, np.cos(phi).T], axis=2)

    def rotate(self, R):
        data = np.array(np.dot(self.__coordSph.reshape((self.__dim[0]*self.__dim[1], 3)),R))
        self.__coordSph = data.reshape((self.__dim[0], self.__dim[1], 3))
        
        x, y, z = data[:,].T

        phi = np.arccos(z)
        theta = np.arctan2(y,x)
        theta[theta < 0] += 2*np.pi        
        theta = self.__dim[1]/(2*np.pi) * theta
        phi = self.__dim[0]/np.pi * phi

        self.mapped = np.stack([theta.reshape(self.__dim[0], self.__dim[1]), phi.reshape(self.__dim[0], self.__dim[1])], axis=2).astype(np.float32)

        self.__colors = cv2.remap(self.__colors, self.mapped, None, cv2.INTER_LINEAR,  borderMode=cv2.BORDER_REFLECT)
        
    def getEquirectangular(self): return self.__colors
    def getSphericalCoords(self): return self.__coordSph
    def getMap(self): return self.mapped


if __name__ == "__main__":

    #R = special_ortho_group.rvs(3)

    # Define angle ranges
    alpha_range = (-30, 30)  # Example range for alpha angle
    beta_range = (-30, 30)   # Example range for beta angle
    gamma_range = (-30, 30)  # Example range for gamma angle

    # Generate a random rotation matrix
    alpha = np.random.uniform(*alpha_range)
    beta = np.random.uniform(*beta_range)
    gamma = np.random.uniform(*gamma_range)

    rotation = R.from_euler('ZYX', [gamma, beta, alpha], degrees=True)
    R = rotation.as_matrix()


    print(R)

    img = imread('/home/mstveras/Pictures/vale/darktable_exported/IMG_0525.jpg')
    img2 = synthesizeRotation(img, R)
    imsave('rotated.png', img2)

