import argparse

from defocus_estimate import *


def get_args():
    parser = argparse.ArgumentParser(description='Defocus map estimation from a single image, '
                                                 'S. Zhuo, T. Sim - Pattern Recognition, 2011 - Elsevier \n')

    parser.add_argument('-i', metavar='--image', required=True,
                        type=str, help='Defocused image \n')

    args = parser.parse_args()
    image = args.i

    return {'image': image}


if __name__ == '__main__':

    args = get_args()

    img = cv2.imread(args['image'])
    fblurmap = estimate_bmap_laplacian(img, sigma_c = 3, std1 = 1, std2 = 1.5)

    cv2.imwrite('image_bmap.png', np.uint8(fblurmap*255))


