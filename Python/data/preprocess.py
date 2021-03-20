import numpy as np
import scipy.ndimage.measurements as scipy_measurements
import miapy.data.transformation as miapy_tfm


class ClipNegativeTransform(miapy_tfm.Transform):

    def __init__(self, entries=('images',)) -> None:
        super().__init__()
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        for entry in self.entries:
            if entry not in sample:
                continue

            img = sample[entry]

            m = np.min(img)
            if m < 0:
                print('Clipping... min: {}'.format(m))
                img = np.clip(img, a_min=0, a_max=None)

            sample[entry] = img
        return sample


class CenterCentroidTransform(miapy_tfm.Transform):

    def __init__(self, entries=('images',)) -> None:
        super().__init__()
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        for entry in self.entries:
            if entry not in sample:
                continue

            img = sample[entry]
            centroid_transform = []

            # move centroid to center
            com = scipy_measurements.center_of_mass(img > 0)
            for axis in range(0, 3):
                diff = com[axis] - int(img.shape[axis] / 2)
                centroid_transform.append(-diff)
                if abs(diff) > 1:
                    img = np.roll(img, int(-diff), axis=axis)

            sample[entry] = img

            # store the centroid transformation (will be written to metadata later)
            sample['centroid_transform'] = np.array(centroid_transform)

        return sample


class RandomRotateShiftTransform(miapy_tfm.Transform):

    def __init__(self, do_rotate=True, shift_amount=0, entries=('images',)) -> None:
        super().__init__()
        self.entries = entries
        self.do_rotate = do_rotate
        self.shift_amount = shift_amount
        print('Using RandomRotateShiftTransform({}, {})'.format(do_rotate, shift_amount))

    def __call__(self, sample: dict) -> dict:
        for entry in self.entries:
            if entry not in sample:
                continue

            img = sample[entry]

            # shift +/- shift_amount pixels
            if self.shift_amount != 0:
                # number of pixels to shift
                n = np.random.randint(-self.shift_amount, self.shift_amount + 1)
                # axis
                k = np.random.randint(0, 3)
                img = np.roll(img, n, axis=k)

            # 3x rotate by 90 degree around a random axis
            if self.do_rotate:
                planes = [(0, 1), (0, 2), (1, 2)]
                for i in range(0, 3):
                    k = np.random.randint(0, 3)
                    plane_idx = np.random.randint(0, 3)
                    img = np.rot90(img, k, planes[plane_idx])

            sample[entry] = img

        return sample


def get_bounding_box(img):
    a = np.argwhere(img)
    min0, min1, min2 = a.min(0)
    max0, max1, max2 = a.max(0)
    return [min0, max0, min1, max1, min2, max2]


# Apply reverse center centroid transform
def revert_centroid_transform(img, centroid_transform):
    for axis in range(0, 3):
        diff = -centroid_transform[axis]
        if abs(diff) > 1:
            img = np.roll(img, int(diff), axis=axis)

    return img
