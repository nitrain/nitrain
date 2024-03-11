from .base_transform import BaseTransform

class StandardNormalize(BaseTransform):
    pass

class Threshold():
    """
   seg_mask = ants.threshold_image(seg, 0, 0, 0, 1)
            if seg_mask.sum() < 1:
                continue
    """
    pass

class RangeNormalize(BaseTransform):
    pass

class AdjustBrightness(BaseTransform):
    pass

class Blur(BaseTransform):
    pass

class Noise:
    """
                noise_parameters = (0.0, random.uniform(0, 0.05))
                imageX = ants.add_noise_to_image(imageX, noise_model="additivegaussian", noise_parameters=noise_parameters)
                imageX = (imageX - imageX.mean()) / imageX.std()
    """
    pass

class HistogramWarpIntensity():
    """
    https://github.com/ntustison/ANTsXNetTraining/blob/main/BrainExtraction/T1/batch_generator.py
    
    if do_histogram_intensity_warping and random.sample((True, False), 1)[0]:
    break_points = [0.2, 0.4, 0.6, 0.8]
    displacements = list()
    for b in range(len(break_points)):
        displacements.append(abs(random.gauss(0, 0.175)))
        if random.sample((True, False), 1)[0]:
            displacements[b] *= -1
    image = antspynet.histogram_warp_image_intensities(image,
        break_points=break_points, clamp_end_points=(True, False),
        displacements=displacements)
    """
    pass