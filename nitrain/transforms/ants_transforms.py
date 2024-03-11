# Transforms that apply specific antspy functions to images

class ApplyAntsTransform():
    pass

class Registration():
    pass

class BrainExtraction():
    pass

class ReorientImage:
    pass

class SimulateBiasField():
    """
            if do_simulate_bias_field and random.sample((True, False), 1)[0]:
                tic = time.perf_counter()
                for ii in range(len(images)):
                    log_field = antspynet.simulate_bias_field(images[ii], number_of_points=10, sd_bias_field=1.0, number_of_fitting_levels=2, mesh_size=10)
                    log_field = log_field.iMath("Normalize")
                    field_array = np.power(np.exp(log_field.numpy()), random.sample((2, 3, 4), 1)[0])
                    images[ii] = images[ii] * ants.from_numpy(field_array, origin=images[ii].origin, spacing=images[ii].spacing, direction=images[ii].direction)
                    images[ii] = (images[ii] - images[ii].min()) / (images[ii].max() - images[ii].min())
                toc = time.perf_counter()
                # print(f"Sim bias field {toc - tic:0.4f} seconds")
    """
    pass


class AlignWithTemplate():
    """
    This function makes sure the images have the same
    orientation as a template.
            
    center_of_mass_template = ants.get_center_of_mass(template)
    center_of_mass_image = ants.get_center_of_mass(image)
    translation = tuple(np.array(center_of_mass_image) - np.array(center_of_mass_template))
    xfrm = ants.create_ants_transform(transform_type=
        "Euler3DTransform", center = center_of_mass_template,
        translation=translation,
        precision='float', dimension=image.dimension)

    imageX = ants.apply_ants_transform_to_image(xfrm, image, template)
    """
    pass
