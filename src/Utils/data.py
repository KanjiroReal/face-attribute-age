def make_weight_for_balance_classes(images, num_classes: int):
    n_images = len(images)
    count_per_class = [0] * num_classes
    for _, image_class in images:
        count_per_class[image_class] += 1
    weight_per_class = [0.] * num_classes
    for i in range(num_classes):
        weight_per_class[i] = float(n_images) / float(count_per_class[i])
    weights = [0] * n_images
    for idx, (image, image_class) in enumerate(images):
        weights[idx] = weight_per_class[image_class]
    return weights, weight_per_class
