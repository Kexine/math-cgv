
class Config:
    use_random_crop = True
    use_adversarial = False
    use_overlapping_loss = True
    use_random_augment = True

    batch_size = 32
    n_epochs = 500
    learning_rate = 5e-4

    l1_kernel_regularization = 0.005
    l2_kernel_regularization = 0.005

    use_normal_dropout = False
    use_spatial_dropout = True
    normal_dropout_rate = 0.0
    spatial_dropout_rate = 0.1

    missing_patch_fill_value = 0.5






