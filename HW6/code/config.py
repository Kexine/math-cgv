
class Config:
    use_random_crop = False
    use_adversarial = False
    use_overlapping_loss = True
    use_random_augment = True

    batch_size = 32
    n_epochs = 500
    learning_rate = 5e-4

    l1_kernel_regularization = 0.01
    l2_kernel_regularization = 0.01

    use_normal_dropout = False
    use_spatial_dropout = False
    normal_dropout_rate = 0.2
    spatial_dropout_rate = 0.0

    missing_patch_fill_value = 0.5






