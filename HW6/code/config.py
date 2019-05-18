
class Config:
    use_random_crop = False
    use_dropout = False
    use_adversarial = False
    use_overlapping_loss = True

    batch_size = 32
    n_epochs = 200
    learning_rate = 1e-3

    l1_kernel_regularization = 0.01
    l2_kernel_regularization = 0.01

    dropout_rate = 0.25






