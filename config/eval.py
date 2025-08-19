import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    # 随机种子
    config.seed = 42
    # 分辨率
    config.resolution = 768

    ###### Pretrained Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # 基础模型路径
    pretrained.model = "stabilityai/stable-diffusion-3-flux"
    # 模型版本
    pretrained.revision = "main"

    ###### Sampling ######
    config.sample = sample = ml_collections.ConfigDict()
    # 评估时的推理步数
    sample.eval_num_steps = 40
    # 分类器引导权重
    sample.guidance_scale = 4.5
    # 测试批次大小
    sample.test_batch_size = 1
    # 噪声水平
    sample.noise_level = 0.0

    ###### Evaluation ######
    config.eval = eval = ml_collections.ConfigDict()
    # 是否保存生成的图像
    eval.save_images = True
    # 保存的图像数量
    eval.num_samples_to_save = 10
    # 是否计算统计信息
    eval.compute_stats = True

    return config
