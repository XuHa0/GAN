# This is GAN！
项目代码分为几块
	1.gan.py主要是Generator和Discriminator的构建
	2.dataset.py主要是负责导入图片数据
	3.gan_train.py主要是负责train部分(DCGAN)，下载好数据集，设置好图片路径和保存路径之后，可以直接运行这部分进行训练，epochs部分可以自行修改。

本项目是在kaggle上进行训练的，由于kaggle kernels只能单文件训练(😂或许是因为我只会直接运行单文件)，这里将代码块进行合并为DCGAN.py。可以直接在kaggle上进行训练。

附WGAN.py (未在kaggle kernels上训练测试过)由于DCGAN的训练并不稳定，在运行超过20000epochs的时候，d_loss降到了零点几，但是g_loss最高却上升到了13，说明了Discriminator训练的过好，使得Generator被不断判错以至于其最后便开始乱画了。而WGAN由于加上了一个正则化项，使得training更加稳定，图片的多样性也有所增加。具体的loss输出可以参考log.txt。

由于本项目是自定义图层，目前TF2.0不支持自动保存整个模型，load_gan.py中给出了恢复模型的方法。

数据集下载地址：[picface | Kaggle](https://www.kaggle.com/xuha0212/picface)

附GAN.pdf  WGAN.pdf帮助回忆相关知识。



