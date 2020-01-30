import  os
import  numpy as np
import  tensorflow as tf
from    tensorflow import keras
#from    scipy.misc import toimage
from PIL import Image
import  glob
from    gan import Generator, Discriminator
import time
from    dataset import make_anime_dataset


def save_result(val_out, val_block_size, image_path, color_mode):#将多个图片的内容拼在一起方便查看
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        # img = img.astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b+1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    Image.fromarray(final_image).save(image_path)
    #toimage(final_image).save(image_path)


def celoss_ones(logits):
    # [b, 1]
    # [b] = [1, 1, 1, 1,]
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.ones_like(logits))#送进去的labels全部为1，因为我们这里送进来的是真的图片
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    # [b, 1]
    # [b] = [1, 1, 1, 1,]
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)

def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    # 1. treat real image as real
    # 2. treat generated image as fake
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    d_real_logits = discriminator(batch_x, is_training)

    d_loss_real = celoss_ones(d_real_logits)#我们用1表示真，用0表示假
    d_loss_fake = celoss_zeros(d_fake_logits)

    loss = d_loss_fake + d_loss_real

    return loss


def g_loss_fn(generator, discriminator, batch_z, is_training):

    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    loss = celoss_ones(d_fake_logits)#我们希望generator生成的图片越接近与1越好

    return loss

def main():

    tf.random.set_seed(22)#随机种子
    np.random.seed(22)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    assert tf.__version__.startswith('2.')


    # hyper parameters
    z_dim = 100
    epochs = 100000
    batch_size = 512
    learning_rate = 0.002
    is_training = True


    #img_path = glob.glob(r'C:\Users\Jackie Loong\Downloads\DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2-master\data\faces\*.jpg')
    #img_path = glob.glob(r'/Users/xuhao/Downloads/GAN实战/faces/*.jpg')#图片的路径
    img_path = glob.glob(r'./faces/*.jpg')  # 图片的路径

    dataset, img_shape, _ = make_anime_dataset(img_path, batch_size)#调用dataset.py中的函数
    print(dataset, img_shape)
    sample = next(iter(dataset))
    print(sample.shape, tf.reduce_max(sample).numpy(),
          tf.reduce_min(sample).numpy())#[512,64,64,3]
    dataset = dataset.repeat()
    db_iter = iter(dataset)#使其可以无限制的从dataset中间sample


    generator = Generator()
    generator.build(input_shape = (None, z_dim))#？？？？？？
    discriminator = Discriminator()
    discriminator.build(input_shape=(None, 64, 64, 3))#？？？？？？build

    g_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)#创建两个优化器,beta为GAN的参数,两个优化器是分开的
    d_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)


    for epoch in range(epochs):

        batch_z = tf.random.uniform([batch_size, z_dim], minval=-1., maxval=1.)#随机sample的值,generator的输入
        batch_x = next(db_iter)

        # train D
        with tf.GradientTape() as tape:
            d_loss = d_loss_fn(generator, discriminator, batch_z, batch_x, is_training)
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))


        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch  == 1:
            generator._set_inputs(batch_z)
            discriminator._set_inputs(batch_x)
            saved_model_path = "./saved_models/{}".format(int(time.time()))
            tf.keras.models.save_model(generator, saved_model_path)
            saved_model_path = "./saved_models/{}".format(int(time.time()))
            tf.keras.models.save_model(discriminator, saved_model_path)


        #if epoch % 100 == 0:
        if epoch % 2 == 0:
            print(epoch, 'd-loss:',float(d_loss), 'g-loss:', float(g_loss))

            z = tf.random.uniform([100, z_dim])
            fake_image = generator(z, training=False)
            img_path = os.path.join('images', 'gan-%d.png'%epoch)
            save_result(fake_image.numpy(), 10, img_path, color_mode='P')#10为行列



if __name__ == '__main__':
    main()