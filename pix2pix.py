import tensorflow as tf
import os
import time
from matplotlib import pyplot as plt
from IPython import display


# 1. 데이터로드
_URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'

path_to_zip = tf.keras.utils.get_file('facedes.tar.gz',
                                        origin=_URL,
                                        extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image

inp, re = load(PATH+'train/100.jpg')
# casting to int for matplotlib to show the image
plt.figure()
plt.imshow(inp/255.0)
plt.figure()
plt.imshow(re/255.0)
plt.show()

# 이미지 사이즈 조절
def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image,
                                [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image,
                                [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image

def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH,3])
    return cropped_image[0], cropped_image[1]

# normalizing the images to [-1,1]
def normalize(input_image, real_image):
    input_image = (input_image / 127.5)-1
    real_image = (real_image / 127.5)-1

    return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

plt.figure(figsize=(6,6))
for i in range(4):
    rj_inp, rj_re = random_jitter(inp, re)
    plt.subplot(2, 2, i+1)
    plt.imshow(rj_inp/255.0)
    plt.axis('off')
plt.show()
'''
더 큰 높이와 너비로 이미지 크기 조정
대상 크기로 무작위로 자르기
이미지를 가로로 무작위로 뒤집기
'''

# train, test 이미지
def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                    IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

#입력 파이프라인
train_dataset = tf.data.Dataset.list_files(PATH+'train/*.jpg')
train_dataset = train_dataset.map(load_image_train,
                                num_parallel_calls=tf.data.experimental)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(PATH+'test/*.jpg')
test_dataset = test_dataset.map(load_image_test)

'''
생성기 구축
제너레이터의 아키텍처는 수정 된 U-Net입니다.
인코더의 각 블록은 (Conv-> Batchnorm-> Leaky ReLU)입니다.
디코더의 각 블록은 (Transposed Conv-> Batchnorm-> Dropout (처음 3 개 블록에 적용)-> ReLU)입니다.
인코더와 디코더 사이에는 스킵 연결이 있습니다 (U-Net에서와 같이).
'''

OUTPUT_CHANNELS = 3

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                        kernel_initializer=initializer, use_bias=False))
    
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

down_model = downsample(3, 4)
down_result = down_model(tf.expand_dims(inp, 0))
print(down_result.shape)


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                padding='same', kernel_initializer=initializer, use_bias=False))
    
    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

up_model = upsample(3, 4)
up_result = up_model(down_result)
print(up_result.shape)


def Generator():
    inputs = tf.keras.layers.Input(shape=[256,256,3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
        downsample(128, 4), # (bs, 64, 64, 128)
        downsample(256, 4), # (bs, 32, 32, 256)
        downsample(512, 4), # (bs, 16, 16, 512)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
        downsample(512, 4), # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4), # (bs, 16, 16, 1024)
        upsample(256, 4), # (bs, 32, 32, 512)
        upsample(128, 4), # (bs, 64, 64, 256)
        upsample(64, 4), # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                            strides=2, padding='same',
                            kernel_initializer=initializer,
                            activation='tanh') ## (bs, 256, 256, 3)
    
    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


generator = Generator()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

gen_output = generator(inp[tf.newaxis, ...], training=False)
plt.imshow(gen_output[0, ...])

'''
발전기 손실
생성 된 이미지와 이미지 배열의 시그 모이 드 교차 엔트로피 손실
생성 된 이미지와 대상 이미지 사이의 MAE (평균 절대 오차) 인 L1 손실도 포함
생성 된 이미지가 대상 이미지와 구조적으로 유사
총 발전기 손실을 계산하는 공식 = gan_loss + LAMBDA * l1_loss, 여기서 LAMBDA = 100.이 값은 논문 저자가 결정
'''

LAMBDA = 100

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss=loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mae
    l1_loss = tf.reduce_mean(tf.abs(target-gen_output))

    total_gen_loss = gen_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

'''
판별자 구축
판별자는 PatchGAN입니다.
식별기의 각 블록은 (Conv-> BatchNorm-> Leaky ReLU)입니다.
마지막 레이어 이후의 출력 모양은 (batch_size, 30, 30, 1)입니다.
출력의 각 30x30 패치는 입력 이미지의 70x70 부분을 분류합니다 (이러한 아키텍처를 PatchGAN이라고 함).

판별 기는 2 개의 입력을받습니다.
실제로 분류해야하는 입력 이미지와 대상 이미지.
입력 이미지와 생성 된 이미지 (생성기의 출력)는 가짜로 분류해야합니다.
이 두 입력을 코드에서 함께 연결합니다 ( tf.concat([inp, tar], axis=-1) )
'''

def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256,256,3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256,256,3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

    down1 = downsample(64,4,False)(x)
    down2 = downsample(128,4)(down1)
    down3 = downsample(356,4)(down2)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    # ZeroPadding2D
    # This layer can add rows and columns of zeros at the top, bottom, 
    # left and right side of an image tensor
    conv = tf.keras.layers.Conv2D(512,4,strides=1,
                    kernel_initializer=initializer,
                    use_bias=False)(zero_pad1)

    bathnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(bathnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

    last = tf.keras.layers.Conv2D(1,4, strides=1,
                        kernel_initializer=initializer)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


discriminator = Discriminator()
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

disc_out = discriminator([inp[tf.newzxis, ...], gen_output], training=False)
plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')

'''
판별자 손실
판별기 손실 함수는 2 개의 입력을받습니다. 실제 이미지, 생성 된 이미지
real_loss는 실제 이미지 와 이미지 의 배열의 시그 모이 드 교차 엔트로피 손실입니다 ( 실제 이미지 이므로).
generated_loss는 생성 된 이미지 와 0 의 배열의 시그 모이 드 교차 엔트로피 손실입니다 (가짜 이미지이기 때문에)
그러면 total_loss는 real_loss와 generated_loss의 합계입니다.
'''

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
'''
Binary_crossentropy나 Categorical_crossentropy 함수에선 공통적으로 from_logits 인자를 설정할 수 있습니다.
딥러닝에서 쓰이는 logit은 모델의 출력값이 문제에 맞게 normalize 되었느냐의 여부
모델이 출력값으로 해당 클래스의 범위에서의 확률을 출력한다면, logit=False
모델의 출력값이 sigmoid 또는 linear를 거쳐서 만들어지게 된다면, logit=True
'''

# 판별자의 총 손실값
def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generator_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generator_loss

    return total_disc_loss

# 최적화(optimizer)
# Adam = Adagrad + Momentum // momentum과 gradient 히스토리 모두를 고려하는 방식
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=generator,
                                discriminator=discriminator)
                        

'''
훈련 중에 일부 이미지를 그리는 함수를 작성
테스트 데이터 세트의 이미지를 생성기로 전달합니다.
생성기는 입력 이미지를 출력으로 변환합니다.
마지막 단계는 예측과 짜잔 을 그리는 것입니다 !
참고 : training=True 는 테스트 데이터 세트에서 모델을 실행하는 동안 배치 통계를 원하기 때문에 여기서 의도 된 것입니다.
training = False를 사용하면 훈련 데이터 세트에서 누적 된 통계를 얻습니다 (원하지 않음).
'''

def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15,15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+i)
        plt.title(title[i])
        # getting the pixel values between [0,1] to plot it.
        plt.imshow(display_list[i]*0.5+0.5)
        plt.axis('off')
    plt.show()

for example_input, example_target in test_dataset.take(1):
    generate_images(generator, example_input, example_target)

'''
훈련
각 예제에 대해 출력을 생성
판별기는 input_image와 생성된 이미지를 첫번째 입력으로 받습니다.
두번째 입력은 input_image 및 target_image입니다.
생성기와 판별기의 손실을 계산
생성기 및 판별 변수(입력)에 대한 손실의 기울기를 계산하고 이를 최적화에 적용
그럼 다음 손실을 TensorBoard에 기록
'''

EPOCHS = 150

import datetime
lod_dir = 'logs/'

summary_writer = tf.summary.create_file_writer(
    lod_dir+'fit/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


@tf.function
def train_step(input_image, target, epoch):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
    tf.summary.scalar('disc_loss', disc_loss, step=epoch)

'''
실제 훈련 루프 :
Epoch 수를 반복합니다.
각 epoch에서 디스플레이를 지우고 generate_images 를 실행 generate_images 진행 상황을 표시합니다.
각 에포크에서 학습 데이터 세트를 반복하여 '.'를 인쇄합니다. 
각 예에 대해 20 epoch마다 체크 포인트를 저장합니다.
'''
def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):
        start = time.time()

        display.clear_output(wait=True)

        for example_input, example_target in test_ds.take(1):
            generate_images(generator, example_input, example_target)
        print('Epoch:', epoch)

        # Train
        for n, (input_image, target) in train_ds.enumerate():
            print('.', end='')
            if (n+1) % 100 == 0:
                print()
            train_step(input_image, target, epoch)
        print()

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Time taken for epoch {} is {} sec\n'.format(epoch+1,
                                                        time.time()-start))
    checkpoint.save(file_prefix = checkpoint_prefix)

fit(train_dataset, EPOCHS, test_dataset)

# Run the trained model on a few examples from the test dataset
for inp, tar in test_dataset.take(5):
  generate_images(generator, inp, tar)