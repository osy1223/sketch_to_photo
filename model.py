# example of pix2pix gan for satellite to map image-to-image translation
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, ELU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot

# 판별자 모델
def define_discriminator(image_shape):
	init = RandomNormal(stddev=0.02)
	in_src_image = Input(shape=image_shape)
	in_target_image = Input(shape=image_shape)

	merged = Concatenate()([in_src_image, in_target_image])

	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	model = Model([in_src_image, in_target_image], patch_out)

	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model

# encoder(downsampling)
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	init = RandomNormal(stddev=0.02)
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	g = ELU(alpha=0.2)(g)
	return g

# decoder (upsampling)
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	init = RandomNormal(stddev=0.02)
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	g = BatchNormalization()(g, training=True)
	if dropout:
		g = Dropout(0.5)(g, training=True)
	g = Concatenate()([g, skip_in])
	g = Activation('relu')(g)
	return g

# 생성자 모델
def define_generator(image_shape=(256,256,3)):
	init = RandomNormal(stddev=0.02)
	in_image = Input(shape=image_shape)
	# encoder model
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	e7 = define_encoder_block(e6, 512)
	# 병목현상방지 
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	# decoder model
	d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)

	# output
	g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)

	# Conv2DTranspose => Upsampling2D+Conv2D
	# g = Reshape((1, 1, n_filters))(d7)
 	# g = UpSampling2D()(g)
    # g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(g)
	
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	return model

# GAN 모델
def define_gan(g_model, d_model, image_shape):
	d_model.trainable = False
	in_src = Input(shape=image_shape)
	gen_out = g_model(in_src)
	dis_out = d_model([in_src, gen_out])
	model = Model(in_src, [dis_out, gen_out])
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])
	return model

# 훈련 이미지 준비
def load_real_samples(filename):
	data = load(filename)
	X1, X2 = data['arr_0'], data['arr_1']
	# 스케일 from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

# data를 n_batch만큼 가져옴
def generate_real_samples(dataset, n_samples, patch_shape):
    # unpack dataset
	trainA, trainB = dataset
	# choose random instances
	ix = randint(0, trainA.shape[0], n_samples)
	# retrieve selected images
	X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y

# 가짜 이미지 생성
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)
	# 가짜(0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, n_samples=3):
	# select a sample of input images
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	# plot real source images
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realA[i])
	# plot generated target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i])
	# plot real target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realB[i])
	# save plot to file
	filename1 = 'plot_%06d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'model_%06d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))

# 훈련
def train(d_model, g_model, gan_model, dataset, n_epochs=150, n_batch=32):
	# 판별자의 output shape
	n_patch = d_model.output_shape[1]
	
	trainA, trainB = dataset

	# epoch마다의 배치 숫자 계산
	bat_per_epo = int(len(trainA) / n_batch)

	# 전체 epoch동안의 train data 분할 횟수(batch 횟수)
	n_steps = bat_per_epo * n_epochs

	print("n_steps : ", n_steps)
	d_loss1_list=[]
	d_loss2_list=[]
	g_loss_list=[]

	for i in range(n_steps):
		# batch 한 번 만큼의 sketch, photo
		[X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)

		# batch 한 번 만큼의 생성된 가짜 사진
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)

		# 진짜 사진을 '진짜'라고 판별
		d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
		d_loss1_list.append(d_loss1)

		# 가짜 사진을 '가짜'라고 판별
		d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
		d_loss2_list.append(d_loss2)

		# GAN 
		g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
		g_loss_list.append(g_loss)
		# print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
		print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))

		# 진행 상황 확인
		if (i+1) % (bat_per_epo * 10) == 0:
			summarize_performance(i, g_model, dataset)

	# loss plot
	#그래프
	import matplotlib.pyplot as plt

	# epochs = len(iterations)
	x_axis = range(0,n_steps)

	fig, ax = plt.subplots()
	ax.plot(x_axis, d_loss1_list, label="d_loss1")
	ax.plot(x_axis, d_loss2_list, label="d_loss2")

	ax.legend()
	plt.ylabel("Loss")
	plt.xlabel("Iteration")
	plt.title("GAN Loss")
	# plt.show()

	fig,ax = plt.subplots()
	ax.plot(x_axis, g_loss_list, label="g_loss")

	ax.legend()
	plt.ylabel("Loss")
	plt.xlabel("Iteration")
	plt.title("GAN Loss")
	# plt.show()


# 데이터 호출
dataset = load_real_samples('berry_bear_pot_car.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)

# sketch와 photo의 sahpe
image_shape = dataset[0].shape[1:]

# 모델
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)

# 컴파일
gan_model = define_gan(g_model, d_model, image_shape)

# 훈련
train(d_model, g_model, gan_model, dataset)

