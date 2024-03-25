import glob, numpy, os, random, soundfile, torch, wave
from scipy import signal
from tools import *
import numpy as np
import librosa

def get_Loader(args, dic_label = None, cluster_only = False):
	# # Get the loader for the cluster, batch_size is set as 1 to handlle the variable length input. Details see 1.2 part from here: https://github.com/TaoRuijie/TalkNet-ASD/blob/main/FAQ.md
	# clusterLoader = cluster_loader(**vars(args))
	# clusterLoader = torch.utils.data.DataLoader(clusterLoader, batch_size = 1, shuffle = True, num_workers = args.n_cpu, drop_last = False)
	#
	# if cluster_only == True: # Only do clustering
	# 	return clusterLoader
	# Get the loader for training
	trainLoader = train_loader(dic_label = dic_label, **vars(args))
	trainLoader = torch.utils.data.DataLoader(trainLoader, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)

	testLoader = test_loader(**vars(args))
	testLoader = torch.utils.data.DataLoader(testLoader, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, drop_last=True)

	return trainLoader, testLoader
class train_loader(object):
	def __init__(self, train_list, train_path, musan_path, rir_path, max_frames, sample_rate, num_frames,**kwargs):
		self.train_path = train_path
		self.max_frames = max_frames * 160 + 240 # Length of segment for training
		self.sample_rate = sample_rate
		self.num_frames = num_frames
		# self.dic_label = dic_label # Pseudo labels dict
		# self.noisetypes = ['noise','speech','music']
		# self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
		# self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
		# self.noiselist = {}
		# augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
		# for file in augment_files:
		# 	if file.split('/')[-4] not in self.noiselist:
		# 		self.noiselist[file.split('/')[-4]] = []
		# 	self.noiselist[file.split('/')[-4]].append(file)
		# self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))
		# self.data_list = []
		# lines = open(train_list).read().splitlines()
		# for index, line in enumerate(lines):
		# 	file_name     = line.split()[1]
		# 	self.data_list.append(file_name)
		# Load data & labels
		self.data_list = []
		self.data_label = []
		lines = open(train_list).read().splitlines()
		dictkeys = list(set([x.split()[0] for x in lines]))
		dictkeys.sort()
		dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
		for index, line in enumerate(lines):
			speaker_label = dictkeys[line.split()[0]]
			file_name = os.path.join(train_path, line.split()[1])
			self.data_label.append(speaker_label)
			self.data_list.append(file_name)

	def __getitem__(self, index):
		# file = self.data_list[index] # Get the filename
		# label = self.data_label[index] # Load the pseudo label
		# segments = self.load_wav(file = file) # Load the augmented segment
		# # segments = torch.FloatTensor(numpy.array(segments))

		audio, sr = librosa.load(os.path.join(self.train_path, self.data_list[index]), sr=self.sample_rate)
		length = self.num_frames * (int(sr / 100)) + 240  # TODO 后续考虑设置长度为固定的3S
		if audio.shape[0] <= length:
			shortage = length - audio.shape[0]
			audio = numpy.pad(audio, (0, shortage), 'wrap')

		startframe = random.choice(range(0, audio.shape[0] - (length)))  # Choose the startframe randomly
		audio_1 = audio[int(startframe):int(startframe) + length]
		# audio_1 = numpy.stack([audio_1], axis=0).astype(np.float)
		audio_1 = numpy.expand_dims(audio_1, axis=0).astype(np.float32)

		augtype1 = random.randint(7, 11)
		audio_aug = []
		audio_aug.append(self.random_aug(augtype1, audio_1, sr))
		audio_aug = numpy.concatenate(audio_aug, axis=0)

		return torch.FloatTensor(audio_aug), self.data_label[index]
		# return segments, label



	def load_wav(self, file):
		# utterance, _ = soundfile.read(os.path.join(self.train_path, file)) # Read the wav file
		# if utterance.shape[0] <= self.max_frames: # Padding if less than required length
		# 	shortage = self.max_frames - utterance.shape[0]
		# 	utterance = numpy.pad(utterance, (0, shortage), 'wrap')
		# startframe = random.choice(range(0, utterance.shape[0] - (self.max_frames))) # Choose the startframe randomly
		# segment = numpy.expand_dims(numpy.array(utterance[int(startframe):int(startframe)+self.max_frames]), axis = 0)
		#
		# if random.random() <= 0.5:
		# 	segment = self.add_rev(segment, length = self.max_frames) # Rever
		# if random.random() <= 0.5:
		# 	segment = self.add_noise(segment, random.choice(['music', 'speech', 'noise']), length = self.max_frames) # Noise

		audio, sr = librosa.load(os.path.join(self.train_path, file), sr=self.sample_rate)
		length = self.num_frames * (int(sr / 100)) + 240  # TODO 后续考虑设置长度为固定的3S
		audiosize = audio.shape[0]
		if audio.shape[0] <= length:
			shortage = length - audio.shape[0]
			audio = numpy.pad(audio, (0, shortage), 'wrap')

		startframe = random.choice(range(0, audio.shape[0] - (length))) # Choose the startframe randomly
		audio_1 = audio[int(startframe):int(startframe) + length]
		# audio_1 = numpy.stack([audio_1], axis=0).astype(np.float)
		audio_1 = numpy.expand_dims(audio_1, axis=0).astype(np.float32)


		augtype1 = random.randint(7, 11)
		audio_aug = []
		audio_aug.append(self.random_aug(augtype1, audio_1, sr))
		audio_aug = numpy.concatenate(audio_aug, axis=0)

		return torch.FloatTensor(audio_aug)

	def random_aug(self, augtype, audio, sr):
		if augtype == 0:  # Original
			audio = audio

		elif augtype == 1:  # Babble
			audio = self.add_noise(audio, sr, 'speech')
		elif augtype == 2:  # Music
			audio = self.add_noise(audio, sr, 'music')
		elif augtype == 3:  # Television noise
			audio = self.add_noise(audio, sr, 'speech')
			audio = self.add_noise(audio, sr, 'music')

		elif augtype == 4:  # Reverberation
			audio = self.add_rev(audio, sr)
		elif augtype == 6:  # Noise
			audio = self.add_noise(audio, 'noise')
		elif augtype == 7:
			audio = self.noise(audio)
		elif augtype == 8:
			audio = self.shift(audio)
		# elif augtype == 8:
		#    audio = self.stretch(audio, sr, rate=0.8)
		elif augtype == 9:
			audio = self.pitch(audio, sr)
		elif augtype == 10:
			audio = self.dyn_change(audio)
		elif augtype == 11:
			audio = self.speedNpitch(audio)
		return audio

	# （6）添加白噪声
	def noise(self, audio):
		noise_amp = 0.05 * np.random.uniform() * np.amax(audio)
		audio = audio + noise_amp * np.random.normal(size=audio.shape[0])
		return audio

	# （7）随机移动
	def shift(self, audio):
		s_range = int(np.random.uniform(low=-5, high=5) * 1000)
		return np.roll(audio, s_range)

	# （8）时域拉伸
	def stretch(self, audio, sr, rate=0.8):
		audio = audio.reshape(audio.shape[1], )
		audio = librosa.effects.time_stretch(audio, rate=0.8)
		audio = np.random.choice(audio, sr * 3 + 240)
		return audio.reshape(1, audio.shape[0])

	# （9）音调变换，该方法强调高音
	def pitch(self, audio, sr):
		audio = audio.reshape(audio.shape[1], )
		bins_per_octave = 12
		pitch_pm = -2
		pitch_change = pitch_pm * 2 * (np.random.uniform())
		audio = librosa.effects.pitch_shift(audio.astype('float64'),
											sr=sr, n_steps=pitch_change,
											bins_per_octave=bins_per_octave)
		return audio.reshape(1, audio.shape[0])

	# （10）动态随机变化
	def dyn_change(self, audio):
		dyn_change = np.random.uniform(low=-0.5, high=7)
		return (audio * dyn_change)

	# （11）加速(压缩)和音调变换
	def speedNpitch(self, audio):
		audio = audio.reshape(audio.shape[1], )
		length_change = np.random.uniform(low=0.8, high=1)
		speed_fac = 1.2 / length_change
		tmp = np.interp(np.arange(0, len(audio), speed_fac), np.arange(0, len(audio)), audio)
		minlen = min(audio.shape[0], tmp.shape[0])
		audio *= 0
		audio[0:minlen] = tmp[0:minlen]
		return audio.reshape(1, audio.shape[0])

	def __len__(self):
		return len(self.data_list)

	def add_rev(self, audio, length):
		rir_file    = random.choice(self.rir_files)
		rir, sr     = soundfile.read(rir_file)
		rir         = numpy.expand_dims(rir.astype(numpy.float),0)
		rir         = rir / numpy.sqrt(numpy.sum(rir**2))
		return signal.convolve(audio, rir, mode='full')[:,:length]

	def add_noise(self, audio, noisecat, length):
		clean_db    = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4)
		numnoise    = self.numnoise[noisecat]
		noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
		noises = []
		for noise in noiselist:
			noiselength = wave.open(noise, 'rb').getnframes() # Read the length of the noise file			
			if noiselength <= length:
				noiseaudio, _ = soundfile.read(noise)
				noiseaudio = numpy.pad(noiseaudio, (0, length - noiselength), 'wrap')
			else:
				start_frame = numpy.int64(random.random()*(noiselength-length)) # If length is enough
				noiseaudio, _ = soundfile.read(noise, start = start_frame, stop = start_frame + length) # Only read some part to improve speed
			noiseaudio = numpy.stack([noiseaudio],axis=0)
			noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4) 
			noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
			noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
		noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
		return noise + audio


class test_loader(object):
	def __init__(self, test_list, test_path, max_frames, sample_rate, num_frames, **kwargs):

		self.sample_rate = sample_rate
		self.num_frames = num_frames
		self.test_path = test_path
		# Load data & labels
		self.data_list = []
		self.data_label = []
		lines = open(test_list).read().splitlines()
		dictkeys = list(set([x.split()[0] for x in lines]))
		dictkeys.sort()
		dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
		for index, line in enumerate(lines):
			speaker_label = dictkeys[line.split()[0]]
			file_name = os.path.join(test_path, line.split()[1])
			self.data_label.append(speaker_label)
			self.data_list.append(file_name)

	def __getitem__(self, index):
		audio, sr = librosa.load(os.path.join(self.test_path, self.data_list[index]), sr=self.sample_rate)
		length = self.num_frames * (int(sr / 100)) + 240  # TODO 后续考虑设置长度为固定的3S
		if audio.shape[0] <= length:
			shortage = length - audio.shape[0]
			audio = numpy.pad(audio, (0, shortage), 'wrap')

		startframe = random.choice(range(0, audio.shape[0] - (length)))  # Choose the startframe randomly
		audio_1 = audio[int(startframe):int(startframe) + length]
		# audio_1 = numpy.stack([audio_1], axis=0).astype(np.float)
		audio_1 = numpy.expand_dims(audio_1, axis=0).astype(np.float32)

		augtype1 = random.randint(7, 11)
		audio_aug = []
		audio_aug.append(audio_1)
		audio_aug = numpy.concatenate(audio_aug, axis=0)

		return torch.FloatTensor(audio_aug), self.data_label[index]

	def load_wav(self, file):

		audio, sr = librosa.load(os.path.join(self.train_path, file), sr=self.sample_rate)
		length = self.num_frames * (int(sr / 100)) + 240  # TODO 后续考虑设置长度为固定的3S
		audiosize = audio.shape[0]
		if audio.shape[0] <= length:
			shortage = length - audio.shape[0]
			audio = numpy.pad(audio, (0, shortage), 'wrap')

		startframe = random.choice(range(0, audio.shape[0] - (length)))  # Choose the startframe randomly
		audio_1 = audio[int(startframe):int(startframe) + length]
		audio_1 = numpy.stack([audio_1], axis=0).astype(np.float)

		return torch.FloatTensor(audio_1)

	def __len__(self):
		return len(self.data_list)

