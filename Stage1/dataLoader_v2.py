import pdb

import torch, numpy, random, os, math, glob, soundfile
from torch.utils.data import Dataset, DataLoader
from scipy import signal
import librosa
import numpy as np


class train_loader(Dataset):
    def __init__(self, max_frames, train_list, train_path, musan_path, num_frames, sample_rate, **kwargs):
        self.max_frames = max_frames
        self.num_frames = num_frames
        self.sample_rate = sample_rate
        self.train_path = train_path
        self.noisetypes = ['noise', 'speech', 'music']  # Type of noise
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}  # The range of SNR
        self.noiselist = {}
        augment_files = glob.glob(os.path.join(musan_path, '*/*/*/*.wav'))  # All noise files in list
        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)  # All noise files in dic
        self.rir_files = numpy.load('rir.npy')  # Load the rir file
        # Load data & labels
        self.data_list = []
        self.data_label = []
        lines = open(train_list).read().splitlines()
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
        for index, line in enumerate(lines):
            speaker_label = dictkeys[line.split()[0]]
            #file_name = os.path.join(train_path, line.split()[1])
            self.data_label.append(speaker_label)
            self.data_list.append(line)

    def __getitem__(self, index):
        # Read the utterance and randomly select the segment
        ls = self.data_list[index].split()
        audio_1, sr = librosa.load(os.path.join(self.train_path, ls[1]), sr=self.sample_rate)
        audio_2, sr = librosa.load(os.path.join(self.train_path, ls[2]), sr=self.sample_rate)

        # min_length = min(audio_1.shape[0], audio_2.shape[0])
        # print('min_length:',min_length)
        # audio_1 = audio_1[:min_length]
        # audio_2 = audio_2[:min_length]

        audio_1 = numpy.stack([audio_1], axis=0).astype(np.float)  # TODO 这里的stack什么意思？
        audio_2 = numpy.stack([audio_2], axis=0).astype(np.float)
        # Data Augmentation

        augtype1 = random.randint(7, 11)  # 不进行数据增强
        augtype2 = random.randint(7, 11)  # 不进行数据增强
        while augtype1 == augtype2:
            augtype2 = random.randint(7, 11)
        audio_aug = []
        audio_aug.append(self.random_aug(augtype1, audio_1, sr))
        # audio_aug.append(self.random_aug(augtype1, audio, sr))
        audio_aug.append(self.random_aug(augtype2, audio_2, sr))
        audio_aug = numpy.concatenate(audio_aug, axis=0)  # Concate and return
        # return torch.FloatTensor(audio_aug), self.data_label[index]
        return torch.FloatTensor(audio_aug), self.data_label[index]

    def __len__(self):
        return len(self.data_list)

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
        elif augtype == 5:  # Noise
            audio = self.add_noise(audio, 'noise')
        elif augtype == 7:
            audio = self.noise(audio)
        elif augtype == 8:
            audio = self.shift(audio)
        # elif augtype == 8:
        #     audio = self.stretch(audio, sr, rate=0.8)
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

    def augment_wav(self, audio, augment):
        if augment['rir_filt'] is not None:
            rir = numpy.multiply(augment['rir_filt'], pow(10, 0.1 * augment['rir_gain']))
            audio = signal.convolve(audio, rir, mode='full')[:len(audio)]
        if augment['add_noise'] is not None:
            noiseaudio = loadWAV(augment['add_noise'], self.max_frames).astype(numpy.float)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2) + 1e-4)
            clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
            noise = numpy.sqrt(10 ** ((clean_db - noise_db - augment['add_snr']) / 10)) * noiseaudio
            audio = audio + noise
        else:
            audio = numpy.expand_dims(audio, 0)
        return audio


class test_loader(Dataset):
    def __init__(self, max_frames, test_list, test_path, musan_path, num_frames, sample_rate, **kwargs):
        # Load data & labels
        self.sample_rate = sample_rate
        self.num_frames = num_frames
        self.test_path = test_path
        self.data_list = []
        self.data_label = []
        lines = open(test_list).read().splitlines()
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
        for index, line in enumerate(lines):
            speaker_label = dictkeys[line.split()[0]]
            #file_name = os.path.join(test_path, line.split()[1])
            self.data_label.append(speaker_label)
            self.data_list.append(line)

    def __getitem__(self, index):
        # Read the utterance and randomly select the segment
        ls = self.data_list[index].split()
        audio_1, sr = librosa.load(os.path.join(self.test_path, ls[1]), sr=self.sample_rate)
        audio_2, sr = librosa.load(os.path.join(self.test_path, ls[2]), sr=self.sample_rate)

        audio_1 = numpy.stack([audio_1], axis=0).astype(np.float)  # TODO 这里的stack什么意思？
        audio_2 = numpy.stack([audio_2], axis=0).astype(np.float)

        min_length = min(audio_1.shape[0], audio_2.shape[0])
        audio_1 = audio_1[:min_length]
        audio_2 = audio_2[:min_length]

        audio_aug = []
        audio_aug.append(audio_1)
        audio_aug.append(audio_2)
        audio_aug = numpy.concatenate(audio_aug, axis=0)  # Concate and return
        # return torch.FloatTensor(audio_aug), self.data_label[index]
        return torch.FloatTensor(audio_aug), self.data_label[index]

    def __len__(self):
        return len(self.data_list)


def loadWAV(filename, max_frames):
    max_audio = max_frames * 160 + 240  # 240 is for padding, for 15ms since window is 25ms and step is 10ms.
    audio, _ = soundfile.read(filename)
    audiosize = audio.shape[0]
    if audiosize <= max_audio:  # Padding if the length is not enough
        shortage = math.floor((max_audio - audiosize + 1) / 2)
        audio = numpy.pad(audio, (shortage, shortage), 'wrap')
        audiosize = audio.shape[0]
    startframe = numpy.int64(
        random.random() * (audiosize - max_audio))  # Randomly select a start frame to extract audio
    feat = numpy.stack([audio[int(startframe):int(startframe) + max_audio]], axis=0)
    return feat


def loadWAVSplit(filename, max_frames):  # Load two segments
    max_audio = max_frames * 160 + 240
    audio, _ = soundfile.read(filename)
    audiosize = audio.shape[0]
    if audiosize <= max_audio:
        shortage = math.floor((max_audio - audiosize) / 2)
        audio = numpy.pad(audio, (shortage, shortage), 'wrap')
        audiosize = audio.shape[0]
    randsize = audiosize - (max_audio * 2)  # Select two segments
    startframe = random.sample(range(0, randsize), 2)
    startframe.sort()
    startframe[1] += max_audio  # Non-overlapped two segments
    startframe = numpy.array(startframe)
    numpy.random.shuffle(startframe)
    feats = []
    for asf in startframe:  # Startframe[0] means the 1st segment, Startframe[1] means the 2nd segment
        feats.append(audio[int(asf):int(asf) + max_audio])
    feat = numpy.stack(feats, axis=0)
    return feat


def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)


def get_loader(args):  # Define the data loader
    trainLoader = train_loader(**vars(args))
    trainLoader = torch.utils.data.DataLoader(
        trainLoader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        prefetch_factor=5,
    )
    return trainLoader


def get_testloader(args):  # Define the data loader
    testLoader = test_loader(**vars(args))
    testLoader = torch.utils.data.DataLoader(
        testLoader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        prefetch_factor=5,
    )
    return testLoader