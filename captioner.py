caffe_dir = '/home/lisaanne/caffe-master/python/'
import sys
sys.path.append(caffe_dir)
import pdb
import caffe
caffe.set_mode_gpu()
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage.transform import resize
import copy

def max_choice_from_probs(softmax_inputs, no_EOS=False, prev_word = None):
  # if no_EOS True, then the next word will not be the end of the sentence
  if prev_word:
    softmax_inputs[prev_word] = 0
  if no_EOS:
      return np.argmax(softmax_inputs[1:]) + 1
  else:
    return np.argmax(softmax_inputs)

def random_choice_from_probs(softmax_inputs, temp=1, already_softmaxed=False, no_EOS=False, prev_word = None):
  #TODO: max_choice and random_choice need same inputs...
  if already_softmaxed:
    probs = softmax_inputs
    assert temp == 1
  else:
    probs = softmax(softmax_inputs, temp)
  r = random.random()
  cum_sum = 0.
  for i, p in enumerate(probs):
    cum_sum += p
    if cum_sum >= r: return i
  return 1  # return UNK?

class Captioner(object):

  def make_feature_extractor_net(self, feature_proto, feature_weights):
    #TODO: assertions to make sure feature_extractor_net right dimensions
    self.feature_extractor_net = caffe.Net(feature_proto, feature_weights, caffe.TEST)

  def make_sentence_generation_net(self, generation_proto, generation_weights):
    #TODO: assertions to make sure sentence_generation_net right dimensions
    self.sentence_generation_net = caffe.Net(generation_proto, generation_weights, caffe.TEST)

  def __init__(self, generation_proto, generation_weights, 
               feature_proto=None, feature_weights=None,
               init_net=None, init_weights= None,
               generation_method='max', beam_size=None, temperature=1,
               sentence_generation_cont_in = 'cont_sentence', sentence_generation_sent_in = 'input_sentence',
               sentence_generation_feature_in = 'image_features', feature_extractor_in = 'data',
               sentence_generation_out='probs',
               max_length = 50, vocab_file='vocab.txt', device_id = 0,
               hidden_inputs=None, hidden_outputs=None, init='zero_init'):

    caffe.set_device(device_id)
    generation_methods = {'max': max_choice_from_probs, 'sample': random_choice_from_probs, 'beam': random_choice_from_probs}
    assert generation_method in generation_methods.keys()
    self.generation_method = generation_method
    self.beam_size = beam_size
    self.temperature = temperature
    self.word_sample_method = generation_methods[generation_method]

    self.fe_out = None 
    self.fe_in = feature_extractor_in
    self.sg_out = sentence_generation_out
    self.sg_cont_in = sentence_generation_cont_in
    self.sg_sent_in = sentence_generation_sent_in
    self.sg_feature_in = sentence_generation_feature_in

    self.max_length = max_length

    self.make_sentence_generation_net(generation_proto, generation_weights)
    self.feature_extractor_net = None
    if feature_proto:
      self.make_feature_extractor_net(feature_proto, feature_weights)
      image_data_shape = self.feature_extractor_net.blobs[self.fe_in].data.shape
      self.transformer = caffe.io.Transformer({self.fe_in: image_data_shape})
      channel_mean = np.zeros(image_data_shape[1:])
      channel_mean_values = [104., 117., 124.]
      #assert channel_mean.shape[0] == len(channel_mean_values)
      if channel_mean.shape[0] == len(channel_mean_values):
        for channel_index, mean_val in enumerate(channel_mean_values):
          channel_mean[channel_index, ...] = mean_val
        self.transformer.set_mean('data', channel_mean)
        self.transformer.set_channel_swap('data', (2, 1, 0))
        self.transformer.set_transpose('data', (2, 0, 1))
      else:
        print "Warning: did not set transformer; assume that image features do not need to be transformed.\n"

    vocab = open(vocab_file).readlines()
    vocab = [v.strip() for v in vocab]
    self.vocab = ['<unk>', '<EOS>'] + vocab
    self.hidden_inputs = hidden_inputs
    self.hidden_outputs = hidden_outputs
    if self.hidden_inputs:
      self.init = init
      self.init_net = caffe.Net(init_net, init_weights, caffe.TEST)
     
  def num_to_words(self, cap):
    if cap[-1] == self.vocab.index('<EOS>'):
      cap = cap[:-1]
    words = [self.vocab[i] for i in cap]
    return ' '.join(words) + '.'

  def preprocess_image(self, image, oversample=False, 
                       fully_convolutional=False): #TODO: currently designed for CUB dataset; need to make more general
    if type(image) in (str, unicode):
      try:
        image = plt.imread(image)
      except: 
        #This fixes a very strange but I had with one image in MSCOCO...
        print "Loading image with PIL"
        img = Image.open(str(image))
        b = type(np.array(img))
        a = np.array(img)
        del img
        img = a.astype(np.float32)
        image = img
    im_size = 512
    im_crop = 448
    image = image * 1. #make floats...

    if len(image.shape) == 2:
      image = np.tile(image[:, :, np.newaxis], (1, 1, 3))

    if fully_convolutional:
      return self.transformer.preprocess('data', image)

    image = caffe.io.resize_image(image, (im_size, im_size))

    if oversample:
      oversample_ims = caffe.io.oversample([image], (im_crop, im_crop))
      preprocessed_image = []
      for im in oversample_ims:
        preprocessed_image.append(self.transformer.preprocess('data', im))
    else:
      crop_edge = (im_size - im_crop) / 2
      cropped_image = image[crop_edge:-crop_edge, crop_edge:-crop_edge,...]
      preprocessed_image = self.transformer.preprocess('data', cropped_image)

    return preprocessed_image

  def set_caption_batch_size(self, batch_size):
    net = self.sentence_generation_net
    net.blobs[self.sg_cont_in].reshape(1, batch_size)
    net.blobs[self.sg_sent_in].reshape(1, batch_size, 1)
    for f_in in self.sg_feature_in:
      net.blobs[f_in].reshape(batch_size, *net.blobs[f_in].data.shape[1:])

    if self.hidden_inputs: #TODO: Figure out cleaner way to do this
      for hidden_input in self.hidden_inputs:
        net.blobs[hidden_input].reshape(1, batch_size, net.blobs[hidden_input].data.shape[-1])

    net.reshape()

  def set_init_batch_size(self, batch_size):
    net = self.init_net
    net.blobs['image_data'].reshape(batch_size, net.blobs['image_data'].shape[-1])  
    net.reshape()

  def compute_descriptors(self, image_list, feature_extractor_out='fc8', 
                          oversample=False, fully_convolutional=False):

    self.fe_out = feature_extractor_out
    net = self.feature_extractor_net
    batch = np.zeros_like(net.blobs[self.fe_in].data)

    batch_shape = batch.shape
    batch_size = batch_shape[0]
    descriptors_shape = (len(image_list), ) + \
        net.blobs[self.fe_out].data.shape[1:]
    descriptors = np.zeros(descriptors_shape)
    if oversample:
      assert batch_size % 10 == 0
      batch_size = batch_size / 10

    for batch_start_index in range(0, len(image_list), batch_size):
      batch_list = image_list[batch_start_index:(batch_start_index + batch_size)]

      for batch_index, image_path in enumerate(batch_list):
        if oversample:
          batch[batch_index*10: batch_index*10+10] = np.array(self.preprocess_image(image_path, oversample=True))
        else:
          batch[batch_index:(batch_index + 1)] = self.preprocess_image(image_path, oversample=False, fully_convolutional=fully_convolutional)

      current_batch_size = min(batch_size, len(image_list) - batch_start_index)
      print 'Computing descriptors for images %d-%d of %d' % \
          (batch_start_index, batch_start_index + current_batch_size - 1,
           len(image_list))
      net.blobs[self.fe_in].data[...] = batch
      net.forward()
      if oversample:
        for batch_index in range(current_batch_size):
          descriptors[batch_start_index+batch_index] = np.mean(net.blobs[self.fe_out].data[batch_index*10: batch_index*10+10], axis=0)
      else:
        descriptors[batch_start_index:(batch_start_index + current_batch_size)] = \
          net.blobs[self.fe_out].data[:current_batch_size]
    return descriptors

  def init_zeros(self, descriptor):
    hidden_unit_in = np.zeros_like(self.sentence_generation_net.blobs['lstm1_h0'].data)
    cell_unit_in = np.zeros_like(self.sentence_generation_net.blobs['lstm1_c0'].data)
    return hidden_unit_in, cell_unit_in

  def sample_captions(self, features):
 
    '''
      features: list of features or ndarray of features with dimentions num_samples X feature_dim
    '''
     
    feature_array = [] 
    for feature in features: 
      if isinstance(feature, list):
        feature = np.array(feature)
      feature_array.append(feature)
    features = feature_array
    if not isinstance(features[0], np.ndarray):
      raise Exception("Descriptors must be either a list of numpy arrays, or a numpy array of dimensions samples X feature dim")

    eos_idx = self.vocab.index('<EOS>')
    net = self.sentence_generation_net
    cont_in = self.sg_cont_in
    sent_in = self.sg_sent_in
    features_in = self.sg_feature_in
    sg_out = self.sg_out

    batch_size = features[0].shape[0]
    self.set_caption_batch_size(batch_size)
    if self.hidden_inputs:
      self.set_init_batch_size(batch_size)

    cont_input = np.zeros_like(net.blobs[cont_in].data)
    word_input = np.zeros_like(net.blobs[sent_in].data)
    for feature_in, feature in zip(features_in, features):
      assert feature.shape == net.blobs[feature_in].data.shape

    outputs = []
    output_captions = [[] for b in range(batch_size)]
    output_probs = [[] for b in range(batch_size)]
    caption_index = 0
    num_done = 0

    if self.hidden_inputs:
      #init
      hidden_inputs = []
      if self.init == 'zero_init':
        for hidden_input in self.hidden_inputs:
          hidden_inputs.append(self.init_zeros(features))
      elif self.init == 'init_net':
        self.init_net.blobs['image_data'].data[...] = features[0]
        self.init_net.forward()
        for hidden_input in self.hidden_inputs:
          hidden_inputs.append(copy.deepcopy(self.init_net.blobs[hidden_input].data))

    while num_done < batch_size and caption_index < self.max_length:
      if caption_index == 0:
        cont_input[:] = 0
      elif caption_index == 1:
        cont_input[:] = 1

      if caption_index == 0:
        word_input[:] = 0
      else:
        for index in range(batch_size):
          word_input[0, index] = \
              output_captions[index][caption_index - 1] if \
              caption_index <= len(output_captions[index]) else 0

      for feature_in, feature in zip(features_in, features):
        net.blobs[feature_in].data[...] = feature
      net.blobs[cont_in].data[...] = cont_input
      net.blobs[sent_in].data[...] = word_input

      if self.hidden_inputs: #TODO: Cleaner way to access these blobs
        for hidden_input_name, hidden_input in zip(self.hidden_inputs, hidden_inputs):
          net.blobs[hidden_input_name].data[...] = hidden_input 
      
      net.forward()
      net_output_probs = copy.deepcopy(net.blobs[sg_out].data[0])
      #pdb.set_trace()

      if self.hidden_inputs: #TODO: Cleaner way to access these blobs
        hidden_inputs = []
        for hidden_output_name in self.hidden_outputs:
          hidden_inputs.append(copy.deepcopy(net.blobs[hidden_output_name].data[...])) 

      samples = [
          self.word_sample_method(dist)
          for dist in net_output_probs
      ]
       
      for index, next_word_sample in enumerate(samples):
        # If the caption is empty, or non-empty but the last word isn't EOS,
        # predict another word.
        if not output_captions[index] or output_captions[index][-1] != eos_idx:
          output_captions[index].append(next_word_sample)
          output_probs[index].append(net_output_probs[index, next_word_sample])
          if next_word_sample == eos_idx: num_done += 1
      #print 'Time to append EOS: %f' %(time.time()-t)
      sys.stdout.write('\r%d/%d done after word %d' %
          (num_done, batch_size, caption_index))
      sys.stdout.flush()
      caption_index += 1
    sys.stdout.write('\n')

    return output_captions, output_probs

  def es_sample_captions(self, features_n, feature_p):
 
    '''
      features: list of features or ndarray of features with dimentions num_samples X feature_dim
    '''
     
    def update_features(features):
      feature_array = [] 
      for feature in features: 
        if isinstance(feature, list):
          feature = np.array(feature)
        feature_array.append(feature)
      features = feature_array
      if not isinstance(features[0], np.ndarray):
        raise Exception("Descriptors must be either a list of numpy arrays, or a numpy array of dimensions samples X feature dim")
      return features

    features_p = update_features(features_p)
    features_n = update_features(features_n)

    eos_idx = self.vocab.index('<EOS>')
    cont_in = self.sg_cont_in
    sent_in = self.sg_sent_in
    features_in = self.sg_feature_in
    sg_out = self.sg_out

    batch_size = features[0].shape[0]*2
    slice_point = features[0].shape[0]
    self.set_caption_batch_size(batch_size)
    if self.hidden_inputs:
      self.set_init_batch_size(batch_size)

    cont_input = np.zeros_like(net.blobs[cont_in].data)
    word_input = np.zeros_like(net.blobs[sent_in].data)
    for feature_in, feature in zip(features_in, features):
      assert feature.shape == net.blobs[feature_in].data.shape

    outputs = []
    output_captions = [[] for b in range(batch_size)]
    output_probs = [[] for b in range(batch_size)]
    caption_index = 0
    num_done = 0

    if self.hidden_inputs:
      #init
      hidden_inputs = []
      if self.init == 'zero_init':
        for hidden_input in self.hidden_inputs:
          hidden_inputs.append(self.init_zeros(features))
      elif self.init == 'init_net':
        self.init_net.blobs['image_data'].data[...] = features[0]
        self.init_net.forward()
        for hidden_input in self.hidden_inputs:
          hidden_inputs.append(copy.deepcopy(self.init_net.blobs[hidden_input].data))

    while num_done < batch_size and caption_index < self.max_length:
      if caption_index == 0:
        cont_input[:] = 0
      elif caption_index == 1:
        cont_input[:] = 1

      if caption_index == 0:
        word_input[:] = 0
      else:
        for index in range(batch_size):
          word_input[0, index] = \
              output_captions[index][caption_index - 1] if \
              caption_index <= len(output_captions[index]) else 0

      for feature_in, feature in zip(features_in, features_p):
        net.blobs[feature_in].data[:slice_point,...] = feature
      for feature_in, feature in zip(features_in, features_n):
        net.blobs[feature_in].data[slice_point:,...] = feature
      net.blobs[cont_in].data[...] = cont_input
      net.blobs[sent_in].data[...] = word_input

      if self.hidden_inputs: #TODO: Cleaner way to access these blobs
        for hidden_input_name, hidden_input in zip(self.hidden_inputs, hidden_inputs):
          net.blobs[hidden_input_name].data[...] = hidden_input 
      
      net.forward()
      net_output_probs = copy.deepcopy(net.blobs[sg_out].data[0])
      #pdb.set_trace()

      if self.hidden_inputs: #TODO: Cleaner way to access these blobs
        hidden_inputs = []
        for hidden_output_name in self.hidden_outputs:
          hidden_inputs.append(copy.deepcopy(net.blobs[hidden_output_name].data[...])) 

      samples = [
          self.word_sample_method(dist)
          for dist in net_output_probs
      ]
       
      for index, next_word_sample in enumerate(samples):
        # If the caption is empty, or non-empty but the last word isn't EOS,
        # predict another word.
        if not output_captions[index] or output_captions[index][-1] != eos_idx:
          output_captions[index].append(next_word_sample)
          output_probs[index].append(net_output_probs[index, next_word_sample])
          if next_word_sample == eos_idx: num_done += 1
      #print 'Time to append EOS: %f' %(time.time()-t)
      sys.stdout.write('\r%d/%d done after word %d' %
          (num_done, batch_size, caption_index))
      sys.stdout.flush()
      caption_index += 1
    sys.stdout.write('\n')

    return output_captions, output_probs

