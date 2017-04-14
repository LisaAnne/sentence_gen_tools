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
sys.path.append('utils/')
from python_utils import *
import itertools

def max_choice_from_probs(softmax_inputs, no_EOS=False, prev_word=None):
  #TODO: need to fix argument in max choice from probs, random choice from probs, and topK choice from probs
  if prev_word:
    softmax_inputs[prev_word] = 0
  if no_EOS:
    softmax_inputs[0] = 0 #EOS token
  return np.argmax(softmax_inputs)

def topK_choice_from_probs(softmax_inputs, k=1):
  return np.argsort(softmax_inputs)[::-1][:k]

def random_choice_from_probs(softmax_inputs, temp=1, already_softmaxed=False, no_EOS=False, prev_word = None):
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
               max_length = 50, vocab_file='data/vocab.txt', device_id = 0,
               hidden_inputs=None, hidden_outputs=None, init='zero_init',
               prev_word_restriction=False):

    caffe.set_device(device_id)
    generation_methods = {'max': max_choice_from_probs, 'sample': random_choice_from_probs, 'beam': topK_choice_from_probs}
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
      if not feature_weights:
        feature_weights = generation_weights 
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
    #this needs to be changed -- commenting out in between rounds for now
    #self.vocab = ['<unk>', '<EOS>'] + vocab
    self.vocab = ['<EOS>'] + vocab
    self.vocab_dict = {}
    for i, w in enumerate(self.vocab):
      self.vocab_dict[w] = i

    self.hidden_inputs = hidden_inputs
    self.hidden_outputs = hidden_outputs
    self.init = init
    if init_net:
      self.init_net = caffe.Net(init_net, init_weights, caffe.TEST)
    else:
      self.init_net = None 
    self.prev_word_restriction = prev_word_restriction
      
  def num_to_words(self, cap):
    if cap[-1] == self.vocab.index('<EOS>'):
      cap = cap[:-1]
    words = [self.vocab[i] for i in cap]
    return ' '.join(words) + '.'

  def words_to_num(self, cap):
    return tokenize_text(cap, self.vocab_dict) 

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

  def init_zeros(self, init_in):
    hidden_init = np.zeros_like(self.sentence_generation_net.blobs[init_in].data)
    return hidden_init

  def sample_captions(self, features, batch_size=None):
 
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

    if not batch_size:
      batch_size = features[0].shape[0]
      self.set_caption_batch_size(batch_size)
      if self.init_net:
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

      if (self.prev_word_restriction) & (caption_index > 0):
        prev_words = [output_caption[-1] for output_caption in output_captions]
      else: prev_words = [False]*len(output_captions)

      samples = [
          self.word_sample_method(dist, prev_word=prev_word)
          for dist, prev_word in zip(net_output_probs, prev_words)
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

  def es_sample_captions(self, features_p, features_n):
 
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
    net = self.sentence_generation_net
    cont_in = self.sg_cont_in
    sent_in = self.sg_sent_in
    features_in = self.sg_feature_in
    sg_out = self.sg_out

    batch_size = features_p[0].shape[0]*2
    slice_point = features_p[0].shape[0]
    self.set_caption_batch_size(batch_size)
    if self.init_net:
      self.set_init_batch_size(batch_size)

    cont_input = np.zeros_like(net.blobs[cont_in].data)
    word_input = np.zeros_like(net.blobs[sent_in].data)

    outputs = []
    output_captions = [[] for b in range(slice_point)]
    output_probs = [[] for b in range(slice_point)]
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
        for index in range(slice_point):
          word_input[0, index] = \
              output_captions[index][caption_index - 1] if \
              caption_index <= len(output_captions[index]) else 0
          word_input[0, slice_point+index] = \
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
      output_probs_p = net_output_probs[:slice_point]
      output_probs_n = net_output_probs[slice_point:]
      l = 0.5 
      #output_probs_div = l*output_probs_p+((1-l)*(output_probs_p/output_probs_n))
      #output_probs_div = l*np.log(output_probs_p) - (1-l)*np.log(output_probs_n)
      output_probs_div = np.log(output_probs_p) - (1-l)*np.log(output_probs_n)

      if self.hidden_inputs: #TODO: Cleaner way to access these blobs
        hidden_inputs = []
        for hidden_output_name in self.hidden_outputs:
          hidden_inputs.append(copy.deepcopy(net.blobs[hidden_output_name].data[...])) 

      samples = [
          self.word_sample_method(dist)
          for dist in output_probs_div 
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

  def get_word_gen_prob(self, input_values, input_names, output='probs', batch_size=1):
    net = self.sentence_generation_net
    self.set_caption_batch_size(batch_size)
    for input_value, input_name in zip(input_values, input_names):
      self.sentence_generation_net.blobs[input_name].data[...] = input_value
    net.forward()
    return net.blobs[output].data.copy()

  def beam_search(self, features, beam_size=5):
    '''
      features: list of features or ndarray of features with dimentions num_samples X feature_dim
      beam_size: Size of beam
      TODO: Check again; this seems to work okay, but I wrote it quickly.  Before using it to generate results for a paper, I would double check this
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
    all_features_rep = []
    for feature in features:
      features_rep = np.zeros(((batch_size*beam_size,) + feature.shape[1:]))
      for i in range(batch_size):
        for j in range(beam_size):
          features_rep[i*beam_size:i*beam_size+j,...] = feature[i,:]
      all_features_rep.append(features_rep)
    features = all_features_rep
    self.set_caption_batch_size(batch_size*beam_size)
    if self.init_net:
      self.set_init_batch_size(batch_size*beam_size)

    cont_input = np.zeros_like(net.blobs[cont_in].data)
    word_input = np.zeros_like(net.blobs[sent_in].data)
    for feature_in, feature in zip(features_in, features):
      assert feature.shape == net.blobs[feature_in].data.shape

    output_captions = [[[] for bs in range(beam_size)] for b in range(batch_size)]
    output_probs = [[[] for bs in range(beam_size)] for b in range(batch_size)]
    caption_index = 0
    num_done = 0

    assert self.hidden_inputs
    assert self.hidden_inputs

    #init
    hidden_inputs = []
    if self.init == 'zero_init':
      for hidden_input in self.hidden_inputs:
        hidden_inputs.append(self.init_zeros(hidden_input))
    elif self.init == 'init_net':
      self.init_net.blobs['image_data'].data[...] = features[0]
      self.init_net.forward()
      for hidden_input in self.hidden_inputs:
        hidden_inputs.append(copy.deepcopy(self.init_net.blobs[hidden_input].data))
      
    while num_done < batch_size*beam_size and caption_index < self.max_length:
      if caption_index == 0:
        cont_input[:] = 0
      elif caption_index == 1:
        cont_input[:] = 1

      if caption_index == 0:
        word_input[:] = 0
      else:
        for index_cap in range(batch_size):
          for index_beam in range(beam_size):
            word_input[0, index_cap*beam_size+index_beam] = \
                output_captions[index_cap][index_beam][caption_index - 1] if \
                caption_index <= len(output_captions[index_cap][index_beam]) else 0

      for feature_in, feature in zip(features_in, features):
        net.blobs[feature_in].data[...] = feature
      net.blobs[cont_in].data[...] = cont_input
      net.blobs[sent_in].data[...] = word_input

      for hidden_input_name, hidden_input in zip(self.hidden_inputs, hidden_inputs):
        net.blobs[hidden_input_name].data[...] = hidden_input 

      net.forward()
      net_output_probs = copy.deepcopy(net.blobs[sg_out].data[0])
      #pdb.set_trace()


      samples = [
          self.word_sample_method(dist, k=beam_size)
          for dist in net_output_probs
      ]

      probs = [
          [net_output_probs[idx, w] 
          for w in sample] for
          idx, sample in zip(range(beam_size), samples)
      ]

     
      #set when we decide which beams to take...
      for hidden_i, hidden_output_name in enumerate(self.hidden_outputs):
        hidden_inputs[hidden_i][...] = 0
      for index_cap, beam_captions in enumerate(output_captions):
        caption_samples = samples[index_cap*beam_size:index_cap*beam_size+beam_size]
        caption_probs = probs[index_cap*beam_size:index_cap*beam_size+beam_size]
        beam_expansions = []
        beam_expansions_probs = []
        for index_beam, caption in enumerate(beam_captions):
          # If the caption is empty, or non-empty but the last word isn't EOS,
          # predict another word.
          beam_samples = caption_samples[index_beam]
          beam_probs = caption_probs[index_beam]
          if not caption or caption[-1] != eos_idx:
            prob = output_probs[index_cap][index_beam]
            for beam_prob, beam_sample in zip(beam_probs, beam_samples):
              beam_expansions.append(caption + [beam_sample])
              beam_expansions_probs.append(prob + [beam_prob])
          else:
            prob = output_probs[index_cap][index_beam]
            beam_expansions.append(caption)
            beam_expansions_probs.append(prob)

        #choose the best expansions based off sum of log probs
        #if caption index == 0, thinks are different because will have five of the same beam...
        if caption_index == 0:
          beam_expansions_probs[beam_size:] = [[0.001] for b in \
                            beam_expansions_probs[beam_size:]]

        log_probs = [np.sum(np.log(beam_prob)) for beam_prob in beam_expansions_probs]
        best_idxs = np.argsort(log_probs)[::-1][:beam_size]

        for idx, best_idx in enumerate(best_idxs):
          output_captions[index_cap][idx] = beam_expansions[best_idx]
          output_probs[index_cap][idx] = beam_expansions_probs[best_idx]
          orig_beam = best_idx % beam_size
          hidden_index = index_cap*beam_size + orig_beam
          for hidden_i, hidden_output_name in enumerate(self.hidden_outputs):
            tmp = net.blobs[hidden_output_name].data[0,hidden_index,:].copy() 
            hidden_inputs[hidden_i][0,index_cap*beam_size+idx,:] = tmp 
      num_done = 0
      for beam_captions in output_captions:
        for caption in beam_captions:
          if caption[-1] == 0:
            num_done += 1
      sys.stdout.write('\r%d/%d done after word %d' %
          (num_done, batch_size, caption_index))
      sys.stdout.flush()
      caption_index += 1
    sys.stdout.write('\n')

    return output_captions, output_probs
 
  def caption_images(self, descriptors, im_list, batch_size=100, caption_method='sample_captions'):  
    final_captions = {}
    num_images = len(im_list)
    descriptor_shape = descriptors.values()[0].shape[0]
    for i in range(0, num_images, batch_size):
      print 'Captioning image: %d/%d' %(i, num_images)
      mini_batch_size = min(batch_size, num_images-i)
      batch_images = im_list[i:i+mini_batch_size]
      batch_descriptors = np.zeros((mini_batch_size, descriptor_shape))
      for count_im, im in enumerate(batch_images):
        batch_descriptors[count_im,:] = descriptors[im]
      network_inputs = [batch_descriptors]
      if caption_method == 'sample_captions':
        output_caps, output_probs = self.sample_captions(network_inputs)
        for im, cap in zip(batch_images, output_caps):
          final_captions[im] = self.num_to_words(cap)
      else:
        raise Exception("Can only use this function if generating captions via sampling")
    return final_captions 










 
