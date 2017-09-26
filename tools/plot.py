# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import argparse
import ast
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import cnn_visualizer

def get_dataframe(file_results_path):
  raw_results_data = []
  results_file = open(file_results_path, 'rb')
  for line in results_file:
    if line.startswith("#") or line == None:
      continue
    raw_results_data.append(line)
  results_file.close()

  results_data = []
  for line in raw_results_data:
    line_info = line.split("|")
    res_type = line_info[0]
    data = ast.literal_eval(line_info[1])
    data["res_type"] = res_type
    results_data.append(data)
  
  dataframe = pd.DataFrame(results_data)
  return dataframe


def get_results_data(file_results_path):
  raw_results_data = []
  results_file = open(file_results_path, 'rb')
  for line in results_file:
    if line.startswith("#") or line == None:
      continue
    raw_results_data.append(line)
  results_file.close()

  results_data = {}

  for line in raw_results_data:
    line_info = line.split("|")
    res_type = line_info[0]
    data = ast.literal_eval(line_info[1])
    if not res_type in results_data:
      results_data[res_type] = []
    results_data[res_type].append(data)
    
  return results_data

def results_data_to_numpy(results_data):
  data = []
  for res_type in results_data:

    for res_data in results_data[res_type]:
      line_data = []
      line_data.append(res_type)                              #0
      line_data.append(res_data['descriptor_type'])           #1
      line_data.append(res_data['feature_type'])              #2
      line_data.append(res_data['thumbnail_size'][0])         #3
      line_data.append(res_data['time_cluster_fit'])          #4
      line_data.append(res_data['time_classificator_fit'])    #5
      line_data.append(res_data['n_sample_descriptors'])      #6
      line_data.append(res_data['n_sample_images'])           #7
      line_data.append(res_data['k_value'])                   #8
      line_data.append(res_data['time_total'])                #9
      line_data.append(res_data['accuracy'])                  #10
      line_data.append(res_data['F1'])                        #11
      line_data.append(res_data['recall'])                    #12
      line_data.append(res_data['precision'])                 #13
      line_data.append(str(res_data['feature_type']) + '+'
       + str(res_data['descriptor_type']))                    #14
      data.append(line_data)

  return numpy.asarray(data)


def plot_accuracy_and_k_by_descriptor_size(results_data):
  print "Plotting accuracy x K"
  data_array = results_data_to_numpy(results_data)

  print "Results shape: ", data_array.shape
  print "Image sizes: ", numpy.unique(data_array[:,7])
  print "Descriptor sizes: ", numpy.unique(data_array[:,6])
  
  fig, ax = plt.subplots()
  desc_sizes = numpy.unique(data_array[:,6])

  colors = "bgrcmykw"
  for idx, desc_size in enumerate(desc_sizes):
    data_array_filtered = data_array[data_array[:,6] == desc_size]
    accuracies = data_array_filtered[:,10]
    k_values = data_array_filtered[:,8]
    print "K values: ", k_values
    ax.plot(k_values, accuracies, "-", c=colors[idx])

  ax.set_title('Tamanho da imagem');
  ax.set_xlabel('K')
  ax.set_ylabel(u'Acurácia')
  ax.set_ylim([0, 1])
  fig.savefig('plots/plot_accuracy1.png')
  #plt.show()

def plot_bovw_accuracy_by_image_size(results_data, title_dataset, filename):
  data_array = results_data_to_numpy(results_data)
  width = 0.35
  
  fdtypes = []
  fd_types = numpy.unique(data_array[:,14]) 
  for fd_type in fd_types:
    accs = []
    thumbs = []
    stds = []
    data_array_filtered = data_array[data_array[:,14] == fd_type]
    img_sizes = numpy.unique(data_array[:,3])  
    for idx, img_size in enumerate(img_sizes):
      data_array_img = data_array_filtered[data_array_filtered[:,3] == img_size]
      img_accuracies = data_array_img[:,10]
      img_accuracies = img_accuracies.astype(numpy.float)
      accuracy = numpy.mean(img_accuracies)
      accuracy_std = numpy.std(img_accuracies)
      thumb_size = data_array_img[:,3][0]
      accs.append(accuracy)
      stds.append(accuracy_std)
      thumbs.append(int(thumb_size))
    data = (fd_type, thumbs, accs, stds)
    fdtypes.append(data)
  
  fig, ax = plt.subplots()

  bar_width = 0.15

  opacity = 0.6
  error_config = {'ecolor': '0.3'}
  index = numpy.arange(len(fdtypes[0][1]))
  colors = ['b','r', 'g', 'y', 'c', 'm', 'y', 'k', 'w'] #bgrcmykw
  patterns = ('-', '*', 'o', '\\', '+', 'o', 'O', '.')
  for idx, fdtype in enumerate(fdtypes):
    pos = index if idx==0 else index + idx*bar_width
    bars = ax.bar(pos, fdtype[2], bar_width,
                     alpha=opacity,
                     color=colors[idx],
                     yerr=fdtype[3],
                     error_kw=error_config,
                     label=fdtype[0],
                     hatch=patterns[idx])
    
  plt.xlabel(u'Tamanho da imagem')
  plt.ylabel(u'Acurácia')
  plt.title(title_dataset + u" - Acurácia VS Tamanho - " + str(data_array[0][0]))
  plt.legend()
  img_sizes = fdtypes[0][1]
  plt.xticks(index + 2.5*bar_width, img_sizes)
  plt.ylim(0.0,1.0)

  plt.tight_layout()
  fig.savefig(filename)



def plot_bovw_f1_by_image_size(results_data, title_dataset, filename):
  data_array = results_data_to_numpy(results_data)
  width = 0.35
  
  fdtypes = []
  fd_types = numpy.unique(data_array[:,14]) 
  for fd_type in fd_types:
    f1s = []
    thumbs = []
    stds = []
    data_array_filtered = data_array[data_array[:,14] == fd_type]
    img_sizes = numpy.unique(data_array[:,3])  
    for idx, img_size in enumerate(img_sizes):
      data_array_img = data_array_filtered[data_array_filtered[:,3] == img_size]
      img_f1 = data_array_img[:,11]
      img_f1 = img_f1.astype(numpy.float)
      f1 = numpy.mean(img_f1)
      f1_std = numpy.std(img_f1)
      thumb_size = data_array_img[:,3][0]
      f1s.append(f1)
      stds.append(f1_std)
      thumbs.append(int(thumb_size))
    data = (fd_type, thumbs, f1s, stds)
    fdtypes.append(data)
  
  fig, ax = plt.subplots()

  bar_width = 0.15

  opacity = 0.6
  error_config = {'ecolor': '0.3'}
  index = numpy.arange(len(fdtypes[0][1]))
  colors = ['b','r', 'g', 'y', 'c', 'm', 'y', 'k', 'w'] #bgrcmykw
  patterns = ('-', '*', 'o', '\\', '+', 'o', 'O', '.')
  for idx, fdtype in enumerate(fdtypes):
    pos = index if idx==0 else index + idx*bar_width
    bars = ax.bar(pos, fdtype[2], bar_width,
                     alpha=opacity,
                     color=colors[idx],
                     yerr=fdtype[3],
                     error_kw=error_config,
                     label=fdtype[0],
                     hatch=patterns[idx])
    
  plt.xlabel(u'Tamanho da imagem')
  plt.ylabel(u'F1')
  plt.title(title_dataset + u" - F1 VS Tamanho - " + str(data_array[0][0]))
  plt.legend()
  img_sizes = fdtypes[0][1]
  plt.xticks(index + 2.5*bar_width, img_sizes)
  plt.ylim(0.0,1.0)

  plt.tight_layout()
  fig.savefig(filename)

def plot_bovw_time_classification(results_data, filename):
  #descritor fixo (MSER+SIFT)
  #tamanho fixo (256)
  #y = tempo
  #x = (Kmeans+SVM, KMeans+OPF, OPF+OPF, OPF+SVM, OVERFEAT+SVM, OVERFEAT+OPF)
  print "Nao implementado"

def plot_bovw_time_clustering(results_data, filename):
  #descritor fixo (MSER+SIFT)
  #tamanho fixo (256)
  #y = tempo
  #x = (Kmeans+SVM, KMeans+OPF, OPF+OPF, OPF+SVM)
  print "Nao implementado"

def plot_bovw_time_total(results_data, filename):
  #descritor fixo (MSER+SIFT)
  #tamanho fixo (256)
  #y = tempo
  #x = (Kmeans+SVM, KMeans+OPF, OPF+OPF, OPF+SVM)
  print "Nao implementado"

def plot_desc_size(dataframe, filename):
  print "Plotting Descriptors Size Chart"
  grouped = dataframe.groupby(['res_type','n_sample_descriptors'])
  means = grouped.F1.mean()
  stds = grouped.F1.std()
  
  fig, ax = plt.subplots()

  bar_width = 0.15

  opacity = 0.6
  error_config = {'ecolor': '0.3'}
  index = numpy.arange(len(means.index.levels[1]))
  colors = ['b','r', 'g', 'y', 'c', 'm', 'y', 'k', 'w'] #bgrcmykw
  patterns = ('-', '*', 'o', '\\', '+', 'o', 'O', '.')
  for idx, res_type in enumerate(means.index.levels[0]):
    
    pos = index if idx==0 else index + idx*bar_width
    bars = ax.bar(pos, means[res_type], bar_width,
                     alpha=opacity,
                     color=colors[idx],
                     yerr=stds[res_type],
                     error_kw=error_config,
                     label=means.index.levels[0][idx],
                     hatch=patterns[idx])
    
  plt.xlabel(u'Quantidade de descritores')
  plt.ylabel(u'F1')
  plt.title(u"F1 x Quantidade de Descritores")
  plt.legend()
  desc_sizes = means.index.levels[1]
  plt.xticks(index + 2.5*bar_width, desc_sizes)
  plt.ylim(0.0,1.0)

  plt.tight_layout()
  fig.savefig(filename)


#Print accuracy mean and std. Grouped by feature_type
#descriptor_type and thumbnail_size
def plot_accuracy(dataframe, title_dataset, filename):
  g1 = dataframe.groupby(['feature_type','descriptor_type','thumbnail_size'])
  res = pd.DataFrame()
  res['mean'] = g1['accuracy'].mean()
  res['std'] = g1['accuracy'].std()

  fig = plt.figure()
  ax = fig.add_subplot(111)

  N = 4
  ind = numpy.arange(N)
  width = 0.15 
  opacity = 0.6
  recs_dense_sift = ax.bar(ind, res['mean']['Dense']['SIFT'], width,
                      color='k',
                      alpha=opacity,
                      yerr=res['std']['Dense']['SIFT'],
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label='Dense+SIFT',
                      hatch='-')

  recs_dense_surf = ax.bar(ind+width, res['mean']['Dense']['SURF'], width,
                      color='blue',
                      alpha=opacity,
                      yerr=res['std']['Dense']['SURF'],
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label='Dense+SURF',
                      hatch='*')

  recs_mser_sift = ax.bar(ind+(width*2), res['mean']['MSER']['SIFT'], width,
                      color='green',
                      alpha=opacity,
                      yerr=res['std']['MSER']['SIFT'],
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label='MSER+SIFT',
                      hatch='\\')

  recs_sift_sift = ax.bar(ind+(width*3), res['mean']['SIFT']['SIFT'], width,
                      color='orange',
                      alpha=opacity,
                      yerr=res['std']['SIFT']['SIFT'],
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label='SIFT+SIFT',
                      hatch='+')

  recs_surf_surf = ax.bar(ind+(width*4), res['mean']['SURF']['SURF'], width,
                      color='m',
                      alpha=opacity,
                      yerr=res['std']['SURF']['SURF'],
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label='SURF+SURF',
                      hatch='o')

  plt.xlabel(u'Tamanho da imagem')
  plt.ylabel(u'Acurácia')
  plt.title(title_dataset)
  plt.legend()
  img_sizes = res['mean'].index.levels[2]
  plt.xticks([0.4,1.4,2.4,3.4],img_sizes)
  plt.ylim(0.0,1.0)

  plt.tight_layout()
  fig.savefig(filename)

def plot_mser_sift_all(df_kmeans_svm, df_kmeans_opf, df_opf_opf, df_opf_svm, df_overfeat,  cnn_yuv_accuracy, cnn_rgb_accuracy, title_dataset, filename):
  gf_kmeans_svm = df_kmeans_svm.groupby(['feature_type','descriptor_type','thumbnail_size'])
  gf_kmeans_opf = df_kmeans_opf.groupby(['feature_type','descriptor_type','thumbnail_size'])
  gf_opf_opf = df_opf_opf.groupby(['feature_type','descriptor_type','thumbnail_size'])
  gf_opf_svm = df_opf_svm.groupby(['feature_type','descriptor_type','thumbnail_size'])
  gf_overfeat_svm = df_overfeat.T[0]
  gf_overfeat_opf = df_overfeat.T[1]

  res_kmeans_svm = pd.DataFrame()
  res_kmeans_opf = pd.DataFrame()
  res_opf_opf = pd.DataFrame()
  res_opf_svm = pd.DataFrame()
  
  res_kmeans_svm['mean'] = gf_kmeans_svm['accuracy'].mean()
  res_kmeans_svm['std'] = gf_kmeans_svm['accuracy'].std()

  res_kmeans_opf['mean'] = gf_kmeans_opf['accuracy'].mean()
  res_kmeans_opf['std'] = gf_kmeans_opf['accuracy'].std()

  res_opf_opf['mean'] = gf_opf_opf['accuracy'].mean()
  res_opf_opf['std'] = gf_opf_opf['accuracy'].std()

  res_opf_svm['mean'] = gf_opf_svm['accuracy'].mean()
  res_opf_svm['std'] = gf_opf_svm['accuracy'].std()


  data_mean_kmeans_svm = res_kmeans_svm['mean']['MSER']['SIFT'][2]
  data_std_kmeans_svm = res_kmeans_svm['std']['MSER']['SIFT'][2]

  data_mean_kmeans_opf = res_kmeans_opf['mean']['MSER']['SIFT'][2]
  data_std_kmeans_opf = res_kmeans_opf['std']['MSER']['SIFT'][2]

  data_mean_opf_opf = res_opf_opf['mean']['MSER']['SIFT'][2]
  data_std_opf_opf = res_opf_opf['std']['MSER']['SIFT'][2]

  data_mean_opf_svm = res_opf_svm['mean']['MSER']['SIFT'][2]
  data_std_opf_svm = res_opf_svm['std']['MSER']['SIFT'][2]

  data_mean_overfeat_opf = gf_overfeat_opf['accuracy']
  data_std_overfeat_opf = 0.0

  data_mean_overfeat_svm = gf_overfeat_svm['accuracy']
  data_std_overfeat_svm = 0.0

  
  data_mean_cnn_yuv = cnn_yuv_accuracy
  data_std_cnn_yuv = 0.0

  data_mean_cnn_rgb = cnn_rgb_accuracy
  data_std_cnn_rgb = 0.0

  width = 0.15 
  opacity = 0.6
  space = 0.0

  fig = plt.figure()
  ax = fig.add_subplot(111)

  N=1
  ind = numpy.arange(N)

  #colors = ['b','r', 'g', 'y', 'c', 'm', 'y', 'k', 'w']
  colors = ['b','b','b','b','b','b','b','b','b']
  patterns = ['-', '*', 'o', '\\', '+', '//', 'O', '.']
  labels = ['Kmeans+SVM','Kmeans+OPF-S','OPF-U+OPF-S','OPF-U+SVM','OVERFEAT+SVM','OVERFEAT+OPF','CNN YUV','CNN RGB']
  recs_kmeans_svm = ax.bar(ind, data_mean_kmeans_svm, width,
                      color=colors[0],
                      alpha=opacity,
                      yerr=data_std_kmeans_svm,
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label=labels[0],
                      )#hatch=patterns[0])

  recs_kmeans_opf = ax.bar(ind+width+space, data_mean_kmeans_opf, width,
                      color=colors[1],
                      alpha=opacity,
                      yerr=data_std_kmeans_opf,
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label=labels[1],
                      )#hatch=patterns[1])

  recs_opf_opf = ax.bar(ind+(width*2)+space, data_mean_opf_opf, width,
                      color=colors[2],
                      alpha=opacity,
                      yerr=data_std_opf_opf,
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label=labels[2],
                      )#hatch=patterns[2])

  recs_opf_svm = ax.bar(ind+(width*3)+space, data_mean_opf_svm, width,
                      color=colors[3],
                      alpha=opacity,
                      yerr=data_std_opf_svm,
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label=labels[3],
                      )#hatch=patterns[3])

  recs_overfeat_svm = ax.bar(ind+(width*4)+space, data_mean_overfeat_svm, width,
                      color=colors[4],
                      alpha=opacity,
                      yerr=data_std_overfeat_svm,
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label=labels[4],
                      )#hatch=patterns[4])

  recs_overfeat_opf = ax.bar(ind+(width*5)+space, data_mean_overfeat_opf, width,
                      color=colors[5],
                      alpha=opacity,
                      yerr=data_std_overfeat_opf,
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label=labels[5],
                      )#hatch=patterns[5])

  recs_cnn_yuv = ax.bar(ind+(width*6)+space, data_mean_cnn_yuv, width,
                      color=colors[6],
                      alpha=opacity,
                      yerr=data_std_cnn_yuv,
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label=labels[6],
                      )#hatch=patterns[6])

  recs_cnn_rgb = ax.bar(ind+(width*7)+space, data_mean_cnn_rgb, width,
                      color=colors[7],
                      alpha=opacity,
                      yerr=data_std_cnn_rgb,
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label=labels[7],
                      )#hatch=patterns[7])

  plt.xlabel(u'Classificador+Agrupador')
  plt.ylabel(u'Acurácia')
  plt.title(title_dataset)
  ax.set_xticks((0.07,0.22,0.37,0.52,0.67,0.82,0.97,1.12))
  ax.set_xticklabels( labels, rotation='vertical' )
  plt.ylim(0.0,1.0)
  ax.set_yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
  #plt.legend(loc='best')
  plt.tight_layout()
  fig.savefig(filename)

def plot_accuracy_best_gui001(df_kmeans_svm, df_kmeans_opf, df_opf_opf, df_opf_svm, df_overfeat,  cnn_yuv_accuracy, cnn_rgb_accuracy, title_dataset, filename):
  gf_kmeans_svm = df_kmeans_svm.groupby(['feature_type','descriptor_type','thumbnail_size'])
  gf_kmeans_opf = df_kmeans_opf.groupby(['feature_type','descriptor_type','thumbnail_size'])
  gf_opf_opf = df_opf_opf.groupby(['feature_type','descriptor_type','thumbnail_size'])
  gf_opf_svm = df_opf_svm.groupby(['feature_type','descriptor_type','thumbnail_size'])
  gf_overfeat_svm = df_overfeat.T[0]
  gf_overfeat_opf = df_overfeat.T[1]

  res_kmeans_svm = pd.DataFrame()
  res_kmeans_opf = pd.DataFrame()
  res_opf_opf = pd.DataFrame()
  res_opf_svm = pd.DataFrame()
  
  res_kmeans_svm['mean'] = gf_kmeans_svm['accuracy'].mean()
  res_kmeans_svm['std'] = gf_kmeans_svm['accuracy'].std()

  res_kmeans_opf['mean'] = gf_kmeans_opf['accuracy'].mean()
  res_kmeans_opf['std'] = gf_kmeans_opf['accuracy'].std()

  res_opf_opf['mean'] = gf_opf_opf['accuracy'].mean()
  res_opf_opf['std'] = gf_opf_opf['accuracy'].std()

  res_opf_svm['mean'] = gf_opf_svm['accuracy'].mean()
  res_opf_svm['std'] = gf_opf_svm['accuracy'].std()


  data_mean_kmeans_svm = res_kmeans_svm['mean']['MSER']['SIFT'][3] #0.587963
  data_std_kmeans_svm = res_kmeans_svm['std']['MSER']['SIFT'][3] #0.020849

  data_mean_kmeans_opf = res_kmeans_opf['mean']['Dense']['SIFT'][1] #0.499074
  data_std_kmeans_opf = res_kmeans_opf['std']['Dense']['SIFT'][1] #0.011226

  data_mean_opf_opf = res_opf_opf['mean']['Dense']['SURF'][2] #0.3296
  data_std_opf_opf = res_opf_opf['std']['Dense']['SURF'][2] #0.0189

  data_mean_opf_svm = res_opf_svm['mean']['Dense']['SURF'][3] #0.328704
  data_std_opf_svm = res_opf_svm['std']['Dense']['SURF'][3] #0.016973

  data_mean_overfeat_opf = gf_overfeat_opf['accuracy']
  data_std_overfeat_opf = 0.0

  data_mean_overfeat_svm = gf_overfeat_svm['accuracy']
  data_std_overfeat_svm = 0.0

  
  data_mean_cnn_yuv = cnn_yuv_accuracy
  data_std_cnn_yuv = 0.0

  data_mean_cnn_rgb = cnn_rgb_accuracy
  data_std_cnn_rgb = 0.0

  width = 0.15 
  opacity = 0.6
  space = 0.0

  fig = plt.figure()
  ax = fig.add_subplot(111)

  N=1
  ind = numpy.arange(N)

  #colors = ['b','r', 'g', 'y', 'c', 'm', 'y', 'k', 'w']
  colors = ['b','b','b','b','b','b','b','b','b']
  patterns = ['-', '*', 'o', '\\', '+', '//', 'O', '.']
  labels = ['OVERFEAT+SVM','OVERFEAT+OPF','CNN RGB','CNN YUV','Kmeans+SVM \n (MSER+SIFT+512)','Kmeans+OPF-S \n (Denso+SIFT+128)','OPF-U+OPF-S \n (Denso+SURF+256)','OPF-U+SVM \n (Denso+SURF+512)']
  

  recs_overfeat_svm = ax.bar(ind, data_mean_overfeat_svm, width,
                      color=colors[0],
                      alpha=opacity,
                      yerr=data_std_overfeat_svm,
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label=labels[0],
                      )#hatch=patterns[0])

  recs_overfeat_opf = ax.bar(ind+(width*1)+space, data_mean_overfeat_opf, width,
                      color=colors[1],
                      alpha=opacity,
                      yerr=data_std_overfeat_opf,
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label=labels[1],
                      )#hatch=patterns[1])

  recs_cnn_rgb = ax.bar(ind+(width*2)+space, data_mean_cnn_rgb, width,
                      color=colors[2],
                      alpha=opacity,
                      yerr=data_std_cnn_rgb,
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label=labels[2],
                      )#hatch=patterns[2])

  recs_cnn_yuv = ax.bar(ind+(width*3)+space, data_mean_cnn_yuv, width,
                      color=colors[3],
                      alpha=opacity,
                      yerr=data_std_cnn_yuv,
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label=labels[3],
                      )#hatch=patterns[3])
  
  recs_kmeans_svm = ax.bar(ind+(width*4)+space, data_mean_kmeans_svm, width,
                      color=colors[4],
                      alpha=opacity,
                      yerr=data_std_kmeans_svm,
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label=labels[4],
                      )#hatch=patterns[4])

  recs_kmeans_opf = ax.bar(ind+(width*5)+space, data_mean_kmeans_opf, width,
                      color=colors[5],
                      alpha=opacity,
                      yerr=data_std_kmeans_opf,
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label=labels[5],
                      )#hatch=patterns[5])

  recs_opf_opf = ax.bar(ind+(width*6)+space, data_mean_opf_opf, width,
                      color=colors[6],
                      alpha=opacity,
                      yerr=data_std_opf_opf,
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label=labels[6],
                      )#hatch=patterns[6])

  recs_opf_svm = ax.bar(ind+(width*7)+space, data_mean_opf_svm, width,
                      color=colors[7],
                      alpha=opacity,
                      yerr=data_std_opf_svm,
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label=labels[7],
                      )#hatch=patterns[7])

  

  plt.xlabel(u'Classificador+Agrupador')
  plt.ylabel(u'Acurácia')
  plt.title(title_dataset)
  ax.set_xticks((0.07,0.22,0.37,0.52,0.67,0.82,0.97,1.12))
  ax.set_xticklabels( labels, rotation='vertical' )
  plt.ylim(0.0,1.0)
  ax.set_yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
  #plt.legend(loc='best')
  plt.tight_layout()
  fig.savefig(filename)

def plot_accuracy_best_caltech101(df_kmeans_svm, df_kmeans_opf, df_opf_opf, df_opf_svm, df_overfeat,  cnn_yuv_accuracy, cnn_rgb_accuracy, title_dataset, filename):
  gf_kmeans_svm = df_kmeans_svm.groupby(['feature_type','descriptor_type','thumbnail_size'])
  gf_kmeans_opf = df_kmeans_opf.groupby(['feature_type','descriptor_type','thumbnail_size'])
  gf_opf_opf = df_opf_opf.groupby(['feature_type','descriptor_type','thumbnail_size'])
  gf_opf_svm = df_opf_svm.groupby(['feature_type','descriptor_type','thumbnail_size'])
  gf_overfeat_svm = df_overfeat.T[0]
  gf_overfeat_opf = df_overfeat.T[1]

  res_kmeans_svm = pd.DataFrame()
  res_kmeans_opf = pd.DataFrame()
  res_opf_opf = pd.DataFrame()
  res_opf_svm = pd.DataFrame()
  
  res_kmeans_svm['mean'] = gf_kmeans_svm['accuracy'].mean()
  res_kmeans_svm['std'] = gf_kmeans_svm['accuracy'].std()

  res_kmeans_opf['mean'] = gf_kmeans_opf['accuracy'].mean()
  res_kmeans_opf['std'] = gf_kmeans_opf['accuracy'].std()

  res_opf_opf['mean'] = gf_opf_opf['accuracy'].mean()
  res_opf_opf['std'] = gf_opf_opf['accuracy'].std()

  res_opf_svm['mean'] = gf_opf_svm['accuracy'].mean()
  res_opf_svm['std'] = gf_opf_svm['accuracy'].std()


  data_mean_kmeans_svm = res_kmeans_svm['mean']['MSER']['SIFT'][2] #0.467667
  data_std_kmeans_svm = res_kmeans_svm['std']['MSER']['SIFT'][2] #0.020034

  data_mean_kmeans_opf = res_kmeans_opf['mean']['Dense']['SIFT'][3] #0.358353
  data_std_kmeans_opf = res_kmeans_opf['std']['Dense']['SIFT'][3] #0.010089

  data_mean_opf_opf = res_opf_opf['mean']['Dense']['SIFT'][3] #0.251732
  data_std_opf_opf = res_opf_opf['std']['Dense']['SIFT'][3] #0.062174

  data_mean_opf_svm = res_opf_svm['mean']['Dense']['SIFT'][3] #0.269438
  data_std_opf_svm = res_opf_svm['std']['Dense']['SIFT'][3] #0.005812

  data_mean_overfeat_opf = gf_overfeat_opf['accuracy']
  data_std_overfeat_opf = 0.0

  data_mean_overfeat_svm = gf_overfeat_svm['accuracy']
  data_std_overfeat_svm = 0.0

  
  data_mean_cnn_yuv = cnn_yuv_accuracy
  data_std_cnn_yuv = 0.0

  data_mean_cnn_rgb = cnn_rgb_accuracy
  data_std_cnn_rgb = 0.0

  width = 0.15 
  opacity = 0.6
  space = 0.0

  fig = plt.figure()
  ax = fig.add_subplot(111)

  N=1
  ind = numpy.arange(N)

  #colors = ['b','r', 'g', 'y', 'c', 'm', 'y', 'k', 'w']
  colors = ['b','b','b','b','b','b','b','b','b']
  patterns = ['-', '*', 'o', '\\', '+', '//', 'O', '.']
  labels = ['OVERFEAT+SVM','OVERFEAT+OPF','CNN RGB','CNN YUV','Kmeans+SVM \n (MSER+SIFT+256)','Kmeans+OPF-S \n (Denso+SIFT+512)','OPF-U+OPF-S \n (Denso+SIFT+512)','OPF-U+SVM \n (Denso+SIFT+512)']
  

  recs_overfeat_svm = ax.bar(ind, data_mean_overfeat_svm, width,
                      color=colors[0],
                      alpha=opacity,
                      yerr=data_std_overfeat_svm,
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label=labels[0],
                      )#hatch=patterns[0])

  recs_overfeat_opf = ax.bar(ind+(width*1)+space, data_mean_overfeat_opf, width,
                      color=colors[1],
                      alpha=opacity,
                      yerr=data_std_overfeat_opf,
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label=labels[1],
                      )#hatch=patterns[1])

  recs_cnn_rgb = ax.bar(ind+(width*2)+space, data_mean_cnn_rgb, width,
                      color=colors[2],
                      alpha=opacity,
                      yerr=data_std_cnn_rgb,
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label=labels[2],
                      )#hatch=patterns[2])

  recs_cnn_yuv = ax.bar(ind+(width*3)+space, data_mean_cnn_yuv, width,
                      color=colors[3],
                      alpha=opacity,
                      yerr=data_std_cnn_yuv,
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label=labels[3],
                      )#hatch=patterns[3])
  
  recs_kmeans_svm = ax.bar(ind+(width*4)+space, data_mean_kmeans_svm, width,
                      color=colors[4],
                      alpha=opacity,
                      yerr=data_std_kmeans_svm,
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label=labels[4],
                      )#hatch=patterns[4])

  recs_kmeans_opf = ax.bar(ind+(width*5)+space, data_mean_kmeans_opf, width,
                      color=colors[5],
                      alpha=opacity,
                      yerr=data_std_kmeans_opf,
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label=labels[5],
                      )#hatch=patterns[5])

  recs_opf_opf = ax.bar(ind+(width*6)+space, data_mean_opf_opf, width,
                      color=colors[6],
                      alpha=opacity,
                      yerr=data_std_opf_opf,
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label=labels[6],
                      )#hatch=patterns[6])

  recs_opf_svm = ax.bar(ind+(width*7)+space, data_mean_opf_svm, width,
                      color=colors[7],
                      alpha=opacity,
                      yerr=data_std_opf_svm,
                      error_kw=dict(elinewidth=2,ecolor='red'),
                      label=labels[7],
                      )#hatch=patterns[7])

  

  plt.xlabel(u'Classificador+Agrupador')
  plt.ylabel(u'Acurácia')
  plt.title(title_dataset)
  ax.set_xticks((0.07,0.22,0.37,0.52,0.67,0.82,0.97,1.12))
  ax.set_xticklabels( labels, rotation='vertical' )
  plt.ylim(0.0,1.0)
  ax.set_yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
  #plt.legend(loc='best')
  plt.tight_layout()
  fig.savefig(filename)

#
def plot_base_criada_n_categorias_x_acuracia(filename):
  #  5 - Kmeans+SVM - 0.87
  #  5 - Kmeans+OPF - 0.88
  #  5 - OPF+OPF    - 0.68
  #  5 - OPF+SVM    - 0.66
  #  5 - OVER+SVM   - 1.00
  #  5 - OVER+OPF   - 0.98
  #  5 - CNN YUV    - 0.90

  # 10 - Kmeans+SVM - 0.85
  # 10 - Kmeans+OPF - 0.77
  # 10 - OPF+OPF    - 0.56
  # 10 - OPF+SVM    - 0.56
  # 10 - OVER+SVM   - 0.97
  # 10 - OVER+OPF   - 0.93
  # 10 - CNN YUV    - 0.88

  # 15 - Kmeans+SVM - 0.71
  # 15 - Kmeans+OPF - 0.62
  # 15 - OPF+OPF    - 0.44
  # 15 - OPF+SVM    - 0.41
  # 15 - OVER+SVM   - 0.93
  # 15 - OVER+OPF   - 0.88
  # 15 - CNN YUV    - 0.74

  # 36 - Kmeans+SVM - 0.59
  # 36 - Kmeans+OPF - 0.50
  # 36 - OPF+OPF    - 0.33
  # 36 - OPF+SVM    - 0.33
  # 36 - OVER+SVM   - 0.90
  # 36 - OVER+OPF   - 0.85
  # 36 - CNN YUV    - 0.69

  x_axis = [5,10,15,36]
  mean_kmeans_svm = [0.87,0.85,0.71,0.59]
  mean_kmeans_opf = [0.88,0.77,0.62,0.50]
  mean_opf_opf = [0.68,0.56,0.44,0.33]
  mean_opf_svm = [0.66,0.56,0.41,0.33]
  mean_overfeat_opf = [0.98,0.93,0.88,0.85]
  mean_overfeat_svm = [1.00,0.97,0.93,0.90]
  mean_cnn_yuv = [0.90,0.88,0.74,0.69]

  fig, ax = plt.subplots()
  ax.plot(x_axis, mean_kmeans_svm, 'o-',label='Kmeans + SVM')
  ax.plot(x_axis, mean_kmeans_opf, 'o-',label='Kmeans + OPF-S')
  ax.plot(x_axis, mean_opf_opf, 'o-',label='OPF-U + OPF-S')
  ax.plot(x_axis, mean_opf_svm, 'o-',label='OPF-U + SVM')
  ax.plot(x_axis, mean_overfeat_opf, 'o-',label='OVERFEAT + OPF-S')
  ax.plot(x_axis, mean_overfeat_svm, 'o-',label='OVERFEAT + SVM')
  ax.plot(x_axis, mean_cnn_yuv, 'o-',label='CNN YUV')
  plt.xlabel(u'Número de categorias')
  plt.ylabel(u'Acurácia')
  plt.ylim(0.0,1.0)
  plt.xlim(5.0,36.0)
  #plt.legend(bbox_to_anchor=(0.42, 0.42), )
  plt.legend(loc='lower left', prop={'size':10})
  ax.set_xticks((5.0,10.0,15.0,36.0))
  ax.set_xticklabels(('5','10','15','36'))
  fig.savefig(filename)

def plot_base_criada_n_categorias_x_f1(filename):
  #  5 - Kmeans+SVM - 0.87
  #  5 - Kmeans+OPF - 0.88
  #  5 - OPF+OPF    - 0.69
  #  5 - OPF+SVM    - 0.67
  #  5 - OVER+SVM   - 1.00
  #  5 - OVER+OPF   - 0.98
  #  5 - CNN YUV    - 0.90

  # 10 - Kmeans+SVM - 0.85
  # 10 - Kmeans+OPF - 0.77
  # 10 - OPF+OPF    - 0.57
  # 10 - OPF+SVM    - 0.52
  # 10 - OVER+SVM   - 0.97
  # 10 - OVER+OPF   - 0.93
  # 10 - CNN YUV    - 0.88

  # 15 - Kmeans+SVM - 0.71
  # 15 - Kmeans+OPF - 0.61
  # 15 - OPF+OPF    - 0.44
  # 15 - OPF+SVM    - 0.37
  # 15 - OVER+SVM   - 0.93
  # 15 - OVER+OPF   - 0.88
  # 15 - CNN YUV    - 0.74

  # 36 - Kmeans+SVM - 0.57
  # 36 - Kmeans+OPF - 0.50
  # 36 - OPF+OPF    - 0.33
  # 36 - OPF+SVM    - 0.27
  # 36 - OVER+SVM   - 0.90
  # 36 - OVER+OPF   - 0.85
  # 36 - CNN YUV    - 0.69

  x_axis = [5,10,15,36]
  mean_kmeans_svm = [0.87,0.85,0.71,0.59]
  mean_kmeans_opf = [0.88,0.77,0.61,0.50]
  mean_opf_opf = [0.69,0.57,0.44,0.33]
  mean_opf_svm = [0.67,0.52,0.37,0.27]
  mean_overfeat_opf = [0.98,0.93,0.87,0.83]
  mean_overfeat_svm = [1.00,0.97,0.92,0.89]
  mean_cnn_yuv = [0.90,0.88,0.74,0.67]

  fig, ax = plt.subplots()
  ax.plot(x_axis, mean_kmeans_svm, 'o-',label='Kmeans + SVM')
  ax.plot(x_axis, mean_kmeans_opf, 'o-',label='Kmeans + OPF-S')
  ax.plot(x_axis, mean_opf_opf, 'o-',label='OPF-U + OPF-S')
  ax.plot(x_axis, mean_opf_svm, 'o-',label='OPF-U + SVM')
  ax.plot(x_axis, mean_overfeat_opf, 'o-',label='OVERFEAT + OPF-S')
  ax.plot(x_axis, mean_overfeat_svm, 'o-',label='OVERFEAT + SVM')
  ax.plot(x_axis, mean_cnn_yuv, 'o-',label='CNN YUV')
  plt.xlabel(u'Número de categorias')
  plt.ylabel(u'F1')
  plt.ylim(0.0,1.0)
  plt.xlim(5.0,36.0)
  #plt.legend(bbox_to_anchor=(0.42, 0.42), )
  plt.legend(loc='lower left', prop={'size':10})
  ax.set_xticks((5.0,10.0,15.0,36.0))
  ax.set_xticklabels(('5','10','15','36'))
  fig.savefig(filename)

def plot_time_consult_gui001(filename):
  time_kmeans_svm = 351
  time_kmeans_opf = 343
  time_opf_opf = 520
  time_opf_svm = 526
  time_cnn_yuv = 20
  time_cnn_rgb = 20
  time_overfeat_opf = 4250
  time_overfeat_svm = 4220


  labels = ['CNN YUV','CNN RGB', 'Kmeans+SVM','Kmeans+OPF-S','OPF-U+OPF-S','OPF-U+SVM','OVERFEAT+SVM','OVERFEAT+OPF']
  title = u"Base Criada (36 cat.) - Tempo de consulta."
  width = 0.15 
  opacity = 0.6
  space = 0.0
  N=1
  ind = numpy.arange(N)
  fig, ax = plt.subplots()
  
  recs_cnn_yuv = ax.bar(ind, time_cnn_yuv, width,
                      color='b',
                      label=labels[0])

  recs_cnn_rgb = ax.bar(ind+(width*1)+space, time_cnn_rgb, width,
                      color='b',
                      label=labels[0])

  recs_kmeans_svm = ax.bar(ind+(width*2)+space, time_kmeans_svm, width,
                      color='b',
                      label=labels[0])
  
  recs_kmeans_opf = ax.bar(ind+(width*3)+space, time_kmeans_opf, width,
                      color='b',
                      label=labels[0])
  
  recs_opf_opf = ax.bar(ind+(width*4)+space, time_opf_opf, width,
                      color='b',
                      label=labels[0])

  recs_opf_svm = ax.bar(ind+(width*5)+space, time_opf_svm, width,
                      color='b',
                      label=labels[0])

  recs_overfeat_opf = ax.bar(ind+(width*6)+space, time_overfeat_opf, width,
                      color='b',
                      label=labels[0])

  recs_overfeat_svm = ax.bar(ind+(width*7)+space, time_overfeat_svm, width,
                      color='b',
                      label=labels[0])

  def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, ('%.0f'%float(height)).replace('.',','),
                ha='center', va='bottom')


  plt.xlabel(u'Agrupador + Classificador')
  plt.ylabel(u'Tempo de consulta (em ms)')
  plt.title(title)
  ax.set_xticks((0.07,0.22,0.37,0.52,0.67,0.82,0.97,1.12))
  ax.set_xticklabels( labels, rotation='vertical' )
  #plt.grid(True,which="both", axis="y")
  plt.ylim(0.0,5000)
  #ax.set_yscale('log')
  #plt.grid(True,which="majorminor",ls="-", color='0.65')
  #plt.ylim(0.0,1.0)
  #ax.set_yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
  #plt.legend(loc='best')
  autolabel(recs_cnn_yuv)
  autolabel(recs_cnn_rgb)
  autolabel(recs_kmeans_svm)
  autolabel(recs_kmeans_opf)
  autolabel(recs_opf_opf)
  autolabel(recs_opf_svm)
  autolabel(recs_overfeat_opf)
  autolabel(recs_overfeat_svm)
  plt.tight_layout()
  fig.savefig(filename)

def plot_time_total_gui001(filename):
  time_kmeans_svm = 167.992293 #(MSER+SIFT - 512 - 0.587)
  time_kmeans_opf = 158.981471 #(Dense + SIFT - 128 - 0.499)
  time_opf_opf = 576.398405    #(Dense + SURF - 256 - 0.329)
  time_opf_svm = 583.201677    #(Dense + SURF - 512 - 0.328)
  time_cnn_yuv = 6.4968e+04    #(0.691)
  time_cnn_rgb = 2.3934e+05    #(0.716)
  time_overfeat_opf = 52765.74642 #(0.850)
  time_overfeat_svm = 52023.197393 #(0.905)

  time_kmeans_svm = float(time_kmeans_svm/60.)
  time_kmeans_opf = float(time_kmeans_opf/60.)
  time_opf_opf = float(time_opf_opf/60.)
  time_opf_svm = float(time_opf_svm/60.)
  time_cnn_yuv = float(time_cnn_yuv/60.)
  time_cnn_rgb = float(time_cnn_rgb/60.)
  time_overfeat_opf = float(time_overfeat_opf/60.)
  time_overfeat_svm = float(time_overfeat_svm/60.)

  # print("---- TEMPO TOTAL GUI001 -----")
  # print("KMeans+SVM: %.2f" % time_kmeans_svm)
  # print("KMeans+OPF: %.2f" % time_kmeans_opf)
  # print("OPF+OPF: %.2f" % time_opf_opf)
  # print("OPF+SVM: %.2f" % time_opf_svm)
  # print("CNN YUV: %.2f" % time_cnn_yuv)
  # print("CNN RGB: %.2f" % time_cnn_rgb)
  # print("OVERFEAT+OPF: %.2f" % time_overfeat_opf)
  # print("OVERFEAT+SVM: %.2f" % time_overfeat_svm)
  # print("-----------------------------")

  import math
  time_kmeans_svm = math.log(time_kmeans_svm)
  time_kmeans_opf = math.log(time_kmeans_opf)
  time_opf_opf = math.log(time_opf_opf)
  time_opf_svm = math.log(time_opf_svm)
  time_cnn_yuv = math.log(time_cnn_yuv)
  time_cnn_rgb = math.log(time_cnn_rgb)
  time_overfeat_opf = math.log(time_overfeat_opf)
  time_overfeat_svm = math.log(time_overfeat_svm)

  # print("---- TEMPO TOTAL GUI001 (LOG) -----")
  # print("KMeans+SVM: %.2f" % time_kmeans_svm)
  # print("KMeans+OPF: %.2f" % time_kmeans_opf)
  # print("OPF+OPF: %.2f" % time_opf_opf)
  # print("OPF+SVM: %.2f" % time_opf_svm)
  # print("CNN YUV: %.2f" % time_cnn_yuv)
  # print("CNN RGB: %.2f" % time_cnn_rgb)
  # print("OVERFEAT+OPF: %.2f" % time_overfeat_opf)
  # print("OVERFEAT+SVM: %.2f" % time_overfeat_svm)
  # print("-----------------------------")

  labels = ['Kmeans+SVM','Kmeans+OPF-S','OPF-U+OPF-S','OPF-U+SVM','CNN YUV','CNN RGB','OVERFEAT+SVM','OVERFEAT+OPF']
  title = u"Base Criada (36 cat.) - Tempo total de processamento"
  width = 0.15 
  opacity = 0.6
  space = 0.0
  N=1
  ind = numpy.arange(N)
  fig, ax = plt.subplots()
  
  
  recs_kmeans_svm = ax.bar(ind, time_kmeans_svm, width,
                      color='b',
                      label=labels[0],
                      log=True)
  
  recs_kmeans_opf = ax.bar(ind+(width*1)+space, time_kmeans_opf, width,
                      color='b',
                      label=labels[0],
                      log=True)
  
  recs_opf_opf = ax.bar(ind+(width*2)+space, time_opf_opf, width,
                      color='b',
                      label=labels[0],
                      log=True)

  recs_opf_svm = ax.bar(ind+(width*3)+space, time_opf_svm, width,
                      color='b',
                      label=labels[0],
                      log=True)

  recs_cnn_yuv = ax.bar(ind+(width*4)+space, time_cnn_yuv, width,
                      color='b',
                      label=labels[0],
                      log=True)

  recs_cnn_rgb = ax.bar(ind+(width*5)+space, time_cnn_rgb, width,
                      color='b',
                      label=labels[0],
                      log=True)

  recs_overfeat_opf = ax.bar(ind+(width*6)+space, time_overfeat_opf, width,
                      color='b',
                      label=labels[0],
                      log=True)

  recs_overfeat_svm = ax.bar(ind+(width*7)+space, time_overfeat_svm, width,
                      color='b',
                      label=labels[0],
                      log=True)

  def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, ('%.2f'%float(height)).replace('.',','),
                ha='center', va='bottom')


  plt.xlabel(u'Agrupador + Classificador')
  plt.ylabel(u'Tempo total em minutos (escala logarítmica)')
  plt.title(title)
  ax.set_xticks((0.07,0.22,0.37,0.52,0.67,0.82,0.97,1.12))
  ax.set_xticklabels( labels, rotation='vertical' )
  plt.grid(True,which="both", axis="y")
  plt.ylim(0.0,12)
  #ax.set_yscale('log')
  #plt.grid(True,which="majorminor",ls="-", color='0.65')
  #plt.ylim(0.0,1.0)
  #ax.set_yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
  #plt.legend(loc='best')
  autolabel(recs_kmeans_svm)
  autolabel(recs_kmeans_opf)
  autolabel(recs_opf_opf)
  autolabel(recs_opf_svm)
  autolabel(recs_cnn_yuv)
  autolabel(recs_cnn_rgb)
  autolabel(recs_overfeat_opf)
  autolabel(recs_overfeat_svm)
  plt.tight_layout()
  fig.savefig(filename)

def plot_time_total_caltech101(filename):
  time_kmeans_svm = 122.891485 #(MSER+SIFT - 256 - 0.467)
  time_kmeans_opf = 405.522284 #(Dense+SIFT - 512 - 0.358)
  time_opf_opf = 5435.363571 #(Dense+SIFT - 512 - 0.251)
  time_opf_svm = 5302.470458 #(Dense+SIFT - 512 - 0.251)
  time_cnn_yuv = 1.8604e+05 #(0.546)
  time_cnn_rgb = 6.1241e+05 #(0.487)
  time_overfeat_opf = 44789.931292 #(0.695)
  time_overfeat_svm = 44291.874468 #(0.855)

  time_kmeans_svm = float(time_kmeans_svm/60.)
  time_kmeans_opf = float(time_kmeans_opf/60.)
  time_opf_opf = float(time_opf_opf/60.)
  time_opf_svm = float(time_opf_svm/60.)
  time_cnn_yuv = float(time_cnn_yuv/60.)
  time_cnn_rgb = float(time_cnn_rgb/60.)
  time_overfeat_opf = float(time_overfeat_opf/60.)
  time_overfeat_svm = float(time_overfeat_svm/60.)

  # print("---- TEMPO TOTAL CALTECH101 -----")
  # print("KMeans+SVM: %.2f" % time_kmeans_svm)
  # print("KMeans+OPF: %.2f" % time_kmeans_opf)
  # print("OPF+OPF: %.2f" % time_opf_opf)
  # print("OPF+SVM: %.2f" % time_opf_svm)
  # print("CNN YUV: %.2f" % time_cnn_yuv)
  # print("CNN RGB: %.2f" % time_cnn_rgb)
  # print("OVERFEAT+OPF: %.2f" % time_overfeat_opf)
  # print("OVERFEAT+SVM: %.2f" % time_overfeat_svm)
  # print("-----------------------------")

  import math
  time_kmeans_svm = math.log(time_kmeans_svm)
  time_kmeans_opf = math.log(time_kmeans_opf)
  time_opf_opf = math.log(time_opf_opf)
  time_opf_svm = math.log(time_opf_svm)
  time_cnn_yuv = math.log(time_cnn_yuv)
  time_cnn_rgb = math.log(time_cnn_rgb)
  time_overfeat_opf = math.log(time_overfeat_opf)
  time_overfeat_svm = math.log(time_overfeat_svm)

  # print("---- TEMPO TOTAL CALTECH101 (LOG) -----")
  # print("KMeans+SVM: %.2f" % time_kmeans_svm)
  # print("KMeans+OPF: %.2f" % time_kmeans_opf)
  # print("OPF+OPF: %.2f" % time_opf_opf)
  # print("OPF+SVM: %.2f" % time_opf_svm)
  # print("CNN YUV: %.2f" % time_cnn_yuv)
  # print("CNN RGB: %.2f" % time_cnn_rgb)
  # print("OVERFEAT+OPF: %.2f" % time_overfeat_opf)
  # print("OVERFEAT+SVM: %.2f" % time_overfeat_svm)
  # print("-----------------------------")


  labels = ['Kmeans+SVM','Kmeans+OPF-S','OPF-U+OPF-S','OPF-U+SVM','CNN YUV','CNN RGB','OVERFEAT+SVM','OVERFEAT+OPF']
  title = u"Caltech101 - Tempo total de processamento"
  width = 0.15 
  opacity = 0.6
  space = 0.0
  N=1
  ind = numpy.arange(N)
  fig, ax = plt.subplots()
  
  recs_kmeans_svm = ax.bar(ind, time_kmeans_svm, width,
                      color='b',
                      label=labels[0],
                      log=True)
  
  recs_kmeans_opf = ax.bar(ind+(width*1)+space, time_kmeans_opf, width,
                      color='b',
                      label=labels[0],
                      log=True)
  
  recs_opf_opf = ax.bar(ind+(width*2)+space, time_opf_opf, width,
                      color='b',
                      label=labels[0],
                      log=True)

  recs_opf_svm = ax.bar(ind+(width*3)+space, time_opf_svm, width,
                      color='b',
                      label=labels[0],
                      log=True)

  recs_cnn_yuv = ax.bar(ind+(width*4)+space, time_cnn_yuv, width,
                      color='b',
                      label=labels[0],
                      log=True)

  recs_cnn_rgb = ax.bar(ind+(width*5)+space, time_cnn_rgb, width,
                      color='b',
                      label=labels[0],
                      log=True)

  recs_overfeat_opf = ax.bar(ind+(width*6)+space, time_overfeat_opf, width,
                      color='b',
                      label=labels[0],
                      log=True)

  recs_overfeat_svm = ax.bar(ind+(width*7)+space, time_overfeat_svm, width,
                      color='b',
                      label=labels[0],
                      log=True)

  def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, ('%.2f'%float(height)).replace('.',','),
                ha='center', va='bottom')

  plt.xlabel(u'Agrupador + Classificador')
  plt.ylabel(u'Tempo total em minutos (escala logarítmica)')
  plt.title(title)
  ax.set_xticks((0.07,0.22,0.37,0.52,0.67,0.82,0.97,1.12))
  ax.set_xticklabels( labels, rotation='vertical' )
  plt.grid(True,which="both", axis="y")
  plt.ylim(0.0,12)
  #ax.set_yscale('log')
  #ax.set_yticks([0.0,1000.0,3000.0,6000.0,10000.0,15000.0, 1000000])
  #plt.legend(loc='best')
  autolabel(recs_kmeans_svm)
  autolabel(recs_kmeans_opf)
  autolabel(recs_opf_opf)
  autolabel(recs_opf_svm)
  autolabel(recs_cnn_yuv)
  autolabel(recs_cnn_rgb)
  autolabel(recs_overfeat_opf)
  autolabel(recs_overfeat_svm)
  plt.tight_layout()
  fig.savefig(filename)


def plot_base_criada_n_categorias_x_tempo_2(filename):

  x_axis = [5, 10, 15, 36]
  time_kmeans_svm   = [66.802133,90.236762,92.185353,167.99]
  time_kmeans_opf   = [84.246373,93.813620,158.98,192.313446]
  time_opf_opf      = [362.627685,723.886606,2013.758292,2626.266887]
  time_opf_svm      = [381.186390,709.125114,1959.763897,2469.434085]
  time_cnn_yuv      = [8366.0,52973.0,34433.0,64968.0]
  time_overfeat_svm = [8635.701758,15581.375471,23299.505833,52023.197393]
  time_overfeat_opf = [8657.854621,15650.895837,23453.045399,54535.462961]

  time_kmeans_svm = (time_kmeans_svm - numpy.min(time_kmeans_svm))/(numpy.max(time_kmeans_svm) - numpy.min(time_kmeans_svm))
  time_kmeans_opf = (time_kmeans_opf - numpy.min(time_kmeans_opf))/(numpy.max(time_kmeans_opf) - numpy.min(time_kmeans_opf))
  time_opf_opf = (time_opf_opf - numpy.min(time_opf_opf))/(numpy.max(time_opf_opf) - numpy.min(time_opf_opf))
  time_opf_svm = (time_opf_svm - numpy.min(time_opf_svm))/(numpy.max(time_opf_svm) - numpy.min(time_opf_svm))
  #time_cnn_yuv = (time_cnn_yuv - numpy.min(time_cnn_yuv))/(numpy.max(time_cnn_yuv) - numpy.min(time_cnn_yuv))
  time_overfeat_svm = (time_overfeat_svm - numpy.min(time_overfeat_svm))/(numpy.max(time_overfeat_svm) - numpy.min(time_overfeat_svm))
  time_overfeat_opf = (time_overfeat_opf - numpy.min(time_overfeat_opf))/(numpy.max(time_overfeat_opf) - numpy.min(time_overfeat_opf))
  

  # import math
  # time_kmeans_svm = numpy.log(time_kmeans_svm)
  # time_kmeans_opf = numpy.log(time_kmeans_opf)
  # time_opf_opf = numpy.log(time_opf_opf)
  # time_opf_svm = numpy.log(time_opf_svm)
  # time_cnn_yuv = numpy.log(time_cnn_yuv)
  # time_overfeat_svm = numpy.log(time_overfeat_svm)
  # time_overfeat_opf = numpy.log(time_overfeat_opf)

  fig = plt.figure(1)
  ax = plt.subplot(321)
  ax.plot(x_axis, time_kmeans_svm, 'o-',label='Kmeans + SVM')
  ax.set_xticks((5.0,10.0,15.0,36.0))
  ax.set_xticklabels(('5','10','15','36'))
  plt.legend(loc='lower right', prop={'size':9})
  # plt.xlabel(u'Categorias')
  # plt.ylabel(u'Tempo')
  
  # plt.xlim(5.0,36.0)
  
  ax = plt.subplot(322)
  ax.plot(x_axis, time_kmeans_opf, 'o-',label='Kmeans + OPF-S')
  ax.set_xticks((5.0,10.0,15.0,36.0))
  ax.set_xticklabels(('5','10','15','36'))
  plt.legend(loc='lower right', prop={'size':9})
  # plt.xlabel(u'Categorias')
  # plt.ylabel(u'Tempo')
  # plt.xlim(5.0,36.0)
  # ax.set_xticks((5.0,10.0,15.0,36.0))
  # ax.set_xticklabels(('5','10','15','36'))

  ax = plt.subplot(323)
  ax.plot(x_axis, time_opf_opf, 'o-',label='OPF-U + OPF-S')
  ax.set_xticks((5.0,10.0,15.0,36.0))
  ax.set_xticklabels(('5','10','15','36'))
  plt.legend(loc='lower right', prop={'size':9})
  # plt.xlabel(u'Categorias')
  # plt.ylabel(u'Tempo')
  # plt.xlim(5.0,36.0)

  ax = plt.subplot(324)
  ax.plot(x_axis, time_opf_svm, 'o-',label='OPF-U + SVM')
  ax.set_xticks((5.0,10.0,15.0,36.0))
  ax.set_xticklabels(('5','10','15','36'))
  plt.legend(loc='lower right', prop={'size':9})
  # plt.xlabel(u'Categorias')
  # plt.ylabel(u'Tempo')
  # plt.xlim(5.0,36.0)

  ax = plt.subplot(325)
  ax.plot(x_axis, time_overfeat_opf, 'o-',label='OVERFEAT + OPF-S')
  ax.set_xticks((5.0,10.0,15.0,36.0))
  ax.set_xticklabels(('5','10','15','36'))
  plt.legend(loc='lower right', prop={'size':9})
  # plt.xlabel(u'Categorias')
  # plt.ylabel(u'Tempo')
  # plt.xlim(5.0,36.0)

  ax = plt.subplot(326)
  ax.plot(x_axis, time_overfeat_svm, 'o-',label='OVERFEAT + SVM')
  ax.set_xticks((5.0,10.0,15.0,36.0))
  ax.set_xticklabels(('5','10','15','36'))
  plt.legend(loc='lower right', prop={'size':9})
  # plt.xlabel(u'Categorias')
  # plt.ylabel(u'Tempo')
  # plt.xlim(5.0,36.0)

  #ax.plot(x_axis, time_cnn_yuv, 'o-',label='CNN YUV')
  #plt.xlabel(u'Número de categorias')
  #plt.ylabel(u'Tempo total normalizado')
  #plt.ylim(0.0,1.0)
  
  #plt.legend(bbox_to_anchor=(0.42, 0.42), )
  
  fig.savefig(filename)


def plot_base_criada_n_categorias_x_tempo(filename):

  x_axis = [5, 10, 15, 36]
  time_kmeans_svm   = [66.802133,90.236762,92.185353,167.99]
  time_kmeans_opf   = [84.246373,93.813620,158.98,192.313446]
  time_opf_opf      = [362.627685,723.886606,2013.758292,2626.266887]
  time_opf_svm      = [381.186390,709.125114,1959.763897,2469.434085]
  time_cnn_yuv      = [3379.0,14527.0,40382.0,64561.0]
  time_overfeat_svm = [8635.701758,15581.375471,23299.505833,52023.197393]
  time_overfeat_opf = [8657.854621,15650.895837,23453.045399,54535.462961]

  time_kmeans_svm = (time_kmeans_svm - numpy.min(time_kmeans_svm))/(numpy.max(time_kmeans_svm) - numpy.min(time_kmeans_svm))
  time_kmeans_opf = (time_kmeans_opf - numpy.min(time_kmeans_opf))/(numpy.max(time_kmeans_opf) - numpy.min(time_kmeans_opf))
  time_opf_opf = (time_opf_opf - numpy.min(time_opf_opf))/(numpy.max(time_opf_opf) - numpy.min(time_opf_opf))
  time_opf_svm = (time_opf_svm - numpy.min(time_opf_svm))/(numpy.max(time_opf_svm) - numpy.min(time_opf_svm))
  time_cnn_yuv = (time_cnn_yuv - numpy.min(time_cnn_yuv))/(numpy.max(time_cnn_yuv) - numpy.min(time_cnn_yuv))
  time_overfeat_svm = (time_overfeat_svm - numpy.min(time_overfeat_svm))/(numpy.max(time_overfeat_svm) - numpy.min(time_overfeat_svm))
  time_overfeat_opf = (time_overfeat_opf - numpy.min(time_overfeat_opf))/(numpy.max(time_overfeat_opf) - numpy.min(time_overfeat_opf))
  

  # import math
  # time_kmeans_svm = numpy.log(time_kmeans_svm)
  # time_kmeans_opf = numpy.log(time_kmeans_opf)
  # time_opf_opf = numpy.log(time_opf_opf)
  # time_opf_svm = numpy.log(time_opf_svm)
  # time_cnn_yuv = numpy.log(time_cnn_yuv)
  # time_overfeat_svm = numpy.log(time_overfeat_svm)
  # time_overfeat_opf = numpy.log(time_overfeat_opf)


  fig, ax = plt.subplots()
  ax.plot(x_axis, time_kmeans_svm, 'o-',label='Kmeans + SVM')
  ax.plot(x_axis, time_kmeans_opf, 'o-',label='Kmeans + OPF-S')
  ax.plot(x_axis, time_opf_opf, 'o-',label='OPF-U + OPF-S')
  ax.plot(x_axis, time_opf_svm, 'o-',label='OPF-U + SVM')
  ax.plot(x_axis, time_overfeat_opf, 'o-',label='OVERFEAT + OPF-S')
  ax.plot(x_axis, time_overfeat_svm, 'o-',label='OVERFEAT + SVM')
  ax.plot(x_axis, time_cnn_yuv, 'o-',label='CNN YUV')
  plt.xlabel(u'Número de categorias')
  plt.ylabel(u'Tempo total (normalizado)')
  #plt.ylim(0.0,1.0)
  plt.xlim(5.0,36.0)
  #plt.legend(bbox_to_anchor=(0.42, 0.42), )
  plt.legend(loc='lower right', prop={'size':9})
  ax.set_xticks((5.0,10.0,15.0,36.0))
  ax.set_xticklabels(('5','10','15','36'))
  fig.savefig(filename)


def plot_base_criada_tempo_classificacao(filename):
  x_axis = [5, 10, 15, 36]
  time_kmeans_svm   = [0.134162,0.340527,0.649457,2.885613]
  time_kmeans_opf   = [2.984562,6.851466,31.937148,45.905539]
  time_opf_opf      = [0.958790,3.462291,6.389268,30.579227]
  time_opf_svm      = [0.296754,1.092073,1.905112,14.290473]
  time_cnn_yuv      = [3379.0,14527.0,40382.0,64561.0]
  time_overfeat_svm = [2.959637,6.599604,15.510890,58.222503]
  time_overfeat_opf = [23.281906,76.119970,169.050456,839.188609]

  time_kmeans_svm = (time_kmeans_svm - numpy.min(time_kmeans_svm))/(numpy.max(time_kmeans_svm) - numpy.min(time_kmeans_svm))
  time_kmeans_opf = (time_kmeans_opf - numpy.min(time_kmeans_opf))/(numpy.max(time_kmeans_opf) - numpy.min(time_kmeans_opf))
  time_opf_opf = (time_opf_opf - numpy.min(time_opf_opf))/(numpy.max(time_opf_opf) - numpy.min(time_opf_opf))
  time_opf_svm = (time_opf_svm - numpy.min(time_opf_svm))/(numpy.max(time_opf_svm) - numpy.min(time_opf_svm))
  time_cnn_yuv = (time_cnn_yuv - numpy.min(time_cnn_yuv))/(numpy.max(time_cnn_yuv) - numpy.min(time_cnn_yuv))
  time_overfeat_svm = (time_overfeat_svm - numpy.min(time_overfeat_svm))/(numpy.max(time_overfeat_svm) - numpy.min(time_overfeat_svm))
  time_overfeat_opf = (time_overfeat_opf - numpy.min(time_overfeat_opf))/(numpy.max(time_overfeat_opf) - numpy.min(time_overfeat_opf))

  #import math
  #time_kmeans_svm = numpy.log(time_kmeans_svm)
  #time_kmeans_opf = numpy.log(time_kmeans_opf)
  #time_opf_opf = numpy.log(time_opf_opf)
  #time_opf_svm = numpy.log(time_opf_svm)
  #time_cnn_yuv = numpy.log(time_cnn_yuv)
  #time_overfeat_svm = numpy.log(time_overfeat_svm)
  #time_overfeat_opf = numpy.log(time_overfeat_opf)


  fig, ax = plt.subplots()
  ax.plot(x_axis, time_kmeans_svm, 'o-',label='Kmeans + SVM')
  ax.plot(x_axis, time_kmeans_opf, 'o-',label='Kmeans + OPF-S')
  ax.plot(x_axis, time_opf_opf, 'o-',label='OPF-U + OPF-S')
  ax.plot(x_axis, time_opf_svm, 'o-',label='OPF-U + SVM')
  ax.plot(x_axis, time_overfeat_opf, 'o-',label='OVERFEAT + OPF-S')
  ax.plot(x_axis, time_overfeat_svm, 'o-',label='OVERFEAT + SVM')
  ax.plot(x_axis, time_cnn_yuv, 'o-',label='CNN YUV')
  plt.xlabel(u'Número de categorias')
  plt.ylabel(u'Tempo de treinamento classificador (segundos)')
  plt.title(u'Base criada - Tempo de treinamento x Quantidade de categorias')
  #plt.ylim(0.0,1.0)
  plt.xlim(5.0,36.0)
  #plt.legend(bbox_to_anchor=(0.42, 0.42), )
  plt.legend(loc='upper left', prop={'size':9})
  ax.set_xticks((5.0,10.0,15.0,36.0))
  ax.set_xticklabels(('5','10','15','36'))
  fig.savefig(filename)

def plot_base_criada_tempo_agrupamento(filename):
  x_axis = [5, 10, 15, 36]
  time_kmeans_svm   = [0.134162,0.340527,0.649457,2.885613]
  time_kmeans_opf   = [2.984562,6.851466,31.937148,45.905539]
  time_opf_opf      = [0.958790,3.462291,6.389268,30.579227]
  time_opf_svm      = [0.296754,1.092073,1.905112,14.290473]
  
  time_kmeans_svm = (time_kmeans_svm - numpy.min(time_kmeans_svm))/(numpy.max(time_kmeans_svm) - numpy.min(time_kmeans_svm))
  time_kmeans_opf = (time_kmeans_opf - numpy.min(time_kmeans_opf))/(numpy.max(time_kmeans_opf) - numpy.min(time_kmeans_opf))
  time_opf_opf = (time_opf_opf - numpy.min(time_opf_opf))/(numpy.max(time_opf_opf) - numpy.min(time_opf_opf))
  time_opf_svm = (time_opf_svm - numpy.min(time_opf_svm))/(numpy.max(time_opf_svm) - numpy.min(time_opf_svm))
  
  #import math
  #time_kmeans_svm = numpy.log(time_kmeans_svm)
  #time_kmeans_opf = numpy.log(time_kmeans_opf)
  #time_opf_opf = numpy.log(time_opf_opf)
  #time_opf_svm = numpy.log(time_opf_svm)
  #time_cnn_yuv = numpy.log(time_cnn_yuv)
  #time_overfeat_svm = numpy.log(time_overfeat_svm)
  #time_overfeat_opf = numpy.log(time_overfeat_opf)


  fig, ax = plt.subplots()
  ax.plot(x_axis, time_kmeans_svm, 'o-',label='Kmeans + SVM')
  ax.plot(x_axis, time_kmeans_opf, 'o-',label='Kmeans + OPF-S')
  ax.plot(x_axis, time_opf_opf, 'o-',label='OPF-U + OPF-S')
  ax.plot(x_axis, time_opf_svm, 'o-',label='OPF-U + SVM')
  plt.xlabel(u'Número de categorias')
  plt.ylabel(u'Tempo de treinamento classificador (segundos)')
  plt.title(u'Base criada - Tempo de treinamento x Quantidade de categorias')
  #plt.ylim(0.0,1.0)
  plt.xlim(5.0,36.0)
  #plt.legend(bbox_to_anchor=(0.42, 0.42), )
  plt.legend(loc='upper left', prop={'size':9})
  ax.set_xticks((5.0,10.0,15.0,36.0))
  ax.set_xticklabels(('5','10','15','36'))
  fig.savefig(filename)

def print_means(title, df):
  grouped = df.groupby(['feature_type','descriptor_type','thumbnail_size'])
  res = pd.DataFrame()
  
  #res['mean'] = grouped['accuracy'].mean()
  res['mean'] = grouped['F1'].mean()
  print("============= " + title + " =============")
  #print res
  print res['mean'].max()
  #print "Acc: " + str(res['mean']['MSER']['SIFT'][2])

def plot_acuracia_ksize_gui001(filename):
  x_axis = [50, 100, 200, 400, 600, 800, 1000, 1200, 1400]
  time_kmeans_svm   = [0.394444,0.470370,0.523148,0.550926,0.560185,0.563889,0.597222,0.588889,0.600926]
  std_kmeans_svm    = [0.038889,0.027405,0.023625,0.004243,0.021216,0.022048,0.038188,0.034694,0.014254]
  time_kmeans_opf   = [0.405556,0.430556,0.453704,0.475000,0.493519,0.475926,0.463889,0.479630,0.478704]
  std_kmeans_opf    = [0.033793,0.010015,0.029441,0.027358,0.019510,0.001604,0.030046,0.013127,0.014254]
  

  
  fig, ax = plt.subplots()
  ax.plot(x_axis, time_kmeans_svm, 'o-',label='Kmeans + SVM')
  ax.plot(x_axis, time_kmeans_opf, 'o-',label='Kmeans + OPF-S')
  plt.xlabel(u'Tamanho do dicionário')
  plt.ylabel(u'Acurácia')
  plt.title(u'Base criada - Acurácia x Tamanho do dicionário')
  plt.ylim(0.0,1.0)
  plt.xlim(0.0,1400.0)
  
  plt.legend(loc='upper left', prop={'size':9})
  ax.set_xticks(x_axis)
  ax.set_xticklabels(x_axis)
  fig.savefig(filename)

def plot_acuracia_ksize_caltech101(filename):
  x_axis = [50, 100, 200, 400, 600, 800, 1000, 1200, 1400]
  time_kmeans_svm   = [0.367590, 0.395689, 0.434950, 0.450346, 0.450346, 0.454580, 0.456120, 0.480754, 0.461509]
  #std_kmeans_svm    = [0.038889,0.027405,0.023625,0.004243,0.021216,0.022048,0.038188,0.034694,0.014254]
  time_kmeans_opf   =  [0.334103, 0.355658, 0.372594, 0.367590, 0.369515, 0.369130, 0.349115, 0.357198, 0.367206]
  #std_kmeans_opf    = [0.033793,0.010015,0.029441,0.027358,0.019510,0.001604,0.030046,0.013127,0.014254]
  

  fig, ax = plt.subplots()
  ax.plot(x_axis, time_kmeans_svm, 'o-',label='Kmeans + SVM')
  ax.plot(x_axis, time_kmeans_opf, 'o-',label='Kmeans + OPF-S')
  plt.xlabel(u'Tamanho do dicionário')
  plt.ylabel(u'Acurácia')
  plt.title(u'Base Caltech101 - Acurácia x Tamanho do dicionário')
  plt.ylim(0.0,1.0)
  plt.xlim(0.0,1400.0)
  
  plt.legend(loc='upper left', prop={'size':9})
  ax.set_xticks(x_axis)
  ax.set_xticklabels(x_axis)
  fig.savefig(filename)

def plot_ksize_tempo_treinamento_kmeans_caltech101_gui001(filename):
  x_axis = [50, 100, 200, 400, 600, 800, 1000, 1200, 1400]
  time_kmeans_caltech101   = [14.566073, 21.730389, 35.460297, 56.354211, 81.770537, 103.190484, 124.501769, 144.714140, 169.599114]
  time_kmeans_gui001   =  [13.792901, 22.318488, 34.954946, 57.703131, 80.274368, 102.978308, 120.925050, 154.003349, 174.624274]


  fig, ax = plt.subplots()
  ax.plot(x_axis, time_kmeans_caltech101, 'o-',label='Caltech101')
  ax.plot(x_axis, time_kmeans_gui001, 'o-',label='Base criada')
  plt.xlabel(u'Tamanho do dicionário')
  plt.ylabel(u'Tempo de treinamento (segundos)')
  plt.title(u'Tempo de treinamento (KMeans) x Tamanho do dicionário')
  #plt.ylim(0.0,1.0)
  plt.xlim(0.0,1400.0)
  
  plt.legend(loc='upper left', prop={'size':9})
  ax.set_xticks(x_axis)
  ax.set_xticklabels(x_axis)
  fig.savefig(filename)

def plot_tempo_classificador_ksize_gui001(filename):
  x_axis = [50, 100, 200, 400, 600, 800, 1000, 1200, 1400]
  time_kmeans_svm   = [1.614668, 1.649338,1.776783,2.085014,2.414104,2.720824,2.940867,3.263202,3.506558]
  time_kmeans_opf   =  [8.837159, 16.376920, 32.920072, 64.875020, 94.094220, 124.130328, 152.170091, 183.595885,210.884048]
  

  fig, ax = plt.subplots()
  ax.plot(x_axis, time_kmeans_svm, 'o-',label='Kmeans + SVM')
  ax.plot(x_axis, time_kmeans_opf, 'o-',label='Kmeans + OPF-S')
  plt.xlabel(u'Tamanho do dicionário')
  plt.ylabel(u'Tempo de treinamento (segundos)')
  plt.title(u'Base criada - Tempo de treinamento x Tamanho do dicionário')
  #plt.ylim(0.0,1.0)
  plt.xlim(0.0,1400.0)
  
  plt.legend(loc='upper left', prop={'size':9})
  ax.set_xticks(x_axis)
  ax.set_xticklabels(x_axis)
  fig.savefig(filename)

def plot_tempo_classificador_ksize_caltech101(filename):
  x_axis = [50, 100, 200, 400, 600, 800, 1000, 1200, 1400]
  time_kmeans_svm   =  [3.576211,3.663198,3.971779,4.429579,4.912357,5.193199,5.563007,5.921405,6.224179]
  time_kmeans_opf   =  [4.926422,8.772612,17.947978,35.406691,51.999701,67.923419,83.649899,100.332589,116.148495]

  fig, ax = plt.subplots()
  ax.plot(x_axis, time_kmeans_svm, 'o-',label='Kmeans + SVM')
  ax.plot(x_axis, time_kmeans_opf, 'o-',label='Kmeans + OPF-S')
  plt.xlabel(u'Tamanho do dicionário')
  plt.ylabel(u'Tempo de treinamento (segundos)')
  plt.title(u'Base Caltech101 - Tempo de treinamento x Tamanho do dicionário')
  #plt.ylim(0.0,1.0)
  plt.xlim(0.0,1400.0)
  
  plt.legend(loc='upper left', prop={'size':9})
  ax.set_xticks(x_axis)
  ax.set_xticklabels(x_axis)
  fig.savefig(filename)


def plot_tempo_total_ksize_caltech101_gui001(filename):
  x_axis = [50, 100, 200, 400, 600, 800, 1000, 1200, 1400]
  time_kmeans_svm_gui001     =  [23.059964,34.253570,52.220209,86.186368,119.072199,151.987881,179.994080,223.867604,254.328509]
  time_kmeans_opf_gui001     =  [30.454298,47.698624,84.955467,150.412426,209.672944,273.784505,334.577831,405.389057,462.762053]
  time_kmeans_svm_caltech101 =  [25.625388, 35.138667, 53.537741, 84.222293, 119.478440, 150.431798, 181.140349, 210.624743, 244.757093]
  time_kmeans_opf_caltech101 =  [27.451320, 46.260469, 67.881357, 119.769362, 167.863522, 217.949817, 260.831726, 315.562416, 356.658560]

  fig, ax = plt.subplots()
  ax.plot(x_axis, time_kmeans_svm_gui001, 'o-',label='Kmeans + SVM (Base criada)')
  ax.plot(x_axis, time_kmeans_opf_gui001, 'o-',label='Kmeans + OPF-S (Base criada)')
  ax.plot(x_axis, time_kmeans_svm_caltech101, 'o-',label='Kmeans + SVM (Caltech101)')
  ax.plot(x_axis, time_kmeans_opf_caltech101, 'o-',label='Kmeans + OPF-S (Caltech101)')
  plt.xlabel(u'Tamanho do dicionário')
  plt.ylabel(u'Tempo de total (segundos)')
  plt.title(u'Tempo total x Tamanho do dicionário')
  #plt.ylim(0.0,1.0)
  plt.xlim(0.0,1400.0)
  
  plt.legend(loc='upper left', prop={'size':9})
  ax.set_xticks(x_axis)
  ax.set_xticklabels(x_axis)
  fig.savefig(filename)


def main():
  parser = argparse.ArgumentParser(prog='plot.py', usage='%(prog)s [options]',
      description='Plot data', 
      epilog="")
  #parser.add_argument('-i', '--input', help='input file with results', required=True)
  args = parser.parse_args()


  #####CALTECH101
  #df = get_dataframe("/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/caltech101/results_kmeans_svm.txt")
  #plot_accuracy(df, u'Caltech101 - Acurácia x Tamanho - Kmeans + SVM', 'plots/caltech101_acuracia_x_tamanho_kmeans_svm.png')

  #df = get_dataframe("/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/caltech101/results_kmeans_opf.txt")
  #plot_accuracy(df, u'Caltech101 - Acurácia x Tamanho - Kmeans + OPF-S', 'plots/caltech101_acuracia_x_tamanho_kmeans_opf.png')

  #df = get_dataframe("/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/caltech101/results_opf_opf.txt")
  #plot_accuracy(df, u'Caltech101 - Acurácia x Tamanho - OPF-U + OPF-S', 'plots/caltech101_acuracia_x_tamanho_opf_opf.png')

  #df = get_dataframe("/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/caltech101/results_opf_svm.txt")
  #plot_accuracy(df, u'Caltech101 - Acurácia x Tamanho - OPF-U + SVM', 'plots/caltech101_acuracia_x_tamanho_opf_svm.png')

  df_kmeans_svm = get_dataframe("/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/caltech101/results_kmeans_svm.txt")
  df_kmeans_opf = get_dataframe("/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/caltech101/results_kmeans_opf.txt")
  df_opf_opf = get_dataframe("/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/caltech101/results_opf_opf.txt")
  df_opf_svm = get_dataframe("/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/caltech101/results_opf_svm.txt")
  df_overfeat = get_dataframe("/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/caltech101/results_overfeat.txt")
  cnn_yuv_accuracy = 0.54
  cnn_rgb_accuracy = 0.48
  
  #plot_mser_sift_all(df_kmeans_svm,df_kmeans_opf,df_opf_opf,df_opf_svm,df_overfeat, cnn_yuv_accuracy, cnn_rgb_accuracy, 
  #  u'Caltech101 - Acurácia (MSER+SIFT) x Agrupador+Classificador', 
  #  'plots/caltech101_acuracia_classificadores_mser_sift.png')
  
  plot_accuracy_best_caltech101(df_kmeans_svm,df_kmeans_opf,df_opf_opf,df_opf_svm,df_overfeat, cnn_yuv_accuracy, cnn_rgb_accuracy, 
    u'Caltech101 - Acurácia x Agrupador+Classificador', 
    'plots/caltech101_acuracia_classificadores.pdf')

  #plot_time_total_caltech101('plots/caltech101_tempo_total.png')

  #CNN
  #train_log = '/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/caltech101/cnn_caltech101_32/train.log'
  #test_log = '/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/caltech101/cnn_caltech101_32/test.log'
  #train_time, train_acc, train_ite = cnn_visualizer.read_cnn_data(train_log)
  #test_time, test_acc, test_ite = cnn_visualizer.read_cnn_data(test_log)
  #cnn_visualizer.plot_acc_vs_time(train_time, train_acc, test_time, test_acc, u"Caltech101 - CNN YUV - Tempo x Acurácia", 'plots/caltech101_cnn_yuv_acuracia_x_tempo.png')

  #train_log = '/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/caltech101/cnn_rgb_caltech101_32/train.log'
  #test_log = '/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/caltech101/cnn_rgb_caltech101_32/test.log'
  #train_time, train_acc, train_ite = cnn_visualizer.read_cnn_data(train_log)
  #test_time, test_acc, test_ite = cnn_visualizer.read_cnn_data(test_log)
  #cnn_visualizer.plot_acc_vs_time(train_time, train_acc, test_time, test_acc, u"Caltech101 - CNN RGB - Tempo x Acurácia", 'plots/caltech101_cnn_rgb_acuracia_x_tempo.png')

  #plot_acuracia_ksize_caltech101('plots/caltech101_ksize_acuracia_k.png')

  #####GUI001
  #df = get_dataframe("/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/gui001/results_kmeans_svm.txt")
  #plot_accuracy(df, u'Base Criada (36 cat.) - Acurácia x Tamanho - Kmeans + SVM', 'plots/gui001_acuracia_x_tamanho_kmeans_svm.png')

  #df = get_dataframe("/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/gui001/results_kmeans_opf.txt")
  #plot_accuracy(df, u'Base Criada (36 cat.) - Acurácia x Tamanho - Kmeans + OPF-S', 'plots/gui001_acuracia_x_tamanho_kmeans_opf.png')

  #df = get_dataframe("/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/gui001/results_opf_opf.txt")
  #plot_accuracy(df, u'Base Criada (36 cat.) - Acurácia x Tamanho - OPF-U + OPF-S', 'plots/gui001_acuracia_x_tamanho_opf_opf.png')

  #df = get_dataframe("/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/gui001/results_opf_svm.txt")
  #plot_accuracy(df, u'Base Criada (36 cat.) - Acurácia x Tamanho - OPF-U + SVM', 'plots/gui001_acuracia_x_tamanho_opf_svm.png')

  df_kmeans_svm = get_dataframe("/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/gui001/results_kmeans_svm.txt")
  df_kmeans_opf = get_dataframe("/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/gui001/results_kmeans_opf.txt")
  df_opf_opf = get_dataframe("/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/gui001/results_opf_opf.txt")
  df_opf_svm = get_dataframe("/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/gui001/results_opf_svm.txt")
  df_overfeat = get_dataframe("/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/gui001/results_overfeat.txt")
  cnn_yuv_accuracy = 0.69
  cnn_rgb_accuracy = 0.71
  
  #plot_mser_sift_all(df_kmeans_svm,df_kmeans_opf,df_opf_opf,df_opf_svm,df_overfeat, cnn_yuv_accuracy, cnn_rgb_accuracy, 
  #  u'Base criada (36 cat.) - Acurácia (MSER+SIFT) x Agrupador+Classificador', 
  #  'plots/gui001_acuracia_classificadores_mser_sift.png')

  plot_accuracy_best_gui001(df_kmeans_svm,df_kmeans_opf,df_opf_opf,df_opf_svm,df_overfeat, cnn_yuv_accuracy, cnn_rgb_accuracy, 
    u'Base criada (36 cat.) - Acurácia x Agrupador+Classificador', 
    'plots/gui001_acuracia_classificadores.pdf')

  #plot_time_total_gui001('plots/gui001_tempo_total.png')

  #CNN
  #train_log = '/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/gui001/cnn_gui001_32/train.log'
  #test_log = '/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/gui001/cnn_gui001_32/test.log'
  #train_time, train_acc, train_ite = cnn_visualizer.read_cnn_data(train_log)
  #test_time, test_acc, test_ite = cnn_visualizer.read_cnn_data(test_log)
  #cnn_visualizer.plot_acc_vs_time(train_time, train_acc, test_time, test_acc, u"Base Criada (36 cat.) - CNN YUV - Tempo x Acurácia", 'plots/gui001_cnn_yuv_acuracia_x_tempo.png')

  #train_log = '/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/gui001/cnn_rgb_gui001_32/train.log'
  #test_log = '/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/gui001/cnn_rgb_gui001_32/test.log'
  #train_time, train_acc, train_ite = cnn_visualizer.read_cnn_data(train_log)
  #test_time, test_acc, test_ite = cnn_visualizer.read_cnn_data(test_log)
  #cnn_visualizer.plot_acc_vs_time(train_time, train_acc, test_time, test_acc, u"Base Criada (36 cat.) - CNN RGB - Tempo x Acurácia", 'plots/gui001_cnn_rgb_acuracia_x_tempo.png')

  #train_log = '/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/gui001_5/cnn_gui001_5_32/train.log'
  #test_log = '/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/gui001_5/cnn_gui001_5_32/test.log'
  #train_time, train_acc, train_ite = cnn_visualizer.read_cnn_data(train_log)
  #test_time, test_acc, test_ite = cnn_visualizer.read_cnn_data(test_log)
  #cnn_visualizer.plot_acc_vs_time(train_time, train_acc, test_time, test_acc, u"Base Criada (5 cat.) - CNN YUV - Tempo x Acurácia", 'plots/gui001_5_cnn_yuv_acuracia_x_tempo.png')

  #train_log = '/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/gui001_10/cnn_gui001_10_32/train.log'
  #test_log = '/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/gui001_10/cnn_gui001_10_32/test.log'
  #train_time, train_acc, train_ite = cnn_visualizer.read_cnn_data(train_log)
  #test_time, test_acc, test_ite = cnn_visualizer.read_cnn_data(test_log)
  #cnn_visualizer.plot_acc_vs_time(train_time, train_acc, test_time, test_acc, u"Base Criada (10 cat.) - CNN YUV - Tempo x Acurácia", 'plots/gui001_10_cnn_yuv_acuracia_x_tempo.png')

  #train_log = '/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/gui001_15/cnn_gui001_15_32/train.log'
  #test_log = '/Volumes/Dados/GUILHERME/MESTRADO/UDESC/Dissertacao/Resultados/gui001_15/cnn_gui001_15_32/test.log'
  #train_time, train_acc, train_ite = cnn_visualizer.read_cnn_data(train_log)
  #test_time, test_acc, test_ite = cnn_visualizer.read_cnn_data(test_log)
  #cnn_visualizer.plot_acc_vs_time(train_time, train_acc, test_time, test_acc, u"Base Criada (15 cat.) - CNN YUV - Tempo x Acurácia", 'plots/gui001_15_cnn_yuv_acuracia_x_tempo.png')

  #plot_time_consult_gui001('plots/gui001_tempo_consulta.png')


  #N CATEGORIAS
  #plot_base_criada_n_categorias_x_acuracia('plots/gui_n_categorias_x_acuracia.png')
  #plot_base_criada_n_categorias_x_f1('plots/gui_n_categorias_x_f1.png')
  #plot_base_criada_n_categorias_x_tempo('plots/gui_n_categorias_x_tempo.png')
  #plot_base_criada_tempo_classificacao('plots/gui_n_categorias_x_tempo_classificacao.png')
  
  #plot_acuracia_ksize_gui001('plots/gui001_ksize_acuracia_k.png')
  #plot_ksize_tempo_treinamento_kmeans_caltech101_gui001('plots/gui001_caltech101_ksize_tempo_agrupador.png')
  #plot_tempo_classificador_ksize_gui001('plots/gui001_ksize_tempo_classificador.png')
  #plot_tempo_classificador_ksize_caltech101('plots/caltech101_ksize_tempo_classificador.png')
  #plot_tempo_total_ksize_caltech101_gui001('plots/gui001_caltech101_tempo_total.png')
if __name__ == "__main__":
  main()



#http://matplotlib.org/examples/api/barchart_demo.html
#http://matplotlib.org/examples/pylab_examples/barchart_demo.html

