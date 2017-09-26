# -*- coding: utf-8 -*-
from collections import namedtuple
import matplotlib
#matplotlib.use('Agg')
import argparse
import ast
import matplotlib.pyplot as plt
import numpy


def read_cnn_data(file_path):
  time_data = []
  acc_data = []
  ite_data = []
  results_file = open(file_path, 'rb')
  for idx, line in enumerate(results_file):
    if line.startswith("#") or line == None or idx == 0:
      continue
    line = line.replace(' ', '')
    line_data = line.split('\t')
    
    v_time = float(line_data[0])
    v_acc = float(line_data[1])
    time_data.append(v_time)
    acc_data.append(v_acc)
    ite_data.append(idx)
  results_file.close()
  time_data = numpy.asarray(time_data)
  acc_data = numpy.asarray(acc_data)
  ite_data = numpy.asarray(ite_data)
  return time_data, acc_data, ite_data

def plot_acc_vs_time(train_time, train_acc, test_time, test_acc, title, filename):
  train_time = train_time/60/60
  test_time = test_time/60/60
  plt.figure()
  plt.plot(train_time, train_acc, color="blue", label='Base Treinamento')
  plt.plot(test_time, test_acc, color="red", label='Base Teste')
  plt.title(title)
  plt.xlabel(u"Tempo (horas)")
  plt.ylabel(u"Acurácia")
  plt.ylim(0,100)
  plt.legend(loc='best')
  #plt.xlim(0,100)
  #plt.xlim(0.5,4.5)
  #plt.xticks(time_data)
  #plt.show()
  plt.tight_layout()
  plt.savefig(filename)

def main():
  parser = argparse.ArgumentParser(prog='cnn_visualizer.py', usage='%(prog)s [options]',
      description='Plot data', 
      epilog="")
  parser.add_argument('-train', '--train', help='input train file with results', required=True)
  parser.add_argument('-test', '--test', help='input test file with results', required=True)
  args = parser.parse_args()

  train_time, train_acc, train_ite = read_cnn_data(args.train)
  test_time, test_acc, test_ite = read_cnn_data(args.test)

  plot_acc_vs_time(train_time, train_acc, test_time, test_acc, u"Tempo vs Acurácia", 'plot.png')
  #plot_acc_vs_time(train_ite, train_acc, test_ite, test_acc)


if __name__ == "__main__":
  main()