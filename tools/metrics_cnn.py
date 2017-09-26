import numpy
from sklearn import metrics
import argparse

def get_results_data(file_results_path):
  targets = None
  predictions = None
  results_file = open(file_results_path, 'rb')
  for idx_line, line in enumerate(results_file):
    line = line[0:-1] if line.endswith(",") else line
    line = line[0:-1] if line.endswith("\n") else line
    line = line[0:-1] if line.endswith("") else line
    if idx_line == 0:
      targets = line.split(",")
    elif idx_line == 1:
      predictions = line.split(",")

  return targets, predictions

def main():
  parser = argparse.ArgumentParser(prog='plot.py', usage='%(prog)s [options]',
      description='Plot data', 
      epilog="")
  parser.add_argument('-i', '--input', help='input file with results', required=True)
  args = parser.parse_args()

  file_results_path = args.input
  targets, predictions = get_results_data(file_results_path)

  print "n_targets", len(targets)
  print "n_predictions", len(predictions)
  
  accuracy = metrics.accuracy_score(targets, predictions)
  precision = metrics.precision_score(targets, predictions)
  recall = metrics.recall_score(targets, predictions)
  f1 = metrics.f1_score(targets, predictions)

  print "Accuracy: ", accuracy
  print "Precision: ", precision
  print "Recall: ", recall
  print "F1: ", f1

if __name__ == "__main__":
  main()


