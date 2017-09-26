from collections import namedtuple
import argparse
import ast
import numpy

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

def pprinttable(rows):
  if len(rows) > 1:
    headers = rows[0]._fields
    lens = []
    for i in range(len(rows[0])):
      lens.append(len(max([x[i] for x in rows] + [headers[i]],key=lambda x:len(str(x)))))
    formats = []
    hformats = []
    for i in range(len(rows[0])):
      if isinstance(rows[0][i], int):
        formats.append("%%%dd" % lens[i])
      else:
        formats.append("%%-%ds" % lens[i])
      hformats.append("%%-%ds" % lens[i])
    pattern = " | ".join(formats)
    hpattern = " | ".join(hformats)
    separator = "-+-".join(['-' * n for n in lens])
    print hpattern % tuple(headers)
    print separator
    for line in rows:
      print pattern % tuple(line)
  elif len(rows) == 1:
    row = rows[0]
    hwidth = len(max(row._fields,key=lambda x: len(x)))
    for i in range(len(row)):
      print "%*s = %s" % (hwidth,row._fields[i],row[i])

def results_data_to_row(results_data):
  #Kmeans+SVM|{'descriptor_type': 'SIFT', 'time_classificator_fit': 2.052706003189087, 'feature_type': 'SURF', 'thumbnail_size': (64, 64), 'F1': 0.33184836239007048, 'recall': 0.31385369840621169, 'time_cluster_fit': 4.337567090988159, 'n_sample_descriptors': 15000, 'precision': 0.4005507233166789, 'k_value': 100, 'n_sample_images': 100, 'time_total': 130.07487082481384, 'accuracy': 0.31385369840621169}
  Row = namedtuple('Row',['type',
                          'descriptor', 
                          'feature', 
                          'thumb', 
                          'k', 
                          'real_k', 
                          'd_type', 
                          'd_param',
                          'n_desc', 
                          'n_real_desc', 
                          'F1', 
                          't_total'])
  data = []
  for res_type in results_data:
    for res_data in results_data[res_type]:
      if str(res_type) == "OPF+OPF" or str(res_type) == "OPF+SVM":
        data_item = Row(str(res_type), 
          str(res_data['descriptor_type']), 
          str(res_data['feature_type']), 
          str(res_data['thumbnail_size'][0]), 
          str(res_data['k_value']), 
          str(res_data['real_k_value']) if (str(res_type) == "OPF+OPF" or str(res_type) == "OPF+SVM") else str(res_data['k_value']),
          str(res_data['distance_type']) if (str(res_type) == "OPF+OPF" or str(res_type) == "OPF+SVM") else " ",
          str(res_data['distance_param']) if (str(res_type) == "OPF+OPF" or str(res_type) == "OPF+SVM") else " ",
          str(res_data['n_sample_descriptors']),
          str(res_data['real_desc_size']),
          "{:6.2f}".format(res_data['F1']),
          "{:6.2f}".format((res_data['time_total'])))
      elif str(res_type) == "OVERFEAT+SVM" or str(res_type) == "OVERFEAT+OPF":
        data_item = Row(str(res_type), 
          str(res_data['descriptor_type']), 
          str(res_data['feature_type']), 
          str(res_data['thumbnail_size'][0]), 
          str(res_data['k_value']), 
          str(res_data['k_value']),
          "",
          "",
          str(res_data['n_sample_descriptors']),
          "",
          "{:6.2f}".format(res_data['F1']),
          "{:6.2f}".format((res_data['time_total'])))
      else:
        data_item = Row(str(res_type), 
          str(res_data['descriptor_type']), 
          str(res_data['feature_type']), 
          str(res_data['thumbnail_size'][0]), 
          str(res_data['k_value']), 
          str(res_data['k_value']),
          "",
          "",
          str(res_data['n_sample_descriptors']),
          str(res_data['real_desc_size']),
          "{:6.2f}".format(res_data['F1']),
          "{:6.2f}".format((res_data['time_total'])))
        
      
      data.append(data_item)
  return data

def main():
  parser = argparse.ArgumentParser(prog='pretty_table.py', usage='%(prog)s [options]',
      description='Print table result', 
      epilog="")
  parser.add_argument('-i', '--input', help='input file with results', required=True)
  args = parser.parse_args()

  results_data = get_results_data(args.input)
  
  data = results_data_to_row(results_data)

  pprinttable(data)
    

if __name__ == "__main__":
  main()