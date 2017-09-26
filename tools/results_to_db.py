# -*- coding: utf-8 -*-

import argparse
import ast
import matplotlib.pyplot as plt
import numpy
import sqlite3
import os

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

class Database:
  database_name = ""
  conn = None
  cursor = None 

  def __init__(self, database_name):
    self.database_name = database_name
    if not os.path.exists(self.database_name):
      self.create_database()
      print "Database created: ", self.database_name
    else:
      self.conn = sqlite3.connect(self.database_name) # ou use :memory: para bota-lo na memoria RAM
      self.conn.row_factory = sqlite3.Row
      self.cursor = self.conn.cursor()

  def create_database(self):
    self.conn = sqlite3.connect(self.database_name) # ou use :memory: para bota-lo na memoria RAM
    self.conn.row_factory = sqlite3.Row
    self.cursor = self.conn.cursor()
    self.cursor.execute("""CREATE TABLE results
                     (result_id integer PRIMARY KEY not null, 
                     process_type text, 
                     k_value integer,
                     n_sample_images integer, 
                     n_sample_descriptors, integer, 
                     thumbnail_size text, 
                     feature_type text,
                     descriptor_type text,
                     time_classificator_fit real,
                     time_cluster_fit real,
                     time_total real,
                     accuracy text,
                     f1 real,
                     precision real,
                     recall real
                     )
                  """)

  def add_result_db(self, process_type, k_value, n_sample_images, n_sample_descriptors, 
    thumbnail_size, feature_type, descriptor_type, time_classificator_fit, time_cluster_fit,
    time_total, accuracy, f1, precision, recall):
    sql = """INSERT INTO results(
      process_type,
      k_value,
      n_sample_images,
      n_sample_descriptors,
      thumbnail_size,
      feature_type, 
      descriptor_type,
      time_classificator_fit,
      time_cluster_fit,
      time_total,
      accuracy,
      f1,
      precision,
      recall) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
    res = self.cursor.execute(sql, [process_type, k_value, n_sample_images, n_sample_descriptors, 
      thumbnail_size, feature_type, descriptor_type, time_classificator_fit, time_cluster_fit,
      time_total, accuracy, f1, precision, recall])
    self.conn.commit()
    self.cursor.execute('SELECT last_insert_rowid()')
    file_entry = self.cursor.fetchone()
    sql = "SELECT * FROM results WHERE result_id = ?"
    self.cursor.execute(sql,[file_entry[0]])
    execution = self.cursor.fetchone()
    return execution

  def insert_result(self, process_type, data):
    k_value = int(data['k_value'])
    n_sample_images = int(data['n_sample_images']) if data['n_sample_images'] != '' else 0
    n_sample_descriptors = int(data['n_sample_descriptors']) if data['n_sample_descriptors'] != '' else 0
    thumbnail_size = str(data['thumbnail_size'])
    feature_type = str(data['feature_type'])
    descriptor_type = str(data['descriptor_type'])
    time_classificator_fit = float(data['time_classificator_fit']) if data['time_classificator_fit'] != '' else 0
    time_cluster_fit = float(data['time_cluster_fit']) if data['time_cluster_fit'] != '' else 0
    time_total = float(data['time_total'])
    accuracy = float(data['accuracy'])
    f1 = float(data['F1'])
    precision = float(data['precision'])
    recall = float(data['recall'])

    db_record = self.add_result_db(process_type=process_type,
      k_value=k_value,
      n_sample_images=n_sample_images,
      n_sample_descriptors=n_sample_descriptors,
      thumbnail_size=thumbnail_size,
      feature_type=feature_type,
      descriptor_type=descriptor_type,
      time_classificator_fit=time_classificator_fit,
      time_cluster_fit=time_cluster_fit,
      time_total=time_total,
      accuracy=accuracy,
      f1=f1,
      precision=precision,
      recall=recall)
    return db_record

  def populate(self, file_results_path):
    results_data = get_results_data(file_results_path)

    for process_type_key in results_data:
      data = results_data[process_type_key]
      for result in data:
        self.insert_result(process_type_key, result)

def main():
  parser = argparse.ArgumentParser(prog='results_to_db.py', usage='%(prog)s [options]',
      description='Plot data', 
      epilog="")
  parser.add_argument('-i', '--input', help='input file with results', required=True)
  parser.add_argument('-d', '--database_name', default="results_database.db", help='input file with results')
  args = parser.parse_args()

  file_results_path = args.input
  database_name = args.database_name

  db = Database(database_name)
  db.populate(file_results_path)

  print  "Data saved to: ", database_name

if __name__ == "__main__":
  main()

##################################################
# QUERIES
##################################################
# SELECT process_type, feature_type, descriptor_type, accuracy, f1 FROM results
# ORDER BY accuracy DESC
################################
#import matplotlib libary
# import matplotlib.pyplot as plt

# #define plot size in inches (width, height) & resolution(DPI)
# fig = plt.figure(figsize=(4, 5), dpi=100)

# #define font size
# plt.rc("font", size=14)

# #define some data
# x = [1,2,3,4]
# y = [20, 21, 20.5, 20.8]

# #error data
# y_error = [0.12, 0.13, 0.2, 0.1]

# #plot data
# plt.plot(x, y, linestyle="dashed", marker="o", color="green")

# #plot only errorbars
# plt.errorbar(x, y, yerr=y_error, linestyle="None", marker="None", color="green")

# #configure  X axes
# plt.xlim(0.5,4.5)
# plt.xticks([1,2,3,4])

# #configure  Y axes
# plt.ylim(19.8,21.2)
# plt.yticks([20, 21, 20.5, 20.8])

# #labels
# plt.xlabel("this is X")
# plt.ylabel("this is Y", size=10)

# #title
# plt.title("Simple plot", size=30)

# #adjust plot
# plt.subplots_adjust(left=0.19)

# #show plot
# plt.show()