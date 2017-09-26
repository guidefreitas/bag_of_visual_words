import sqlite3
import os

class Database:
  database_name = ""
  conn = None
  def __init__(self, database_name):
    self.database_name = database_name
    if not os.path.exists(self.database_name):
      self.create_database()
    else:
      self.conn = sqlite3.connect(self.database_name) # ou use :memory: para bota-lo na memoria RAM
      self.conn.row_factory = sqlite3.Row
      self.cursor = self.conn.cursor()

  def create_database(self):
    self.conn = sqlite3.connect(self.database_name) # ou use :memory: para bota-lo na memoria RAM
    self.conn.row_factory = sqlite3.Row
    self.cursor = self.conn.cursor()
    self.cursor.execute("""CREATE TABLE results
                     (execution_id INTEGER PRIMARY KEY not null, 
                     process_type text, 
                     k INTEGER,
                     n_sample_img INTEGER, 
                     n_sample_descriptors, INTEGER, 
                     image_size INTEGER, 
                     feature_type text,
                     descriptor_type text,
                     time_execution text,
                     accuracy text)
                  """)

#process_type (OVERFEAT,KMeans+SVM,KMeans+OPF,OPF+OPF,OPF+SVM) k (kmeans or opf kmax), n_sample_img, n_sample_descriptors, image_size, feature, descriptor
  
  def get_images_by_execution(self, execution_id):
    sql = 'SELECT * FROM images WHERE execution_id = ?'
    self.cursor.execute(sql, [execution_id])
    data = self.cursor.fetchall()
    return data

  def get_image_by_execution_and_url(self, execution_id, image_url):
    sql = 'SELECT * FROM images WHERE execution_id = ? AND image_url = ?'
    self.cursor.execute(sql, [execution_id, image_url])
    data = self.cursor.fetchone()
    return data

  def add_image_to_execution(self, execution_id, image_url, histogram=None):
    sql = """INSERT INTO images(execution_id,image_url,histogram) 
    VALUES (?, ?, ?)"""
    res = self.cursor.execute(sql, [execution_id, image_url, histogram])
    self.conn.commit()
  
  def update_image_histogram(self, execution_id, image_url, histogram):
    sql = """UPDATE images SET histogram = ? 
          WHERE execution_id = ? AND image_url = ?"""
    self.cursor.execute(sql, [histogram, execution_id, image_url])
    data = self.get_image_by_execution_and_url(execution_id, image_url)
    return data

  def add_execution(self, process_type, k, n_sample_img, n_sample_descriptors, image_size, feature_type, descriptor_type):
    sql = """INSERT INTO executions(process_type,k,n_sample_img,n_sample_descriptors,image_size,feature_type, descriptor_type) 
    VALUES (?, ?, ?, ?, ?, ?, ?)"""
    res = self.cursor.execute(sql, [process_type, k, n_sample_img, n_sample_descriptors, image_size, feature_type, descriptor_type])
    self.conn.commit()
    self.cursor.execute('SELECT last_insert_rowid()')
    file_entry = self.cursor.fetchone()
    sql = "SELECT * FROM executions WHERE execution_id = ?"
    self.cursor.execute(sql,[file_entry[0]])
    execution = self.cursor.fetchone()
    return execution

  def get_all_exections(self):
    self.cursor.execute('SELECT * FROM executions')
    data = self.cursor.fetchall()
    return data

  def get_exection_by_id(self, execution_id):
    sql = 'SELECT * FROM executions WHERE execution_id = ?'
    self.cursor.execute(sql, [execution_id])
    data = self.cursor.fetchone()
    return data

  def update_accuracy(self, execution_id, accuracy):
    sql = 'UPDATE executions SET accuracy = ? WHERE execution_id = ?'
    self.cursor.execute(sql, [accuracy, execution_id])
    data = self.get_exection_by_id(execution_id)
    return data

  def update_time_execution(self, execution_id, time_execution):
    sql = 'UPDATE executions SET time_execution = ? WHERE execution_id = ?'
    self.cursor.execute(sql, [time_execution, execution_id])
    data = self.get_exection_by_id(execution_id)
    return data      

    
db = Database('mydatabase.db')
execution = db.add_execution('OVERFEAT2', 100, 100, 1000, 128, 'OVERFEAT2', 'OVERFEAT2')
data = db.update_accuracy(execution['execution_id'], '98%')
data = db.update_time_execution(execution['execution_id'], '1295.0')
print data

data = db.get_images_by_execution(execution['execution_id'])
print "Images: ", len(data)
image = db.add_image_to_execution(execution['execution_id'], 'teste_url')
image = db.update_image_histogram(execution['execution_id'], 'teste_url', "teste_histogram")
print image

