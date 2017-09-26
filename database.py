import sqlite3
import os
from threading import Thread
from Queue import Queue

class Database(Thread):
  database_name = ""
  conn = None
  def __init__(self, database_name):
    super(Database, self).__init__()
    self.database_name = database_name
    self.reqs=Queue()
    self.start()

  def run(self):
    if not os.path.exists(self.database_name):
      self.create_database()
    else:
      self.conn = sqlite3.connect(self.database_name) # ou use :memory: para bota-lo na memoria RAM
      self.conn.row_factory = sqlite3.Row
      self.cursor = self.conn.cursor()

      while True:
        req, arg, res = self.reqs.get()
        if req=='--close--': break
        cursor = self.conn.cursor()
        cursor.execute(req, arg)
        if res:
          for rec in cursor:
            res.put(rec)
          res.put('--no more--')
      self.conn.close()

  def execute(self, req, arg=None, res=None):
    self.reqs.put((req, arg or tuple(), res))

  def select(self, req, arg=None):
    res=Queue()
    self.execute(req, arg, res)
    while True:
      rec=res.get()
      if rec=='--no more--': break
      yield rec
  
  def close(self):
    self.execute('--close--')

  def create_database(self):
    self.conn = sqlite3.connect(self.database_name) # ou use :memory: para bota-lo na memoria RAM
    self.conn.row_factory = sqlite3.Row
    self.cursor = self.conn.cursor()
    self.cursor.execute("""CREATE TABLE images
                     (image_path text not null, 
                     feature_type text not null,
                     descriptor_type text not null,
                     thumbnail_size text not null,
                     descriptors BLOB,
                     PRIMARY KEY (image_path, feature_type,descriptor_type, thumbnail_size ))
                  """)

  def get_image(self, image_path, feature_type, descriptor_type, thumbnail_size):
    sql = '''SELECT * FROM images
             WHERE image_path = ?
             AND feature_type = ?
             AND descriptor_type = ?
             AND thumbnail_size = ?'''
    
    for data in self.select(sql, [str(image_path), str(feature_type), str(descriptor_type), str(thumbnail_size)]):
        return data
    
  def insert_image(self, image_path, feature_type, descriptor_type, thumbnail_size):
    sql = """INSERT INTO images(image_path,feature_type,descriptor_type, thumbnail_size) 
    VALUES (?, ?, ?, ?)"""
    res = self.cursor.execute(sql, [str(image_path), str(feature_type), str(descriptor_type), str(thumbnail_size)])
    self.conn.commit()
    entry = self.get_image(image_path, feature_type, descriptor_type, thumbnail_size)
    return entry

  def create_image(self, image_path, feature_type, descriptor_type, thumbnail_size, descriptors):
    if self.get_image(image_path, feature_type, descriptor_type, thumbnail_size) == None:
      self.insert_image(image_path, feature_type, descriptor_type, thumbnail_size)

    sql = '''UPDATE images
             SET descriptors = ? 
             WHERE image_path = ? 
             AND feature_type = ?
             AND descriptor_type = ?
             AND thumbnail_size = ?'''
    self.cursor.execute(sql, [sqlite3.Binary(descriptors), str(image_path), str(feature_type), str(descriptor_type), str(thumbnail_size)])
    self.conn.commit()
    return self.get_image(image_path,feature_type,descriptor_type, thumbnail_size)



#import database    
#db = database.Database('mydatabase.db')
#image = db.add_image('/home/image/teste.jpg', 'SIFT', 'SIFT')
#print image

