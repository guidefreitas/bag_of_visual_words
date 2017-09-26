import os

def get_files(path):
  ''' Retorna uma tupla com (caminho_da_imagem, pasta)
  '''

  files_tuple = []
  for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
      if filename.lower().endswith('jpg') or filename.lower().endswith('png') or filename.lower().endswith('ppm'):
        path, folder = os.path.split(dirpath)
        folder = folder.replace('/','')
        files_tuple.append((os.path.join(dirpath, filename), folder))

  return files_tuple