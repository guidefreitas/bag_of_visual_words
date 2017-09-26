import caltech


dataset_path = "/Volumes/Imagens/CALTECH256/256_ObjectCategories"
train_path = "/Volumes/Imagens/CALTECH256/train"
test_path = "/Volumes/Imagens/CALTECH256/validation"
validation_path = "/Volumes/Imagens/CALTECH256/test"

ctk = caltech.Caltech(dataset_path=dataset_path, train_path=train_path,
  validation_path=validation_path, test_path=test_path)

ctk.process()