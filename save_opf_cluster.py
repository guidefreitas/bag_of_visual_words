import featureextractor
import sys

image_path = sys.argv[1]
print "Processing: ", image_path
kp, desc = featureextractor.extract_descriptor(image_path, (256,256), feature_type="SIFT", descriptor_type="SIFT")
print "Descritores: ", len(desc)

n_samples = len(desc)
n_features = len(desc[0])
result_file = open("desc_opf.txt", "w")
result_file.write(str(n_samples) + " " + "0" + " " + str(n_features) + "\n")
for idx, dsc in enumerate(desc):
  result_file.write(str(idx) + " ")
  for d in dsc:
    result_file.write(str(d) + " ")
  result_file.write("\n")
result_file.close()
