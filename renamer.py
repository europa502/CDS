import os,argparse

def rename(dirs):
	counter=0
	#print os.listdir(dirs)
	for files in os.listdir(dirs):
		#print files
		os.rename(dirs+files,dirs+str(counter)+".jpeg")
		counter=counter+1
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dir',
      type=str,
      default='',
      help='Path to folders of labeled image')

  dir= parser.parse_args()
  print dir.dir

rename(dir.dir)
