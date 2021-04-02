import imageio
import glob

anim_file = 'images_3-1.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('./images_1/image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)