import pylab
import imageio
filename = 'project_video.mp4'
vid = imageio.get_reader(filename,  'ffmpeg')
num_frames=vid._meta['nframes']
print(num_frames)
nums = [0,1,2,3,4]
for num in nums:
    image = vid.get_data(num)
    fig = pylab.figure()
    fig.suptitle('image #{}'.format(num), fontsize=20)
    pylab.imshow(image)
pylab.show()