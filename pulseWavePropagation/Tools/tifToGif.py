import matplotlib.animation as animation
import matplotlib.pyplot as plt
import imageio

from Configuration import Config as cfg
from Tools import utils

def build_gif(imgs, show_gif=True, save_gif=True, title=''):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    plt.rcParams['animation.ffmpeg_path'] = unicode('C:\\Users\\DIN035\\AppData\\Local\\Continuum\\Anaconda2\\pkgs\\ffmpeg-2.7.0-0\\Scripts\\ffmpeg.exe')

    ims = map(lambda x: (ax.imshow(x), ax.set_title(title)), imgs)

    im_ani = animation.ArtistAnimation(fig, ims, interval=cfg.gifInterval, repeat_delay=0, blit=False)

    #Not working without ffmpeg in PATH (need admin rights)
    #if save_gif:
        #im_ani.save('animation.gif', writer='imagemagick')

    if show_gif:
        plt.show()

    return

def run(imgDir = cfg.imgDir):
    print imgDir
    loadedImages = utils.loadAllImages(imgDir, cfg.imgFormats)
    print len(loadedImages)
    build_gif(loadedImages,True,True)