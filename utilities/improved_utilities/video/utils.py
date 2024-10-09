import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from matplotlib import animation


def video_save(data_list, video_save_path, iteration, fps=100, dpi=100, prefix=""):
    if not os.path.exists(video_save_path):
        os.makedirs(video_save_path)
    fig = plt.figure(figsize=(12, 12))

    ims = []

    for i, data in enumerate(data_list):

        grid = make_grid(
            data.view([-1, 3, 32, 32]),
            value_range=(-1, 1),
            padding=0,
            nrow=2,
        )
        grid = grid / 2.0 + 0.5
        img = ToPILImage()(grid)
        im = plt.imshow(img)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True)
    ani.save(
        os.path.join(
            video_save_path,
            f"{prefix}_{iteration}.mp4",
        ),
        fps=fps,
        dpi=dpi,
    )
    # clean up
    plt.close(fig)
    plt.clf()
