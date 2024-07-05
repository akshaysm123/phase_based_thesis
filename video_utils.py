import numpy as np
import torch
import cv2

class Video:
    def __init__(self, videopath):
        self.videopath = videopath
        self.array, self.fps = self.load_video()
        self.dim = (self.array.shape[1], self.array.shape[2]) 

    def load_video(self):
        frames = []
        cap = cv2.VideoCapture(self.videopath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        ret = True
        while ret:
            ret, img = cap.read()
            if ret:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                frames.append(img)
        video = np.stack(frames, axis=0) # (Time,Height,Width,Color)
        return video, fps


def rgb2yiqImage(image):
    yiq_transform = np.array([[0.299,   0.587,      0.114],
                              [0.5959,  -0.2746,    -0.3213],
                              [0.2115,  -0.5227,    0.3112]],
                              dtype=np.float32)
    imshape = image.shape
    image = image.reshape(-1,3)
    image = image @ yiq_transform.T
    image = image.reshape(imshape)
    return image

def rgb2yiqVideo(video, backward=False, forceCPU=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if forceCPU:
        device = torch.device('cpu')

    if not backward:
        yiq_transform = torch.tensor([[0.299,   0.587,      0.114],
                                      [0.5959,  -0.2746,    -0.3213],
                                      [0.2115,  -0.5227,    0.3112]],
                                      dtype=torch.float32).to(device)      
    else:
        yiq_transform = torch.tensor([[1.0, 0.956,  0.619],
                                      [1.0, -0.272, -0.647],
                                      [1.0, -1.106, 1.703]],
                                      dtype=torch.float32).to(device)

    video = torch.from_numpy(video).float().to(device)
    vshape = video.shape
    video = video.reshape(-1, 3)
    video = video @ yiq_transform.T
    video = video.reshape(vshape)
    if backward:
        video = torch.clamp(video, 0, 255).int()

    result = video.cpu().numpy()
    del video, yiq_transform
    torch.cuda.empty_cache()
    return result

def separate_ychannel(video):
    ychannel = video[:,:,:,0]
    iqchannels = video[:,:,:,1:]
    return ychannel, iqchannels


def create_video(video_tensor, output_filename, fps):
    height, width, channels = video_tensor[0].shape
    size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # *'MP4V' for mp4
    out = cv2.VideoWriter(output_filename, fourcc, fps, size)

    for img in video_tensor:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_bgr = img_bgr.astype(np.uint8)
        if img_bgr.shape == (height, width, channels) and img_bgr.dtype == np.uint8:
            out.write(img_bgr)

    out.release()
    print(f"Video saved as {output_filename}")

