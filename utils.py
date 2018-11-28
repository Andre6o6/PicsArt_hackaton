import numpy as np
import pandas as pd
import tqdm


EPS = 1e-10

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    from: https://www.kaggle.com/kmader/baseline-u-net-model-part-1
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(320, 240)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    from: https://www.kaggle.com/kmader/baseline-u-net-model-part-1
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape).T

def write_results(net, test_data_loader, path_images, name='submission.csv', threshold=0.25):
    
    predictions = []

    net.eval()
    for batch in tqdm.tqdm(test_data_loader):
        batch['img'] = batch['img'].to(device)
        with torch.no_grad():
            output = net2.forward(batch['img'])
        for i in range(output.shape[0]):
            img = output[i].detach().cpu().numpy()
            post_img = remove_small_holes(remove_small_objects(img > threshold))
            rle = rle_encode(post_img)
            predictions.append(rle)  
    
    df = pd.DataFrame.from_dict({'image': path_images, 'rle_mask': predictions})
    df.to_csv(name, index=False)