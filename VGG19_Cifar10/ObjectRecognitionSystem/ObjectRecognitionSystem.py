import VGG19Model
import tensorflow as TensorFlow
import skimage
import skimage.io
import skimage.transform
import numpy as Numpy

def LoadImage(path):
    # load image
    image = skimage.io.imread(path)
    image = image / 255.0
    assert (0 <= image).all() and (image <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(image.shape[:2])
    yy = int((image.shape[0] - short_edge) / 2)
    xx = int((image.shape[1] - short_edge) / 2)
    crop_img = image[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img

def LoadSynset(file_path):
    return [l.strip() for l in open(file_path).readlines()]

# returns the top1 string
def PrintProb(prob, synset):
    # print prob
    pred = Numpy.argsort(prob)[::-1]

    # Get top5 label
    print("Top5: ")
    for i in range(5):
        print((synset[pred[i]], prob[pred[i]]))

def main():
    vgg = VGG19Model.VGG19Model();
    x = TensorFlow.placeholder(TensorFlow.float32, [None, 224, 224, 3]);
    output = TensorFlow.placeholder(TensorFlow.float32, [None, 1000]);
    vgg.Build(x);
    synset = LoadSynset("synset.txt");

    session = TensorFlow.InteractiveSession()

    while True:    # infinite loop
        imagePath = input("\nPlease input image path(input 'exit' to close): ")
        if imagePath == "exit":
            break  # stops the loop
        else:
            PrintProb(vgg.prob.eval(feed_dict={x : [LoadImage(imagePath)]})[0], synset)
    session.close();

if __name__ == "__main__":
    main()