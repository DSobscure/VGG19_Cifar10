import Cifar10Manager
import VGG19Model
import tensorflow as TensorFlow
import numpy as Numpy

def main(argv = None):
    tranningSet = Cifar10Manager.SetupCifar10TranningResources()
    testingImageSet, testingLabelSet = Cifar10Manager.SetupCifar10TestingResources()
    print('Setup Cifar10Manager Successful')
    x = TensorFlow.placeholder(TensorFlow.float32, [None, 32, 32, 3])
    output = TensorFlow.placeholder(TensorFlow.float32, [None, 10])
    keepProbe = TensorFlow.placeholder(TensorFlow.float32)

    network = VGG19Model.CreateNetwork(x, keepProbe, True)
    learningRate = TensorFlow.placeholder(TensorFlow.float32)
    crossEntropy = TensorFlow.reduce_mean(TensorFlow.nn.softmax_cross_entropy_with_logits(labels=output, logits=network))
    l2_loss = TensorFlow.add_n([TensorFlow.nn.l2_loss(v) for v in TensorFlow.trainable_variables()])
    cost = crossEntropy + 0.0001 * l2_loss
    trainStep = TensorFlow.train.MomentumOptimizer(learningRate, 0.9, use_nesterov = True).minimize(cost)

    correct_prediction = TensorFlow.equal(TensorFlow.argmax(network,1), TensorFlow.argmax(output,1))
    accuracy = TensorFlow.reduce_mean(TensorFlow.cast(correct_prediction, TensorFlow.float32))

    session = TensorFlow.InteractiveSession()
    session.run(TensorFlow.global_variables_initializer())
    saver = TensorFlow.train.Saver()
    lossSum = 0.

    def Testing(sess):
        acc = 0.0
        for i in range(1,11):
            image = Numpy.array(testingImageSet[0+(i-1)*1000:1000*i])
            label = Numpy.array(testingLabelSet[0+(i-1)*1000:1000*i])
            acc= acc + accuracy.eval(session=sess, feed_dict={x: image, output: label, keepProbe: 1.0})
        return acc/10

    for i in range(1,391*80+1): #1~80 epochs
        imageBatch, labelBatch = Cifar10Manager.TakeRandomTranningSampleBatch(tranningSet, 128)
        ## test every epochs
        if i%391 == 0:
            acc = Testing(session)
            print("step %d, Test accuracy %g loss %g"%(i,  acc, lossSum/391))
            with open('Models/record', 'a') as file:
                file.writelines(str(acc) + "\t" + str(lossSum/391) + "\n")
            lossSum = 0.
        ## trianing
        [_, loss] = session.run([trainStep, crossEntropy],feed_dict={x: imageBatch, output: labelBatch, keepProbe: 0.5, learningRate: 0.002})
        lossSum = lossSum + loss
        
    savePath = saver.save(session, "Models/model.ckpt")
    print("Model saved in file: %s" % savePath)


    for i in range(1,391*41+1): #81~121 epochs
        imageBatch, labelBatch = Cifar10Manager.TakeRandomTranningSampleBatch(tranningSet, 128)
        ## test every epochs
        if i%391 == 0:
            acc = Testing(session)
            print("step %d, Test accuracy %g loss %g"%(i,  acc, lossSum/391))
            with open('Models/record', 'a') as file:
                file.writelines(str(acc) + "\t" + str(lossSum/391) + "\n")
            lossSum = 0.
        ## trianing
        [_, loss] = session.run([trainStep, crossEntropy],feed_dict={x: imageBatch, output: labelBatch, keepProbe: 0.5, learningRate: 0.001})
        lossSum = lossSum + loss
    
    savePath = saver.save(session, "Models/model.ckpt")
    print("Model saved in file: %s" % savePath)

    for i in range(1,391*42): #122~164 epochs
        imageBatch, labelBatch = Cifar10Manager.TakeRandomTranningSampleBatch(tranningSet, 128)
        ## test every epochs
        if i%391 == 0:
            acc = Testing(session)
            print("step %d, Test accuracy %g loss %g"%(i,  acc, lossSum/391))
            with open('Models/record', 'a') as file:
                file.writelines(str(acc) + "\t" + str(lossSum/391) + "\n")
            lossSum = 0.
        ## trianing
        [_, loss] = session.run([trainStep, crossEntropy],feed_dict={x: imageBatch, output: labelBatch, keepProbe: 0.5, learningRate: 0.0001})
        lossSum = lossSum + loss

    savePath = saver.save(session, "Models/model.ckpt")
    print("Model saved in file: %s" % savePath)

    session.close()


if __name__ == '__main__':
    TensorFlow.app.run()