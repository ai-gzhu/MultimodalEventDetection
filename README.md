pip install opencv-python

pip install opencv-contrib-python


pip install gensim

pip install --upgrade mmdnn

pip install --upgrade torch torchvision

pip install mxnet

python download_models.py

python -m mmdnn.conversion._script.convertToIR -f mxnet -n imagenet11k-resnet-152-symbol.json -w imagenet11k-resnet-152-0000.params -d resnet152 --inputShape 3,224,224

pip install numpy==1.16.1

python -m mmdnn.conversion._script.IRToCode -f pytorch --IRModelPath resnet152.pb --dstModelPath kit_imagenet.py --IRWeightPath resnet152.npy -dw kit_pytorch.npy  

python -m mmdnn.conversion.examples.pytorch.imagenet_test --dump resnet152Full.pth -n kit_imagenet.py -w kit_pytorch.npy 
