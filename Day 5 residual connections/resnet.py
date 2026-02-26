import tensorflow as tf 
from tensorflow.keras.layers import  BatchNormalization , Add , Input  , ReLU , Conv2D
from tensorflow.keras.models import Model
def demo_resnet(X  ):
    #First Conv Layer
    conv2D = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(X)
    b1 = BatchNormalization()(conv2D)
    relu1 = ReLU()(b1)
    con2D2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(relu1)
    b2 = BatchNormalization()(con2D2)
    # Match channels using 1x1 Conv
    shortcut = Conv2D(64, (1,1), padding='same')(X)

    # Residual Add
    #Adding the input to the output of the second batch normalization layer 
    # Which result in residual connection or resnet block
    skip = Add()([b2,  shortcut])
    #There is a size mismatch 
    #skip = Add()([b2 , X] )
    relu2 = ReLU()(skip)
    return relu2

input_shape = (32, 32, 3)
input = Input (shape=input_shape)
output = demo_resnet(input)
model = Model(inputs=input, outputs=output)
model.summary()