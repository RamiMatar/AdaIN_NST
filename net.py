import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    '''This is the pretrained VGG encoder which we will use to calculate our perceptual features, which are used both for the AdaIN layer
    and the loss function. The encoder is the first 4 layers of the VGG network, with the rest omitted. The forward method returns a list of the intermediate
    outputs of the encoder, which represent the perceptual features of the input image. The output layers are chosen as the relu layers of the VGG network, based on the
    implementation in the AdaIN paper.
    Generally speaking, using different layers from the VGG to calculate the loss will lead to different types of style matching. The earlier layers will capture
    the low level features, while the later layers will capture the high level features.'''
    def __init__(self, model = 'VGG16', device = "cpu"):
        super().__init__()
        if model == 'VGG19':
            vgg = models.vgg19(pretrained=True).eval()
        else:
            vgg = models.vgg16(pretrained=True).eval()
        vgg = vgg.features
        vgg = vgg.to(device)
        self.layers = [vgg[0:2], vgg[2:7], vgg[7:12], vgg[12:21]]

        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, X):
        results = []
        for i in range(len(self.layers)):
            X = self.layers[i](X)
            results.append(X)
        return results
    

class Decoder(nn.Module):
    '''The decoder is similar structurally to the VGG encoder but uses upsampling and reflection padding to upsample the image to the desired output size. 
    Otherwise, it is a simple convolutionalal network with relus. The decoder is the part of the network we want to train, and it will take inputs from the encoder
    after passing through the AdaIN layer, and output the stylized image. 
    '''
    def __init__(self, device):
        super().__init__()
        self.decoder = nn.Sequential(
          nn.ReflectionPad2d((1, 1, 1, 1)),
          nn.Conv2d(512, 256, (3, 3)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='nearest'),
          nn.ReflectionPad2d((1, 1, 1, 1)),
          nn.Conv2d(256, 256, (3, 3)),
          nn.ReLU(),
          nn.ReflectionPad2d((1, 1, 1, 1)),
          nn.Conv2d(256, 256, (3, 3)),
          nn.ReLU(),
          nn.ReflectionPad2d((1, 1, 1, 1)),
          nn.Conv2d(256, 256, (3, 3)),
          nn.ReLU(),
          nn.ReflectionPad2d((1, 1, 1, 1)),
          nn.Conv2d(256, 128, (3, 3)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='nearest'),
          nn.ReflectionPad2d((1, 1, 1, 1)),
          nn.Conv2d(128, 128, (3, 3)),
          nn.ReLU(),
          nn.ReflectionPad2d((1, 1, 1, 1)),
          nn.Conv2d(128, 64, (3, 3)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='nearest'),
          nn.ReflectionPad2d((1, 1, 1, 1)),
          nn.Conv2d(64, 64, (3, 3)),
          nn.ReLU(),
          nn.ReflectionPad2d((1, 1, 1, 1)),
          nn.Conv2d(64, 3, (3, 3)),
        ).to(device)

    def forward(self, x):
        return self.decoder(x)
    
def AdaIN(content_features, style_features):
    batch_size, channels = content_features.shape[0:2]
    # first we compute the mean and standard deviation for each instance and channel for both the content and style features.
    content_std, content_mean = torch.std_mean(content_features, dim = (2,3))
    style_std, style_mean = torch.std_mean(style_features, dim = (2,3))

    # we reshape them to be broadcastable with the content features
    content_mean = content_mean.reshape(batch_size, channels, 1, 1)
    content_std = content_std.reshape(batch_size, channels, 1, 1)
    style_mean = style_mean.reshape(batch_size, channels, 1, 1)
    style_std = style_std.reshape(batch_size, channels, 1, 1)

    # we normalize the content features and then adapt them to the style features. small error term used for numerical stability.
    normalized_content_features = (content_features - content_mean) / (content_std + 1e-9)
    adapted_content_features = (normalized_content_features + (style_mean / (style_std + 1e-9))) * style_std
    
    return adapted_content_features

class Model(nn.Module):
    '''The model is a combination of the encoder, decoder, and the AdaIN layer. The forward method takes in the content and style images and returns the stylized image.
    First, both of the content image and style image are passed through the pretrained encoder to obtain the four output layers. Then, those get fed into the AdaIN layer
    which normalizes the content image to the statistics of the style image. Ulyanov, et al. found that Instance Norm is particularly effective for style transfer and that much
    of the style information is contained in the statistics of the perceptual features, and simply matching those can produce style transfer.
    The output of the AdaIN layer is then passed through the decoder network to produce the stylized image.
    The loss is computed by taking the mean squared error of the perceptual features of the content image after the AdaIN layer and the stylized output image. 
    The paper suggests it might be useful to also include second order statistics in the loss computation using correlation alignment, though I did not test that.'''

    def __init__(self, device, vgg_model = "VGG19", training = True, style_weight = 1):
        super().__init__()
        self.training = training
        self.encoder = Encoder(model = vgg_model, device = device)
        self.decoder = Decoder(device)
        self.loss = nn.MSELoss()
        self.style_weight = style_weight

    def forward(self, input, alpha = 1.0):
        content = input[0]
        style = input[1]
        content_loss = 0
        style_loss = 0
        # First, we pass both the content and style through the fixed encoder, we take the last result in the content features and all intermediate steps for the style features
        content_features = self.encoder(content)[-1]
        style_features = self.encoder(style)
        # After we have the features, we compute the Adaptive Instance Normalization module and do a forward pass in the decoder
        decoder_input = AdaIN(content_features, style_features[-1])
        decoder_input = alpha * decoder_input + (1 - alpha) * content_features
        decoded = self.decoder(decoder_input)
        stylized = decoded
        # After the forward pass, we have a predicted output image, which we feed through the fixed encoder to receive our features for the loss function, so we calculate the content and style loss
        if self.training:
            predicted_features = self.encoder(decoded)
            content_loss = self.loss(predicted_features[-1], decoder_input)
            for i in range(len(self.encoder.layers)):
                predicted_std_statistics, predicted_mean_statistics = torch.std_mean(predicted_features[i], dim = (2,3))
                target_std_statistics, target_mean_statistics = torch.std_mean(style_features[i], dim = (2,3))
                style_loss += self.loss(predicted_mean_statistics, target_mean_statistics) + self.loss(predicted_std_statistics, target_std_statistics)
        return content_loss, style_loss, stylized

