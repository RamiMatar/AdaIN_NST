import torch
import torch.nn as nn
import torchvision.models as models
class Encoder(nn.Module):
    def __init__(self, model = 'VGG16', device = "cpu"):
        super().__init__()
        if model == 'VGG19':
            vgg = models.vgg19(weights='DEFAULT').eval()
        else:
            vgg = models.vgg16(weights='DEFAULT').eval()
        vgg = vgg.features
        vgg = vgg.to(device)
        self.layers = [vgg[0:4], vgg[4:9], vgg[9:16], vgg[16:23]]

        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, X):
        results = []
        for i in range(len(self.layers)):
            X = self.layers[i](X)
            results.append(X)
        return results
    

class Decoder(nn.Module):
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
        self.decoder = self.decoder
    def forward(self, x):
        return self.decoder(x)
    
def AdaIN(content_features, style_features):
    # first we compute the mean and standard deviation for each instance and channel for both the content and style features.
    batch_size, channels = content_features.shape[0:2]
  
    content_std, content_mean = torch.std_mean(content_features, dim = (2,3))
    style_std, style_mean = torch.std_mean(style_features, dim = (2,3))
    content_mean = content_mean.reshape(batch_size, channels, 1, 1)
    content_std = content_std.reshape(batch_size, channels, 1, 1)
    style_mean = style_mean.reshape(batch_size, channels, 1, 1)
    style_std = style_std.reshape(batch_size, channels, 1, 1)
    normalized_content_features = (content_features - content_mean) / (content_std + 1e-9)
    adapted_content_features = (normalized_content_features + (style_mean / (style_std + 1e-9))) * style_std
    
    adapted_content_std, adapted_content_mean = torch.std_mean(adapted_content_features, dim = (2,3))
    return adapted_content_features

class Model(nn.Module):
    def __init__(self, device, vgg_model = "VGG19", style_weight = 1):
        super().__init__()
        self.encoder = Encoder(model = vgg_model, device = device)
        self.decoder = Decoder(device)
        self.loss = nn.MSELoss()
        self.style_weight = style_weight

    def forward(self, input, alpha = 1.0):
        content = input[0]
        style = input[1]
        loss = 0
        # First, we pass both the content and style through the fixed encoder, we take the last result in the content features and all intermediate steps for the style features
        content_features = self.encoder(content)[-1]
        style_features = self.encoder(style)
        # After we have the features, we compute the Adaptive Instance Normalization module and do a forward pass in the decoder
        decoder_input = AdaIN(content_features, style_features[-1])
        decoder_input = alpha * decoder_input + (1 - alpha) * content_features
        decoded = self.decoder(decoder_input)
        stylized = decoded
        # After the forward pass, we have a predicted output image, which we feed through the fixed encoder to receive our features for the loss function, so we calculate the content and style loss
        predicted_features = self.encoder(decoded)
        content_loss = self.loss(predicted_features[-1], decoder_input)
        style_loss = 0
        for i in range(len(self.encoder.layers)):
            predicted_std_statistics, predicted_mean_statistics = torch.std_mean(predicted_features[i], dim = (2,3))
            target_std_statistics, target_mean_statistics = torch.std_mean(style_features[i], dim = (2,3))
            style_loss += self.loss(predicted_mean_statistics, target_mean_statistics) + self.loss(predicted_std_statistics, target_std_statistics)
        return content_loss, style_loss, stylized

