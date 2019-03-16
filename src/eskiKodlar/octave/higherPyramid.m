1;

pkg load signal
pkg load image

inImage = imread("../../data/1600.ppm");
Gauss = fspecial('gaussian', 5);

downSampledImageRed   = downsample(conv2(inImage(:,:,1), Gauss), 2);
downSampledImageGreen = downsample(conv2(inImage(:,:,2), Gauss), 2);
downSampledImageBlue  = downsample(conv2(inImage(:,:,3), Gauss), 2);

downSampledImage = zeros(size(downSampledImageRed)(1), size(downSampledImageRed)(2), 3);

downSampledImage(:,:,1) = downSampledImageRed;
downSampledImage(:,:,2) = downSampledImageGreen;
downSampledImage(:,:,3) = downSampledImageBlue;

imwrite(downSampledImageRed, "../../output/octavePyramidUp.jpg");
