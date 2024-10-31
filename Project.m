%% Noises

clear ; clc ; close all ;

originalImage = imread('david-villa.jpg');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Impulse noise

figure;
subplot(1, 2, 1);
imshow(originalImage);
title('Original Image');

noisyImage1 = originalImage;

noiseIntensity = 0.10; 
[numRows, numCols, ~] = size(originalImage);

numPixelsToChange = round(noiseIntensity * numRows * numCols);
pixelIndices = randperm(numRows * numCols, numPixelsToChange);

% Set half of the randomly chosen pixels to black and half to white
noisyImage1(pixelIndices(1:numPixelsToChange/2)) = 0; 
noisyImage1(pixelIndices(numPixelsToChange/2+1:end)) = 255; 

subplot(1, 2, 2);
imshow(noisyImage1);
title('Impulse Noise');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Additive noise

figure;
subplot(1, 2, 1);
imshow(originalImage);
title('Original Image');

noisyImage2 = double(originalImage);

noiseIntensity = 15; 

noise = noiseIntensity * randn(size(originalImage));
noisyImage2 = noisyImage2 + noise;
noisyImage2 = uint8(max(0, min(255, noisyImage2)));

subplot(1, 2, 2);
imshow(noisyImage2);
title('Additive Noise');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Salt-and-Pepper noise

figure;
subplot(1, 2, 1);
imshow(originalImage);
title('Original Image');

noisyImage3 = originalImage;

noiseIntensity = 0.05; 
saltProbability = 0.5; 

% Generate random matrix with the same size as the image
randomValues = rand(size(originalImage));
% Set pixels to black where randomValues are less than noiseIntensity/2
noisyImage3(randomValues < noiseIntensity/2) = 0;
% Set pixels to white where randomValues are between noiseIntensity/2 and noiseIntensity
noisyImage3(randomValues >= noiseIntensity/2 & randomValues < noiseIntensity) = 255;

subplot(1, 2, 2);
imshow(noisyImage3);
title('Salt-and-Pepper Noise');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Shot noise

figure;
subplot(1, 2, 1);
imshow(originalImage);
title('Original Image');

noisyImage4 = double(originalImage);

noiseIntensity = 30; 

% Generate shot noise based on Poisson distribution
shotNoise = poissrnd(noiseIntensity, size(originalImage));

noisyImage4 = noisyImage4 + shotNoise;
noisyImage4 = uint8(max(0, min(255, noisyImage4)));

subplot(1, 2, 2);
imshow(noisyImage4);
title('Shot Noise');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Speckle noise (Multiplicative noise)

figure;
subplot(1, 2, 1);
imshow(originalImage);
title('Original Image');

noisyImage5 = double(originalImage);

speckleIntensity = 0.15; 

% Generate speckle noise based on a Gaussian distribution
speckleNoise = 1 + speckleIntensity * randn(size(originalImage));
% Multiply the speckle noise with the image
noisyImage5 = noisyImage5 .* speckleNoise;
noisyImage5 = uint8(max(0, min(255, noisyImage5)));

subplot(1, 2, 2);
imshow(noisyImage5);
title('Speckle Noise (Gaussian Distribution)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Uniform Additive noise

figure;
subplot(1, 2, 1);
imshow(originalImage);
title('Original Image');

noisyImage6 = double(originalImage);

noiseIntensity = 20; 

% Generate uniform additive noise
uniformAdditiveNoise = noiseIntensity * (rand(size(originalImage)) - 0.5) * 2;

noisyImage6 = noisyImage6 + uniformAdditiveNoise;
noisyImage6 = uint8(max(0, min(255, noisyImage6)));

subplot(1, 2, 2);
imshow(noisyImage6);
title('Uniform Additive Noise');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Gaussian noise

figure;
subplot(1, 2, 1);
imshow(originalImage);
title('Original Image');

noisyImage7 = double(originalImage);

noiseIntensity = 20; 

% Generate Gaussian noise
gaussianNoise = noiseIntensity * randn(size(originalImage));

noisyImage7 = noisyImage7 + gaussianNoise;
noisyImage7 = uint8(max(0, min(255, noisyImage7)));

subplot(1, 2, 2);
imshow(noisyImage7);
title('Gaussian Noise');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Poisson noise

figure;
subplot(1, 2, 1);
imshow(originalImage);
title('Original Image');

noisyImage8 = double(originalImage);

noiseIntensity = 40;

% Generate Poisson noise
poissonNoise = poissrnd(noiseIntensity, size(originalImage));

noisyImage8 = noisyImage8 + poissonNoise;
noisyImage8 = uint8(max(0, min(255, noisyImage8)));

subplot(1, 2, 2);
imshow(noisyImage8);
title('Poisson Noise');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Periodic noise

figure;
subplot(1, 2, 1);
imshow(originalImage);
title('Original Image');

noisyImage9 = double(originalImage);

frequency = 0.05; 
amplitude = 3;   

% Create a meshgrid to generate the sinusoidal pattern
[x, y] = meshgrid(1:size(originalImage, 2), 1:size(originalImage, 1));
periodicNoise = amplitude * sin(2 * pi * frequency * x);


noisyImage9 = noisyImage9 + periodicNoise;
noisyImage9 = uint8(max(0, min(255, noisyImage9)));

subplot(1, 2, 2);
imshow(noisyImage9);
title('Periodic Noise');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Filters

%% Applying Filters for picture with Impulse noise

figure;
subplot(1, 2, 1);
imshow(noisyImage1);
title('Noisy Image(Impulse noise)');

% Apply Linear Smoothing Filter (3x3 kernel)
linearFiltered = applyLinearFilter(noisyImage1, ones(3) / 9);
subplot(1, 2, 2);
imshow(linearFiltered);
title('Linear Smoothing Filter');

figure;
subplot(1, 2, 1);
imshow(noisyImage1);
title('Noisy Image(Impulse noise)');


% Apply Wiener Filter
wienerFiltered = applyWienerFilter(noisyImage1, [5, 5]);
subplot(1, 2, 2);
imshow(wienerFiltered);
title('Wiener Filter');

figure;
subplot(1, 2, 1);
imshow(noisyImage1);
title('Noisy Image(Impulse noise)');


% Apply Median Filter (3x3 kernel)
medianFiltered = applyMedianFilter(noisyImage1, [3, 3]);
subplot(1, 2, 2);
imshow(medianFiltered);
title('Median Filter');

figure;
subplot(1, 2, 1);
imshow(noisyImage1);
title('Noisy Image(Impulse noise)');


% Apply Gaussian Filter (3x3 kernel, sigma = 1)
gaussianFiltered = applyGaussianFilter(noisyImage1, 3, 1);
subplot(1, 2, 2);
imshow(gaussianFiltered);
title('Gaussian Filter');

%% Applying Filters for picture with Additive noise

figure;
subplot(1, 2, 1);
imshow(noisyImage2);
title('Noisy Image(Additive noise)');

% Apply Linear Smoothing Filter (3x3 kernel)
linearFiltered = applyLinearFilter(noisyImage2, ones(3) / 9);
subplot(1, 2, 2);
imshow(linearFiltered);
title('Linear Smoothing Filter');

figure;
subplot(1, 2, 1);
imshow(noisyImage2);
title('Noisy Image(Additive noise)');


% Apply Wiener Filter
wienerFiltered = applyWienerFilter(noisyImage2, [5, 5]);
subplot(1, 2, 2);
imshow(wienerFiltered);
title('Wiener Filter');

figure;
subplot(1, 2, 1);
imshow(noisyImage2);
title('Noisy Image(Additive noise)');


% Apply Median Filter (3x3 kernel)
medianFiltered = applyMedianFilter(noisyImage2, [3, 3]);
subplot(1, 2, 2);
imshow(medianFiltered);
title('Median Filter');

figure;
subplot(1, 2, 1);
imshow(noisyImage2);
title('Noisy Image(Additive noise)');


% Apply Gaussian Filter (3x3 kernel, sigma = 1)
gaussianFiltered = applyGaussianFilter(noisyImage2, 3, 1);
subplot(1, 2, 2);
imshow(gaussianFiltered);
title('Gaussian Filter');

%% Applying Filters for picture with Salt-and-Pepper noise

figure;
subplot(1, 2, 1);
imshow(noisyImage3);
title('Noisy Image(Salt-and-Pepper noise)');

% Apply Linear Smoothing Filter (3x3 kernel)
linearFiltered = applyLinearFilter(noisyImage3, ones(3) / 9);
subplot(1, 2, 2);
imshow(linearFiltered);
title('Linear Smoothing Filter');

figure;
subplot(1, 2, 1);
imshow(noisyImage3);
title('Noisy Image(Salt-and-Pepper noise)');


% Apply Wiener Filter
wienerFiltered = applyWienerFilter(noisyImage3, [5, 5]);
subplot(1, 2, 2);
imshow(wienerFiltered);
title('Wiener Filter');

figure;
subplot(1, 2, 1);
imshow(noisyImage3);
title('Noisy Image(Salt-and-Pepper noise)');


% Apply Median Filter (3x3 kernel)
medianFiltered = applyMedianFilter(noisyImage3, [3, 3]);
subplot(1, 2, 2);
imshow(medianFiltered);
title('Median Filter');

figure;
subplot(1, 2, 1);
imshow(noisyImage3);
title('Noisy Image(Salt-and-Pepper noise)');


% Apply Gaussian Filter (3x3 kernel, sigma = 1)
gaussianFiltered = applyGaussianFilter(noisyImage3, 3, 1);
subplot(1, 2, 2);
imshow(gaussianFiltered);
title('Gaussian Filter');

%% Applying Filters for picture with Shot noise

figure;
subplot(1, 2, 1);
imshow(noisyImage4);
title('Noisy Image(Shot noise)');

% Apply Linear Smoothing Filter (3x3 kernel)
linearFiltered = applyLinearFilter(noisyImage4, ones(3) / 9);
subplot(1, 2, 2);
imshow(linearFiltered);
title('Linear Smoothing Filter');

figure;
subplot(1, 2, 1);
imshow(noisyImage4);
title('Noisy Image(Shot noise)');


% Apply Wiener Filter
wienerFiltered = applyWienerFilter(noisyImage4, [5, 5]);
subplot(1, 2, 2);
imshow(wienerFiltered);
title('Wiener Filter');

figure;
subplot(1, 2, 1);
imshow(noisyImage4);
title('Noisy Image(Shot noise)');


% Apply Median Filter (3x3 kernel)
medianFiltered = applyMedianFilter(noisyImage4, [3, 3]);
subplot(1, 2, 2);
imshow(medianFiltered);
title('Median Filter');

figure;
subplot(1, 2, 1);
imshow(noisyImage4);
title('Noisy Image(Shot noise)');


% Apply Gaussian Filter (3x3 kernel, sigma = 1)
gaussianFiltered = applyGaussianFilter(noisyImage4, 3, 1);
subplot(1, 2, 2);
imshow(gaussianFiltered);
title('Gaussian Filter');

%% Applying Filters for picture with Speckle noise(Multiplicative noise)

figure;
subplot(1, 2, 1);
imshow(noisyImage5);
title('Noisy Image(Speckle noise)');

% Apply Linear Smoothing Filter (3x3 kernel)
linearFiltered = applyLinearFilter(noisyImage5, ones(3) / 9);
subplot(1, 2, 2);
imshow(linearFiltered);
title('Linear Smoothing Filter');

figure;
subplot(1, 2, 1);
imshow(noisyImage5);
title('Noisy Image(Speckle noise)');


% Apply Wiener Filter
wienerFiltered = applyWienerFilter(noisyImage5, [5, 5]);
subplot(1, 2, 2);
imshow(wienerFiltered);
title('Wiener Filter');

figure;
subplot(1, 2, 1);
imshow(noisyImage5);
title('Noisy Image(Speckle noise)');


% Apply Median Filter (3x3 kernel)
medianFiltered = applyMedianFilter(noisyImage5, [3, 3]);
subplot(1, 2, 2);
imshow(medianFiltered);
title('Median Filter');

figure;
subplot(1, 2, 1);
imshow(noisyImage5);
title('Noisy Image(Speckle noise)');


% Apply Gaussian Filter (3x3 kernel, sigma = 1)
gaussianFiltered = applyGaussianFilter(noisyImage5, 3, 1);
subplot(1, 2, 2);
imshow(gaussianFiltered);
title('Gaussian Filter');

%% Applying Filters for picture with Uniform Additive Noise;

subplot(1, 2, 1);
imshow(noisyImage6);
title('Noisy Image(Uniform Additive noise)');

% Apply Linear Smoothing Filter (3x3 kernel)
linearFiltered = applyLinearFilter(noisyImage6, ones(3) / 9);
subplot(1, 2, 2);
imshow(linearFiltered);
title('Linear Smoothing Filter');

figure;
subplot(1, 2, 1);
imshow(noisyImage6);
title('Noisy Image(Uniform Additive noise)');


% Apply Wiener Filter
wienerFiltered = applyWienerFilter(noisyImage6, [5, 5]);
subplot(1, 2, 2);
imshow(wienerFiltered);
title('Wiener Filter');

figure;
subplot(1, 2, 1);
imshow(noisyImage6);
title('Noisy Image(Uniform Additive noise)');


% Apply Median Filter (3x3 kernel)
medianFiltered = applyMedianFilter(noisyImage6, [3, 3]);
subplot(1, 2, 2);
imshow(medianFiltered);
title('Median Filter');

figure;
subplot(1, 2, 1);
imshow(noisyImage6);
title('Noisy Image(Uniform Additive noise)');


% Apply Gaussian Filter (3x3 kernel, sigma = 1)
gaussianFiltered = applyGaussianFilter(noisyImage6, 3, 1);
subplot(1, 2, 2);
imshow(gaussianFiltered);
title('Gaussian Filter'); 

%% Applying Filters for picture with Gaussian noise

subplot(1, 2, 1);
imshow(noisyImage7);
title('Noisy Image(Gaussian noise)');

% Apply Linear Smoothing Filter (3x3 kernel)
linearFiltered = applyLinearFilter(noisyImage7, ones(3) / 9);
subplot(1, 2, 2);
imshow(linearFiltered);
title('Linear Smoothing Filter');

figure;
subplot(1, 2, 1);
imshow(noisyImage7);
title('Noisy Image(Gaussian noise)');


% Apply Wiener Filter
wienerFiltered = applyWienerFilter(noisyImage7, [5, 5]);
subplot(1, 2, 2);
imshow(wienerFiltered);
title('Wiener Filter');

figure;
subplot(1, 2, 1);
imshow(noisyImage7);
title('Noisy Image(Gaussian noise)');


% Apply Median Filter (3x3 kernel)
medianFiltered = applyMedianFilter(noisyImage7, [3, 3]);
subplot(1, 2, 2);
imshow(medianFiltered);
title('Median Filter');

figure;
subplot(1, 2, 1);
imshow(noisyImage7);
title('Noisy Image(Gaussian noise)');


% Apply Gaussian Filter (3x3 kernel, sigma = 1)
gaussianFiltered = applyGaussianFilter(noisyImage7, 3, 1);
subplot(1, 2, 2);
imshow(gaussianFiltered);
title('Gaussian Filter'); 

%% Applying Filters for picture with Poisson noise

subplot(1, 2, 1);
imshow(noisyImage8);
title('Noisy Image(Poisson noise)');

% Apply Linear Smoothing Filter (3x3 kernel)
linearFiltered = applyLinearFilter(noisyImage8, ones(3) / 9);
subplot(1, 2, 2);
imshow(linearFiltered);
title('Linear Smoothing Filter');

figure;
subplot(1, 2, 1);
imshow(noisyImage8);
title('Noisy Image(Poisson noise)');


% Apply Wiener Filter
wienerFiltered = applyWienerFilter(noisyImage8, [5, 5]);
subplot(1, 2, 2);
imshow(wienerFiltered);
title('Wiener Filter');

figure;
subplot(1, 2, 1);
imshow(noisyImage8);
title('Noisy Image(Poisson noise)');


% Apply Median Filter (3x3 kernel)
medianFiltered = applyMedianFilter(noisyImage8, [3, 3]);
subplot(1, 2, 2);
imshow(medianFiltered);
title('Median Filter');

figure;
subplot(1, 2, 1);
imshow(noisyImage8);
title('Noisy Image(Poisson noise)');


% Apply Gaussian Filter (3x3 kernel, sigma = 1)
gaussianFiltered = applyGaussianFilter(noisyImage8, 3, 1);
subplot(1, 2, 2);
imshow(gaussianFiltered);
title('Gaussian Filter'); 

%% Applying Filters for picture with Periodic noise

subplot(1, 2, 1);
imshow(noisyImage9);
title('Noisy Image(Periodic noise)');

% Apply Linear Smoothing Filter (3x3 kernel)
linearFiltered = applyLinearFilter(noisyImage9, ones(3) / 9);
subplot(1, 2, 2);
imshow(linearFiltered);
title('Linear Smoothing Filter');

figure;
subplot(1, 2, 1);
imshow(noisyImage9);
title('Noisy Image(Periodic noise)');


% Apply Wiener Filter
wienerFiltered = applyWienerFilter(noisyImage9, [5, 5]);
subplot(1, 2, 2);
imshow(wienerFiltered);
title('Wiener Filter');

figure;
subplot(1, 2, 1);
imshow(noisyImage9);
title('Noisy Image(Periodic noise)');


% Apply Median Filter (3x3 kernel)
medianFiltered = applyMedianFilter(noisyImage9, [3, 3]);
subplot(1, 2, 2);
imshow(medianFiltered);
title('Median Filter');

figure;
subplot(1, 2, 1);
imshow(noisyImage9);
title('Noisy Image(Periodic noise)');


% Apply Gaussian Filter (3x3 kernel, sigma = 1)
gaussianFiltered = applyGaussianFilter(noisyImage9, 3, 1);
subplot(1, 2, 2);
imshow(gaussianFiltered);
title('Gaussian Filter'); 


%% Filter Functions

function result = applyLinearFilter(image, kernel)
    [rows, cols, ~] = size(image);
    [kRows, kCols] = size(kernel);
    padRows = floor(kRows / 2);
    padCols = floor(kCols / 2);

    paddedImage = padarray(image, [padRows, padCols], 'replicate', 'both');
    
    result = zeros(size(image), 'like', image);

    for i = 1:rows
        for j = 1:cols
            window = double(paddedImage(i:i+kRows-1, j:j+kCols-1, :)); 
            result(i, j, :) = sum(sum(window .* kernel, 1), 2);
        end
    end

    result = uint8(result);
end

function result = applyWienerFilter(image, windowSize)
    [rows, cols, ~] = size(image);
    result = zeros(size(image), 'like', image);

    for i = 1:rows
        for j = 1:cols
            window = double(getNeighborhood(image, i, j, windowSize)); 
            localVariance = var(window(:));
            localMean = mean(window(:));
            noiseVariance = localVariance / 5; 
            alpha = noiseVariance / (noiseVariance + localVariance);
            result(i, j, :) = localMean + alpha * (image(i, j, :) - localMean);
        end
    end

    result = uint8(result);
end

function result = applyMedianFilter(image, windowSize)
    [rows, cols, ~] = size(image);
    result = zeros(size(image));

    for i = 1:rows
        for j = 1:cols
            window = getNeighborhood(image, i, j, windowSize);
            result(i, j, :) = median(window(:));
        end
    end

    result = uint8(result);
end

function result = applyGaussianFilter(image, kernelSize, sigma)
    kernel = fspecial('gaussian', [kernelSize, kernelSize], sigma);
    result = applyLinearFilter(image, kernel);
end

function window = getNeighborhood(image, i, j, windowSize)
    [rows, cols, ~] = size(image);
    halfSize = floor(windowSize / 2);

    iStart = max(i - halfSize, 1);
    iEnd = min(i + halfSize, rows);

    jStart = max(j - halfSize, 1);
    jEnd = min(j + halfSize, cols);

    window = image(iStart:iEnd, jStart:jEnd, :);
end
