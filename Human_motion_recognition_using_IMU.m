clc;
clear;
close all;

% =======================
% 1. Load and Preprocess Data
% =======================
dataFile = 'UCI_HAR_Sample.csv';
data = readtable(dataFile);

% Extract features and labels
Xraw = data{:, {'accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ'}};  % [N x 6]
Yraw = categorical(data{:, 'Label'});  % [N x 1]

% Normalize features
mu = mean(Xraw);
sigma = std(Xraw);
Xnorm = (Xraw - mu) ./ sigma;

% =======================
% 2. Create Overlapping Sequences
% =======================
sequenceLength = 100;  % Larger window for better temporal context
stride = 2;            % Smaller stride for more samples
numSamples = floor((size(Xnorm, 1) - sequenceLength) / stride) + 1;

Xseq = cell(1, numSamples);
Yseq = categorical();

for i = 1:numSamples
    startIdx = (i - 1) * stride + 1;
    endIdx = startIdx + sequenceLength - 1;

    Xseq{i} = reshape(Xnorm(startIdx:endIdx, :)', [6, sequenceLength, 1]);

    % Majority label in the window
    windowLabels = Yraw(startIdx:endIdx);
    Yseq(i, 1) = mode(windowLabels);
end

% =======================
% 3. Train-Test Split
% =======================
cv = cvpartition(length(Xseq), 'HoldOut', 0.2);
XTrain = Xseq(~cv.test);
YTrain = Yseq(~cv.test);
XTest  = Xseq(cv.test);
YTest  = Yseq(cv.test);

% Convert to 4D arrays
XTrainMat = cat(4, XTrain{:});  % [6 x seqLength x 1 x N]
XTestMat = cat(4, XTest{:});

% =======================
% 4. Define CNN Architecture (Deeper & Better)
% =======================
inputSize = [6, sequenceLength, 1];
numClasses = numel(categories(Yseq));

layers = [
    imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none')

    convolution2dLayer([3 3], 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')

    maxPooling2dLayer([2 2], 'Stride', 2, 'Name', 'pool1')

    convolution2dLayer([3 3], 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')

    maxPooling2dLayer([2 2], 'Stride', 2, 'Name', 'pool2')

    convolution2dLayer([3 3], 128, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')

    dropoutLayer(0.4, 'Name', 'dropout')

    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu4')
    fullyConnectedLayer(numClasses, 'Name', 'fc2')

    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% =======================
% 5. Training Options
% =======================
options = trainingOptions('adam', ...
    'MaxEpochs', 400, ...
    'InitialLearnRate', 0.0005, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 100, ...
    'LearnRateDropFactor', 0.2, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'L2Regularization', 0.0001, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% =======================
% 6. Train the Model
% =======================
net = trainNetwork(XTrainMat, YTrain, layers, options);

% =======================
% 7. Evaluate Accuracy
% =======================
YPred = classify(net, XTestMat);
accuracy = sum(YPred == YTest) / numel(YTest);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

% =======================
% 8. Confusion Matrix
% =======================
figure;
confusionchart(YTest, YPred);
title('Confusion Matrix - CNN HAR Model');