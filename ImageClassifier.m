%% Loading the datset
%IncludeSubfolders tells matlab to look for images in subdirectories
%LabelSource provides a source for image labels and foldernames tells
%matlab to use the name of the folders as image labels
ds = imageDatastore("animals","IncludeSubfolders",true, "LabelSource","foldernames");



%% getting labels
labels = unique(ds.Labels)


%% renaming the images to 1 to n '.jpg'
%iterating through each subfolders
for i = 1:numel(labels) %numel stands for number of elements
    labelFiles = ds.Files(ds.Labels == labels(i)) %iterating through each files in current label

    %iterating through files in the current subfolder
    for j = 1:numel(labelFiles)
        [filepath, name, ext] = fileparts(labelFiles{j});
        newFilename = fullfile(filepath, sprintf('%d.jpg',j));
        img = imread(labelFiles{j})

        % If the image is not already a JPG, convert and save it
        if ~strcmpi(ext, '.jpg') && ~strcmpi(ext, '.jpeg')
            imwrite(img, newFilename, 'jpg');
            delete(labelFiles{j});  % Delete the original file
        else
            % If it's already a JPG, just rename it
            movefile(labelFiles{j}, newFilename);
        end
    end
    
    fprintf('Processed files in subfolder: %s\n', string(labels(i)));
end

% Reload the datastore to reflect changes
ds = imageDatastore("animals", "IncludeSubfolders", true, "LabelSource", "foldernames");

fprintf('Processing complete. Total images: %d\n', numel(ds.Files));

%%


% counting samples in each class
sampleCounts = countEachLabel(ds)

% class with the most samples
maxSample = max(sampleCounts.Count)
minSample = min(sampleCounts.Count)



%% training, test and validation sets
[temp, testData] = splitEachLabel(ds,0.8,'randomized');

% further splitting tempSet into training and validaion sets
[trainingData, validationData] = splitEachLabel(temp,0.8,'randomized');

% counting labels in each data set
testCounts = countEachLabel(testData)
trainingCounts = countEachLabel(trainingData)
validationCounts = countEachLabel(validationData)

%augmenting data
augmenter = imageDataAugmenter('RandRotation', [-20, 20], ...
                               'RandXReflection', true, ...
                               'RandYReflection', true, ...
                               'RandXScale', [0.9, 1.1], ...
                               'RandYScale', [0.9, 1.1]);

resized_trainingData = augmentedImageDatastore(imageSize, trainingData, ...
    'DataAugmentation', ...
    augmenter);
%resizing images 
imageSize = [224 224];
resized_testData = augmentedImageDatastore(imageSize, testData);
resized_validationData = augmentedImageDatastore(imageSize, validationData);
    
%%  Network Architecture
numClasses = numel(categories(ds.Labels));

layers = [
    imageInputLayer([224 224 3])

    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
      
    convolution2dLayer(3, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 256, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(3, 256, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 512, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3, 512, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    globalAveragePooling2dLayer
    
    fullyConnectedLayer(512)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

%%  Training Parameters
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 10, ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', resized_validationData, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'gpu');
%%  Training Model
[net, info] = trainNetwork(resized_trainingData, layers, options);
save('trainedNetwork.mat','net','info');
save('trainingData.mat', 'trainingData', 'validationData', 'testData');
%%  Predicting Test Data
YPred = classify(net, resized_testData);
YTest = testData.Labels;

% Calculate various performance metrics
accuracy = mean(YPred == YTest);
confMat = confusionmat(YTest, YPred);
precision = diag(confMat) ./ sum(confMat, 1)';
recall = diag(confMat) ./ sum(confMat, 2);
f1Score = 2 * (precision .* recall) ./ (precision + recall);

% Display results
disp(['Test Accuracy: ', num2str(accuracy)]);
disp(['Mean Precision: ', num2str(mean(precision))]);
disp(['Mean Recall: ', num2str(mean(recall))]);
disp(['Mean F1 Score: ', num2str(mean(f1Score))]);

% Plot confusion matrix
confusionchart(YTest, YPred);




