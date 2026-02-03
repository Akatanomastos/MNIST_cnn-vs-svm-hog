clc;
clear;
close all;

data_train = readmatrix("mnist_train.csv");
images = data_train(:,2:end);
images = mat2gray(images);
labels = data_train(:,1);

data_test = readmatrix("mnist_test.csv");
images_test = data_test(:,2:end);
images_test = mat2gray(images_test);
labels_test = data_test(:,1);

num_classes = 10;

figure;

for class_label = 0:num_classes-1
    class_indices = find(labels == class_label);
    first_sample_index = class_indices(1);
    
    img = reshape(images(first_sample_index, :), [28, 28]);

    subplot(2, 5, class_label + 1);
    imshow(img');
    title(['Class ' num2str(class_label)]);
end

labels_categorical = categorical(labels, 0:num_classes-1);
labels_categorical_test = categorical(labels_test, 0:num_classes-1);

ratio = 0.66;
num_samples = size(images,1);
point_index = round(num_samples*ratio);

Xt = images(1:point_index,:);
X_train = reshape(Xt',[28, 28, 1, point_index]);
Y_train = labels_categorical(1:point_index);

test_samples = size(images_test,1);
X_test = reshape(images_test',[28, 28, 1, test_samples]);
Y_test = labels_categorical_test;

% Έλεγχος ότι το training set ανταποκρίνεται σωστά σε samples εκτός του set
% ώστε να ελεγχθεί η εγκυρότητας της εκπαίδευσης

Xv = images(point_index+1:end,:);
X_val = reshape(Xv',[28, 28, 1, num_samples-point_index]);
Y_val = labels_categorical(point_index+1:end);

layers = [
    imageInputLayer([28 28 1])
    convolution2dLayer(3,6,"Padding",0,"Stride",1)
    reluLayer

    batchNormalizationLayer
    averagePooling2dLayer(2,"Stride",2);
    convolution2dLayer(3,16,"Padding",0,"Stride",1)
    reluLayer

    batchNormalizationLayer
    averagePooling2dLayer(2,"Stride",2)

    fullyConnectedLayer(120)
    reluLayer
    fullyConnectedLayer(84)
    reluLayer
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
];


minibatch_size = 256; %64, 128, 192, 256
% learning rate= 0.01, 0.001, 0.0001

options = trainingOptions("sgdm","MaxEpochs",8,'InitialLearnRate',0.01, "Shuffle","every-epoch","MiniBatchSize",minibatch_size,"ValidationData",{X_val,Y_val},"Plots","training-progress");
net = trainNetwork(X_train, Y_train, layers, options);
    
pred_labels = classify(net, X_test);
test_accuracy = sum(pred_labels == Y_test) / test_samples;
fprintf('Test Accuracy with mini batch size (%d): %.4f\n', minibatch_size ,test_accuracy);
C = calculateConfusionMatrix(pred_labels, Y_test, num_classes);


function conf_matrix = calculateConfusionMatrix(pred_labels, true_labels, num_classes)
    conf_matrix = zeros(num_classes, num_classes);
    for i = 1:length(pred_labels)
        predicted_class = pred_labels(i);
        true_class = true_labels(i);
        conf_matrix(true_class, predicted_class) = conf_matrix(true_class, predicted_class) + 1;
    end
end
