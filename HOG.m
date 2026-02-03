clc;
clear;
close all;

data_train = readmatrix("mnist_train.csv");
data_test = readmatrix("mnist_test.csv");
images_test = data_test(:,2:end);
images = data_train(:,2:end);

num_images = size(images,1);
num_images_test = size(images_test,1);

shuffled_indices = randperm(num_images);
images = images(shuffled_indices, :);

shuffled_indices_test = randperm(num_images_test);
images_test = images_test(shuffled_indices_test, :);

n=1; % values n=1, 4, 9
hogfeatures_size = 36; % values hogfeatures_size=36, 324
hogfeatures = zeros(num_images,hogfeatures_size*n);
hogfeatures_test = zeros(num_images_test, hogfeatures_size*n);

cellsize = [12 12];
blocksize = [2 2];
numbins = 9;

for i=1:num_images
    reshaped_image = reshape(images(i,:), [28 28]);
    extract_hogFeature = extractHOGFeatures(reshaped_image,'CellSize',cellsize, "BlockSize",blocksize, 'NumBins',numbins);
    hogfeatures(i,:) = extract_hogFeature;
end

for i=1:num_images_test
    reshaped_image_test = reshape(images_test(i,:), [28 28]);
    extract_hogFeature_test = extractHOGFeatures(reshaped_image_test,'CellSize',cellsize, "BlockSize",blocksize, 'NumBins',numbins);
    hogfeatures_test(i,:) = extract_hogFeature_test;
end

ratio = 0.66;
point_index = round(num_images*ratio);
num_classes = 10;

labels_train = data_train(:,1);
labels_train = labels_train(shuffled_indices);
labels_train_categorical = categorical(labels_train, 0:num_classes-1);

X_train = hogfeatures(1:point_index,:); 
Y_train = labels_train_categorical(1:point_index);
X_val = hogfeatures(point_index+1:end,:);
Y_val = labels_train_categorical(point_index+1:end);

labels_test = data_test(:,1); 
labels_test = labels_test(shuffled_indices_test);

labels_test_categorical = categorical(labels_test, 0:num_classes-1);
Y_test = labels_test_categorical;
X_test = hogfeatures_test;

svmModel = fitcecoc(X_train, Y_train,'Learners','svm');

Y_val_pred = predict(svmModel, X_val);
accuracy_val = sum(Y_val_pred == Y_val) / numel(Y_val);
fprintf('Accuracy on the validation set: %.4f\n', accuracy_val);

pred_labels = predict(svmModel, X_test);
accuracy_test = sum(pred_labels == Y_test) / numel(Y_test);
fprintf('Accuracy on the test set: %.4f\n', accuracy_test);

C = calculateConfusionMatrix(pred_labels, Y_test, num_classes);

function conf_matrix = calculateConfusionMatrix(pred_labels, true_labels, num_classes)
    conf_matrix = zeros(num_classes, num_classes);
    for i = 1:length(pred_labels)
        predicted_class = pred_labels(i);
        true_class = true_labels(i);
        conf_matrix(true_class, predicted_class) = conf_matrix(true_class, predicted_class) + 1;
    end
end