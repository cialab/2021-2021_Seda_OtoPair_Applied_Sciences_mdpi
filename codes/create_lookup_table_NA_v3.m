%%
%For new pairs I added all new image to train set and recreate lookup
%table 
%%
addpath('/isilon/datalake/cialab/scratch/cialab/scamalan/Autoscope');
train_location = '/isilon/datalake/cialab/scratch/cialab/scamalan/Autoscope/Pair/ENT_lookup_table_analysis/Normal_abnormal_v3/Train/';
validation_location = '/isilon/datalake/cialab/scratch/cialab/scamalan/Autoscope/Pair/ENT_lookup_table_analysis/Normal_abnormal_v3/Validation/';
test_location = '/isilon/datalake/cialab/scratch/cialab/scamalan/Autoscope/Pair/ENT_lookup_table_analysis/Normal_abnormal/Test1/';
destination = '/isilon/datalake/cialab/scratch/cialab/scamalan/Autoscope/Pair/ENT_lookup_table_analysis/Normal_abnormal_v4/Net_list_8_12_20/Heatmap/';

imdsTrain = imageDatastore(train_location,'LabelSource','foldernames','IncludeSubfolders',true);
imdsValidation =  imageDatastore(validation_location,'LabelSource','foldernames','IncludeSubfolders',true);
imdsTest = imageDatastore(test_location,'LabelSource','foldernames','IncludeSubfolders',true);

numTrainImages = numel(imdsTrain.Labels);
            
            net = inceptionresnetv2;
            % analyzeNetwork(net)

            lgraph = layerGraph(net);
            % figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
            % plot(lgraph)
            inputSize = net.Layers(1).InputSize
            lgraph = removeLayers(lgraph, {'predictions','predictions_softmax','ClassificationLayer_predictions'});

            numClasses = numel(categories(imdsTrain.Labels));
            newLayers = [
            fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
            softmaxLayer('Name','softmax')
            classificationLayer('Name','classoutput')];
            lgraph = addLayers(lgraph,newLayers);

            lgraph = connectLayers(lgraph,'avg_pool','fc');
            layers = lgraph.Layers;
            connections = lgraph.Connections;

            layers(1:820) = freezeWeights(layers(1:820));
            lgraph = createLgraphUsingConnections(layers,connections);

            pixelRange = [-30 30];
            imageAugmenter = imageDataAugmenter( ...
                'RandXReflection',true, ...
                'RandYReflection',true, ...
                'RandRotation',[0 360],...
                'RandScale',[0.7 2], ...
                'RandXShear', [0 45], ...
                'RandXShear', [0 45],...
                'RandXTranslation',pixelRange, ...
                'RandYTranslation',pixelRange);
            augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
                'DataAugmentation',imageAugmenter);
            augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

            options = trainingOptions('sgdm', ...
                'MiniBatchSize',12, ...
                'MaxEpochs',30, ...
                'InitialLearnRate',3e-4, ...%     'Shuffle','every-epoch', ...
                'ValidationData',augimdsValidation, ...
                'ValidationFrequency',5, ...
                'ValidationPatience',10);%, ...%'Verbose',false, ...
%                 'Plots','training-progress');
            % lgraph = layerGraph(net);
            % figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
%             plot(lgraph)
            % netTransfer = trainNetwork(augimdsTrain,layers,options);
            net = trainNetwork(augimdsTrain,lgraph,options);
            
            inputSize = net.Layers(1).InputSize
            augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
            augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);
           
            netTransfer = net;
            layer = 'avg_pool';%'conv_7b_ac';%'block17_2_ac';%'mixed_7a';%'block17_20_ac';                    %% should be change according to the inceptionresnetv2 net!!!!
            featuresTrain = activations(netTransfer,augimdsTrain,layer,'OutputAs','rows');
            featuresTest = activations(netTransfer,augimdsTest,layer,'OutputAs','rows');
            
            weights = net.Layers(822,1).Weights;
            wgt = weights';
            lookup = featuresTrain*wgt;
            lookup_test = featuresTest*wgt;
           
            match ={strcat(train_location,'/Normal/'),strcat(train_location,'/Abnormal/')};
            im_list=erase(imdsTrain.Files,match);
            labels=imdsTrain.Labels;
            
            test_match ={strcat(test_location,'/Normal/'),strcat(test_location,'/Abnormal/')};
            test_im_list=erase(imdsTest.Files,test_match);
            test_labels=imdsTest.Labels;

augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);
YPred = classify(net,augimdsTest);
YTest = imdsTest.Labels;
test_accuracy = sum(YPred == YTest)/numel(YTest)
 C_test=confusionmat(YPred,YTest)
 Test_img_pre_tru = [string(test_im_list) string(YTest) string(YPred)];
 
 augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
YPred = classify(net,augimdsTrain);
YTest = imdsTrain.Labels;
train_accuracy = sum(YPred == YTest)/numel(YTest)
 C_train=confusionmat(YPred,YTest)
 Train_img_pre_tru = [string(im_list) string(YTest) string(YPred)];
 
 netName ='ResNet_v2';
 
 match={strcat(train_location,'Normal/'),...
        strcat(train_location,'Abnormal/'),'.JPG','.jpg','.png'};
all_im_names = erase(imdsTrain.Files,match);

for im_num=1:length(imdsTrain.Files)
img     = imread(char(imdsTrain.Files(im_num)));
gt_label = imdsTrain.Labels(im_num);
im_name = all_im_names(im_num);

r_num=find(contains(Train_img_pre_tru(:,1),im_name))
if~(Train_img_pre_tru(r_num,2)==Train_img_pre_tru(r_num,3))
    img     = imresize(img,inputSize(1:2));
    imgSize = size(img);
    imgSize = imgSize(1:2);


% CAM
    layer   = 819;
    name    = net.Layers(layer).Name;
    classes = net.Layers(end).Classes;
    
    activationsM = activations(net,img,name);
    act1         = mat2gray(activationsM(:,:,1));
    
    scores = squeeze(mean(activationsM,[1 2]));
    
    if netName ~= "squeezenet"
        fcWeights = net.Layers(end-2).Weights;
        fcBias    = net.Layers(end-2).Bias;
        scores    =  fcWeights*scores + fcBias;
        
        [~,classIds] = maxk(scores,3);
        
        weightVector       = shiftdim(fcWeights(classIds(1),:),-1);
        classActivationMap = sum(activationsM.*weightVector,3);
    else
        [~,classIds] = maxk(scores,3);
        classActivationMap = activationsM(:,:,classIds(1));
    end
    
    % Calculate the top class labels and the final normalized class scores.
    scores    = exp(scores)/sum(exp(scores));     
    maxScores = scores(classIds);
    labels    = classes(classIds);
    
    fig=figure
    subplot(1,2,1)
    imshow(img)
    title('True Label:'+string(gt_label));
    
    subplot(1,2,2)
    outImg = CAMshow(img,classActivationMap);
%     if (maxScores(gt_label==labels)>=0.59)
        imwrite(outImg,strcat(destination,'Images/',char(im_name),'.jpg'));
%     end
    title(string(labels) + ", " + string(maxScores));
    saveas(fig,strcat(destination,'Figures/',string(im_name),'.fig'));
    saveas(fig,strcat(destination,'Figures/',string(im_name),'.png'));
    close all;
  end
end