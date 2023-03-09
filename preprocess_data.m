%% Process dataset into mat files %%
%{
1) Reading csv files 
2) Parse fields including sptail grid and maneuver labels
3) Using unique vehicle ids to spilit train(70%)/validation(10%)/test(20%)
4) Only reserve those data sample with at least 3s history and 5s future
5) Save the dataset with a fixed 8Veh targets
Optional: filter on-ramp and off-ramp part or not. (Our result is obtained without filtering lane)
%}


%% Hyperparameters:
%{
30ms for history traj自车看3s
50ms for future traj预测5s
%}




clear;
clc;

%% Inputs:
% Locations of raw input files:sdff
us101_1 = 'raw/us101-0750-0805.txt';
us101_2 = 'raw/us101-0805-0820.txt';
us101_3 = 'raw/us101-0820-0835.txt';
i80_1 = 'raw/i80-1600-1615.txt';
i80_2 = 'raw/i80-1700-1715.txt';
i80_3 = 'raw/i80-1715-1730.txt';


%% Fields: 
%重点看这里！！！！！！,没用速度v 加速度a，车宽v_width、车长v_length,只用了xy时间，laneID（算roadfeature）
%{ 
1: Dataset Id
2: Vehicle Id
3: Frame Number
4: Local X
5: Local Y
6: Lane Id
7: Lateral maneuver
8: Longitudinal maneuver
9-47: Neighbor Car Ids at grid location %39格(13*3)
%}



%% Load data and add dataset id
disp('Loading data...')
%只是加了数据集id
traj{1} = load(us101_1);    
traj{1} = single([ones(size(traj{1},1),1),traj{1}]);
traj{2} = load(us101_2);
traj{2} = single([2*ones(size(traj{2},1),1),traj{2}]);
traj{3} = load(us101_3);
traj{3} = single([3*ones(size(traj{3},1),1),traj{3}]);
traj{4} = load(i80_1);    
traj{4} = single([4*ones(size(traj{4},1),1),traj{4}]);
traj{5} = load(i80_2);
traj{5} = single([5*ones(size(traj{5},1),1),traj{5}]);
traj{6} = load(i80_3);
traj{6} = single([6*ones(size(traj{6},1),1),traj{6}]);

%原txt里: 1:dataset id, 2:Vehicle id, 3:Frame index, 6:Local X, 7:Local Y, 15:Lane id.
%过滤所有驶入匝道和驶出匝道的车辆
for k = 1:6
    traj{k} = traj{k}(:,[1,2,3,6,7,15]);
    if k <=3 %: US101 make all lane id >= 6 to 6.
        traj{k}(traj{k}(:,6)>=6,6) = 6;
    end
end

vehTrajs{1} = containers.Map;
vehTrajs{2} = containers.Map;
vehTrajs{3} = containers.Map;
vehTrajs{4} = containers.Map;
vehTrajs{5} = containers.Map;
vehTrajs{6} = containers.Map;

vehTimes{1} = containers.Map;
vehTimes{2} = containers.Map;
vehTimes{3} = containers.Map;
vehTimes{4} = containers.Map;
vehTimes{5} = containers.Map;
vehTimes{6} = containers.Map;

%% Parse fields (listed above):
disp('Parsing fields...')

    for ii = 1:6 //traj的每个dataset中创建空的一列存vehID
    vehIds = unique(traj{ii}(:,2));
    for v = 1:length(vehIds) //把vehID都填进traj去
        vehTrajs{ii}(int2str(vehIds(v))) = traj{ii}(traj{ii}(:,2) == vehIds(v),:);
    end
    
    timeFrames = unique(traj{ii}(:,3)); %traj中创建空的一列存vehID
    for v = 1:length(timeFrames)%把frameID都填进traj去
        vehTimes{ii}(int2str(timeFrames(v))) = traj{ii}(traj{ii}(:,3) == timeFrames(v),:);
    end
    

    for k = 1:length(traj{ii}(:,1))        
        time = traj{ii}(k,3);
        dsId = traj{ii}(k,1);
        vehId = traj{ii}(k,2);
        vehtraj = vehTrajs{ii}(int2str(vehId));
        ind = find(vehtraj(:,3)==time);
        ind = ind(1);
        lane = traj{ii}(k,6);
        
        
        % Get lateral maneuver:
        // 接下来比对将来、现在、过去车辆所在的车道线位置，即6th column的lane id，判断车辆左转还是右转，在7th column的预留位打标签。
// ind代表当前帧时间，ub代表40帧以后的时间也就是将来时间，lb代表40帧之前也就是过去时间。
        ub = min(size(vehtraj,1),ind+40);
        lb = max(1, ind-40);
        if vehtraj(ub,6)>vehtraj(ind,6) || vehtraj(ind,6)>vehtraj(lb,6) //prepare to turn 或 stablize after turn
            traj{ii}(k,7) = 3; //右转
        elseif vehtraj(ub,6)<vehtraj(ind,6) || vehtraj(ind,6)<vehtraj(lb,6)
            traj{ii}(k,7) = 2;
        else
            traj{ii}(k,7) = 1;
        end
        
        
        % Get longitudinal maneuver:
        // 比将来、现在、过去的行驶速度，
        //即delta Y(根据5th column的Y坐标计算)/delta t，判断车辆是正常行驶还是减速刹车，在8th column的预留位打标签。
        // ind代表当前帧时间，ub代表50帧以后的时间也就是将来时间，lb代表30帧之前也就是过去时间。
        ub = min(size(vehtraj,1),ind+50);
        lb = max(1, ind-30);
        if ub==ind || lb ==ind
            traj{ii}(k,8) =1;
        else
            vHist = (vehtraj(ind,5)-vehtraj(lb,5))/(ind-lb);
            vFut = (vehtraj(ub,5)-vehtraj(ind,5))/(ub-ind);
            if vFut/vHist <0.8 //<0.8就是减速
                traj{ii}(k,8) =2; //brake
            else
                traj{ii}(k,8) =1;
            end
        end
        
        
        % Get grid locations:
        // 以ego车辆作为中心以0.5xgrid_lengthxcell_length为长以0.5xgrid_widthxcell_width为宽画长方形，
        //判断其他车辆是否在长方形内，若是则成为邻域车辆。
        //精确计算出邻域车辆的网格位置(grid location)并将二维index展成一维index called exactGridLocation，（）
        //根据exactGridLocation在9th column及以后的预留位记录邻域车辆的id。9-47列

        t = vehTimes{ii}(int2str(time)); //同一个dataset的时间
        frameEgo = t(t(:,6) == lane,:); //当前车道的所有帧
        frameL = t(t(:,6) == lane-1,:);//左车道的所有帧
        frameR = t(t(:,6) == lane+1,:);//右车道的所有帧
        if ~isempty(frameL)
            for l = 1:size(frameL,1)
                y = frameL(l,5)-traj{ii}(k,5); 
                if abs(y) <90 //abs：所以看+-90(前后6格)
                    gridInd = 1+round((y+90)/15); //从左
                    traj{ii}(k,8+gridInd) = frameL(l,2); //存vehID
                end
            end
        end
        for l = 1:size(frameEgo,1)
            y = frameEgo(l,5)-traj{ii}(k,5); //从14开始
            if abs(y) <90 && y~=0
                gridInd = 14+round((y+90)/15);
                traj{ii}(k,8+gridInd) = frameEgo(l,2);
            end
        end
        if ~isempty(frameR)
            for l = 1:size(frameR,1)
                y = frameR(l,5)-traj{ii}(k,5);
                if abs(y) <90
                    gridInd = 27+round((y+90)/15);
                    traj{ii}(k,8+gridInd) = frameR(l,2);
                end
            end
        end
        
    end
end


%% Split train, validation, test
disp('Splitting into train, validation and test sets...')

trajAll = [traj{1};traj{2};traj{3};traj{4};traj{5};traj{6}];
clear traj;

trajTr = [];
trajVal = [];
trajTs = [];
for k = 1:6
    ul1 = round(0.7*max(trajAll(trajAll(:,1)==k,2)));
    ul2 = round(0.8*max(trajAll(trajAll(:,1)==k,2)));
    //train 0.7 valid 0.1 test 0.2 ;每个dataset按这个比例分然后汇总到train,test,val里
    trajTr = [trajTr;trajAll(trajAll(:,1)==k & trajAll(:,2)<=ul1, :)];
    trajVal = [trajVal;trajAll(trajAll(:,1)==k & trajAll(:,2)>ul1 & trajAll(:,2)<=ul2, :)];
    trajTs = [trajTs;trajAll(trajAll(:,1)==k & trajAll(:,2)>ul2, :)];
end

tracksTr = {};
for k = 1:6
    trajSet = trajTr(trajTr(:,1)==k,:);
    carIds = unique(trajSet(:,2)); //遍历每个dataset里的不重复的每辆车
    for l = 1:length(carIds) //车的长度，多少辆车
        vehtrack = trajSet(trajSet(:,2) ==carIds(l),3:5); //只取了frameid,local_x,local_y
        tracksTr{k,carIds(l)} = vehtrack;
    end
end

tracksVal = {};
for k = 1:6
    trajSet = trajVal(trajVal(:,1)==k,:);
    carIds = unique(trajSet(:,2));
    for l = 1:length(carIds)
        vehtrack = trajSet(trajSet(:,2) ==carIds(l),3:5)';
        tracksVal{k,carIds(l)} = vehtrack;
    end
end

tracksTs = {};
for k = 1:6
    trajSet = trajTs(trajTs(:,1)==k,:);
    carIds = unique(trajSet(:,2));
    for l = 1:length(carIds)
        vehtrack = trajSet(trajSet(:,2) ==carIds(l),3:5)';
        tracksTs{k,carIds(l)} = vehtrack;
    end
end


%% Filter edge cases: 边缘数据扔掉
% Since the model uses 3 sec of trajectory history for prediction, the initial 3 seconds of each trajectory is not used for training/testing

disp('Filtering edge cases...')

indsTr = zeros(size(trajTr,1),1);
for k = 1: size(trajTr,1)
    t = trajTr(k,3);
    if tracksTr{trajTr(k,1),trajTr(k,2)}(1,31) <= t && tracksTr{trajTr(k,1),trajTr(k,2)}(1,end)>t+1
        indsTr(k) = 1;
    end
end
trajTr = trajTr(find(indsTr),:);


indsVal = zeros(size(trajVal,1),1);
for k = 1: size(trajVal,1)
    t = trajVal(k,3);
    if tracksVal{trajVal(k,1),trajVal(k,2)}(1,31) <= t && tracksVal{trajVal(k,1),trajVal(k,2)}(1,end)>t+1
        indsVal(k) = 1;
    end
end
trajVal = trajVal(find(indsVal),:);


indsTs = zeros(size(trajTs,1),1);
for k = 1: size(trajTs,1)
    t = trajTs(k,3);
    if tracksTs{trajTs(k,1),trajTs(k,2)}(1,31) <= t && tracksTs{trajTs(k,1),trajTs(k,2)}(1,end)>t+1
        indsTs(k) = 1;
    end
end
trajTs = trajTs(find(indsTs),:);

%% Save mat files:
disp('Saving mat files...')

traj = trajTr;
tracks = tracksTr;
save('TrainSet','traj','tracks');

traj = trajVal;
tracks = tracksVal;
save('ValSet','traj','tracks');

traj = trajTs;
tracks = tracksTs;
save('TestSet','traj','tracks');

//traj  : n*(13+grid_cells), n is the data number.
// tracks: 6*maxVehicleId, each cell is specified for (datasetId, vehicleId), with size (11+grid_cells)*totalFramNum.









