%DataDir = ['F:' filesep '2025-05-09'];
%DataDir = ['F:' filesep '2025-05-10'];
%DataDir = ['S:' filesep 'DataMEA1K_NL' filesep '2025-05-09'];
%DataDir = ['D:' filesep 'DataMEA1K_NL' filesep '2025-05-08'];
%DataDir = ['E:' filesep 'AndreasChipData' filesep '2025-05-08'];
% DataDir = ['E:' filesep 'AndreasChipData' filesep 'ToSimon20250514'];

DataDir = [filesep 'Volumes' filesep 'large' filesep 'BMI' filesep 'VirtualReality' filesep 'SpatialSequenceLearning' filesep 'RUN_rYL010' filesep 'rYL010_P1100' filesep '2025-05-08_15-51_rYL010_P1100_FreelyMoving_3min'];

%OutputFile = 'Rat_20250509';
%OutputFile = 'Bench_20250510';
OutputFile = 'ephys_output'; %virtual reality 3 min
GoodChannels = [149 204 305 373 440 520 684 888];

%FileExt = '.dat';
FileExt = '.loggerRaw';
Param.FPGALength = 2588;
Param.FrameLength = Param.FPGALength+44; %2632;
NFrames = 1536;    %in block
NCounts = 2*NFrames;    %in block
Param.BlockSize = Param.FrameLength*NFrames;
Param.LookUp = gray2bi_lookup; %generates lookup table
Param.sps = 20e3; %20 kHz

bs = matlab.io.datastore.BlockedFileSet(DataDir,...
        'FileExtensions', FileExt,...
        'BlockSize',Param.BlockSize);
disp(['Number of blocks: ' num2str(bs.NumBlocks)]); 
%100 blocks takes 3.141877 seconds.
t100 = 3.141877; %seconds
disp(['Estimated run time: ' num2str(round(bs.NumBlocks*t100/100/60)) ' min']);
%Estimated run time 48879/100*t100 = 
%NumBlocks_tmp = 100; %NumBlocks_tmp = bs.NumBlocks; %10 - for debugging
NumBlocks_tmp = bs.NumBlocks;
TotalFrames = NumBlocks_tmp*NFrames;
TotalSamples = TotalFrames*2;
FrameCounter = zeros(TotalFrames,1,'uint32');
Sensors = zeros(13,TotalFrames,'uint8');
SampleCounter = zeros(TotalSamples,1,'uint32');
fileID = fopen([DataDir filesep OutputFile '.uint16'],'w');
MEA1K_good = zeros(length(GoodChannels),2*TotalFrames,'uint16');

%convert and save MEA1K neural data in binary uint16 file 
tic
%for i_t = 0:(NumBlocks_tmp-1)
i_t = 0;
while hasNextBlock(bs)     
    [FrameCounter1, Sensors1, SampleCounter1, MEA1K1] = ReadBlock(nextblock(bs), Param);
    FrameCounter(i_t*NFrames+(1:NFrames)) = FrameCounter1;
    Sensors(:, i_t*NFrames+(1:NFrames)) = Sensors1;
    SampleCounter(i_t*NCounts+(1:NCounts)) = SampleCounter1;
    MEA1K_good(:,i_t*NCounts+(1:NCounts)) =  MEA1K1(1+GoodChannels,:);
    fwrite(fileID, MEA1K1,'uint16');
    if (mod(i_t+1,100) == 0); disp(num2str(i_t+1)); end
    i_t = i_t+1;
end
toc

fclose(fileID);

FrameCounter((i_t*NFrames+1):end) = [];
Sensors(:,(i_t*NFrames+1):end) = [];
SampleCounter((i_t*NCounts+1):end) = [];
MEA1K_good(:,(i_t*NCounts+1):end) = [];

%%
%save parameters and supplementary data to .h5 file
filename = [DataDir filesep OutputFile '.h5'];
if exist(filename, 'file')>0; delete(filename); end %delete file if it exists
ds = '/Param/FPGALength'; h5create(filename,ds,[1 1]); h5write(filename,ds,Param.FPGALength);
ds = '/Param/FrameLength'; h5create(filename,ds,[1 1]); h5write(filename,ds,Param.FrameLength);
ds = '/Param/BlockSize'; h5create(filename,ds,[1 1]); h5write(filename,ds,Param.BlockSize);
ds = '/Param/LookUp'; h5create(filename,ds,size(Param.LookUp),"Datatype","uint16"); 
                      h5write(filename,ds,Param.LookUp);
ds = '/Param/sps'; h5create(filename,ds,[1 1]); h5write(filename,ds,Param.sps);
ds = '/FrameCounter'; h5create(filename,ds,size(FrameCounter),"Datatype","uint32"); 
                      h5write(filename,ds,FrameCounter);
ds = '/Sensors'; h5create(filename,ds,size(Sensors),"Datatype","uint8"); 
                      h5write(filename,ds,Sensors);
ds = '/SampleCounter'; h5create(filename,ds,size(SampleCounter),"Datatype","uint32"); 
                      h5write(filename,ds,SampleCounter);
ds = '/MEA1K_good'; h5create(filename,ds,size(MEA1K_good),"Datatype","uint16"); 
                      h5write(filename,ds, MEA1K_good);
%%

%Now make plots
PlotCharts(DataDir, OutputFile, Param, FrameCounter, SampleCounter, Sensors, MEA1K1, MEA1K_good);

