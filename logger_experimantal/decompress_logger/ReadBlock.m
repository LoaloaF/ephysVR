function [FrameCounter, Sensors, SampleCounter, MEA1K] = ReadBlock(blkInfo, Param)
% Outputs:%FrameCounter, Sensors, SampleCounter, MEA1K
%   Detailed explanation goes here
fr = matlab.io.datastore.DsFileReader(blkInfo.Filename);
seek(fr,blkInfo.Offset);
D = read(fr,Param.BlockSize);
D = reshape(D,Param.FrameLength,[]);

%framecounter: Elapsed time is 0.678631 seconds for 10 records.
Position = Param.FrameLength-14; %in both cases 4 bytes are added at the end(?)
FrameCounter8 = D(Position+(1:4),:);
FrameCounter = uint32(FrameCounter8(1,:)) + bitshift(uint32(FrameCounter8(2,:)),8)+...
           + bitshift(uint32(FrameCounter8(3,:)),16)+ bitshift(uint32(FrameCounter8(4,:)),24);

%sensors: the same time
Sensors = D(Param.FPGALength+(1:13),:);

%Prepare MEA1K data
MaxCount = 1032*2*5/4;
B = D(2+(1:MaxCount),:);
clear D;
%Reshape to have one time point in a column
B = reshape(B,MaxCount/2,[]);

%conversion to 10-bit representation
%reshape to five 8-bit values in a column first
B = reshape(B,5,[]);
D = zeros(4, size(B,2),'uint16');
D(1,:) = uint16(B(1,:))+bitshift(uint16(bitand(B(5,:),0x03)),8);
D(2,:) = uint16(B(2,:))+bitshift(uint16(bitand(B(5,:),bitshift(0x03,2))),6);
D(3,:) = uint16(B(3,:))+bitshift(uint16(bitand(B(5,:),bitshift(0x03,4))),4);
D(4,:) = uint16(B(4,:))+bitshift(uint16(bitand(B(5,:),bitshift(0x03,6))),2);
clear B;
%reshape back to 1032 raws
D = reshape(D,1032,[]);

%Take SampleCounter
SampleCounter2 = D(1024+(1:2),:);
SampleCounter = uint32(SampleCounter2(1,:)) + bitshift(uint32(SampleCounter2(2,:)),10);

%Take MEA1K data
D(1025:end,:) = [];
D = reshape(D,1,[]);
D = Param.LookUp(D+1); %D is array
MEA1K = reshape(D,1024,[]);
end

