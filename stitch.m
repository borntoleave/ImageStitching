function [ output_args ] = stitch( image1,image2 )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
corner_threshold=0.005;
enlarge_factor=2;
dist_threshold=0.3;
err_threshold=2;
inlier_threshold=0.45;
inlier_min=7;
f=im2single(imresize(imread(image1),1/2));
g=im2single(imresize(imread(image2),1/2));
size(f)
f_redch=f(:,:,1);
%imshow(redch)
%corner detection
[f_row,f_col]=find(cornerDetection(f_redch,2,corner_threshold,3)==1);
%scatter(col,row)
f_circles=[f_col,f_row,4*ones(size(f_col,1),1)];
Df=find_sift(f_redch,f_circles,enlarge_factor);%descriptor of f
g_redch=g(:,:,1);
[g_row,g_col]=find(cornerDetection(g_redch,2,corner_threshold,3)==1);
g_circles=[g_col,g_row,4*ones(size(g_col,1),1)];
Dg=find_sift(g_redch,g_circles,enlarge_factor);%descriptor of g

size(Df)%number of descriptors of f
size(Dg)%number of descriptors of g
dist_mat=zeros(size(Df,1),size(Dg,1));%distance matrix
[irange,jrange]=size(dist_mat);
for i = 1:irange
    for j =1:jrange
       dist_mat(i,j)= norm(Df(i,:)-Dg(j,:));%Euclidean distance
    end
end
imagesc(dist_mat)
pause(1);
hist(dist_mat(:))
pause(1);
[f_idx,g_idx]=find(dist_mat<dist_threshold);%pick out similar points
cfx=f_col(f_idx);
cfy=f_row(f_idx);
cgx=g_col(g_idx);
cgy=g_row(g_idx);
symbol = 'o';    % matlab
[fRows, fCols, ch] = size(f);
[gRows, gCols, ch] = size(g);
% Show the selected descriptors over the images
figure(1); clf;
imshow([f g]); axis image; hold on
plot(cfx,cfy, ['r',symbol]);   	% centers of key points in f
hold on
plot(cgx+fCols, cgy, ['r',symbol]);	% Centers of key points in g
pause()
size(cfx,1)
inliers=1:size(cfx,1);
while(1)
    A=[];inlierN=0;new_inliers=[];
    sample_idx=randperm(size(inliers,2),4);%random permuation of all the pairs
    for s=inliers(sample_idx)
        A=[A;
            0 0 0 cfx(s) cfy(s) 1 -cgy(s)*cfx(s) -cgy(s)*cfy(s) -cgy(s);
            cfx(s) cfy(s) 1 0 0 0 -cgx(s)*cfx(s) -cgx(s)*cfy(s) -cgx(s)];
    end
    %solve Ah=0
    [U,S,V]=svd(A);
    h = V(:,end);
    h = reshape(h,3,3)';
    for s=inliers
       c_tran=h*[cfx(s) cfy(s) 1]';
       c_tran=c_tran/c_tran(3);
       err=norm(c_tran-[cgx(s) cgy(s) 1]');
       if(err<err_threshold)
           inlierN=inlierN+1;
           new_inliers=[new_inliers s];
       end
    end
    %new_inliers=new_inliers';
    inlierN/size(inliers,2)
    figure(1); clf;
        imshow([f g]); axis image; hold on
        plot(cfx(new_inliers),cfy(new_inliers), ['r',symbol]);   	% centers of key points in f
        hold on
        plot(cgx(new_inliers)+fCols, cgy(new_inliers), ['r',symbol]);	% Centers of key points in g
        line([cfx(new_inliers)';cgx(new_inliers)'+fCols ],[cfy(new_inliers)';  cgy(new_inliers)'],'Color','g')
       % pause(0.05)
    
    if(inlierN>=inlier_min)
        break
    end
    
end
pause()

%put images together
x1=[];x2=[];y1=[];y2=[];
for s=inliers(sample_idx)
    x1=[x1;cfx(s)];
    y1=[y1;cfy(s)];
    x2=[x2;cgx(s)];
    y2=[y2;cgy(s)];
end
T=maketform('projective',[x2 y2],[x1 y1])
T.tdata.T           %find the transform
[im2t,xdataim2t,ydataim2t]=imtransform(g,T);
xdataout=[min(1,xdataim2t(1)) max(size(f,2),xdataim2t(2))];
ydataout=[min(1,ydataim2t(1)) max(size(f,1),ydataim2t(2))];
im2t=imtransform(g,T,'XData',xdataout,'YData',ydataout);
im1t=imtransform(f,maketform('affine',eye(3)),'XData',xdataout,'YData',ydataout);
ims=im1t/2+im2t/2;
figure, imshow(ims) %naive averaging
pause()
ims=max(im1t,im2t); %maximum merging
imshow(ims);
end





