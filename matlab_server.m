function matlab_server()
    % Create output directory if it doesn't exist
    if ~exist('outputs', 'dir')
        mkdir('outputs');
    end
    
    % Create TCP/IP server
    disp('==========================================================');
    disp('  MATLAB Image Filter Server');
    disp('==========================================================');
    disp('  Starting server on port 5050...');
    
    try
        % Create TCP server
        if exist('tcpserver', 'file')
            server = tcpserver('0.0.0.0', 5050);
            configureTerminator(server, "LF");
            disp('  ✓ Server started successfully (using tcpserver)');
        else
            server = tcpip('0.0.0.0', 5050, 'NetworkRole', 'server');
            set(server, 'Terminator', 'LF');
            fopen(server);
            disp('  ✓ Server started successfully (using tcpip)');
        end
        
        disp('  ✓ Waiting for connections...');
        
        while true
            if exist('tcpserver', 'file')
                if server.NumBytesAvailable > 0
                    message = readline(server);
                    message = char(message);
                    disp(['[RECEIVED] ', message]);
                    try
                        output_path = processFilterRequest(message);
                        writeline(server, output_path);
                        disp(['[RESPONSE] ', output_path]);
                    catch err
                        disp(['[ERROR] ', err.message]);
                        writeline(server, ['ERROR: ', err.message]);
                    end
                end
            else
                if server.BytesAvailable > 0
                    message = fscanf(server, '%s');
                    disp(['[RECEIVED] ', message]);
                    try
                        output_path = processFilterRequest(message);
                        fprintf(server, '%s\n', output_path);
                        disp(['[RESPONSE] ', output_path]);
                    catch err
                        disp(['[ERROR] ', err.message]);
                        fprintf(server, '%s\n', ['ERROR: ', err.message]);
                    end
                end
            end
            pause(0.1);
        end
        
    catch err
        disp(['[FATAL ERROR] ', err.message]);
        if exist('server', 'var'), clear server; end
    end
end

function output_path = processFilterRequest(message)
    parts = strsplit(message, '|');
    
    if length(parts) < 2
        error('Invalid message format. Expected: image_path|filter_name|params');
    end
    
    image_path = strtrim(parts{1});
    filter_name = lower(strtrim(parts{2}));
    
    if ~exist(image_path, 'file')
        error(['Image file not found: ', image_path]);
    end

    % if isa(result, 'double') && max(result(:)) <= 1.01 && min(result(:)) >= -0.01 
    % % If result is a normalized double, imwrite handles it.
    %     imwrite(result, output_path); 
    % else
    %     % Otherwise, assume it's a standard uint8/integer image.
    %     imwrite(result, output_path);
    % end
    
    img = imread(image_path);
    timestamp = round(now * 100000);
    output_path = sprintf('outputs/output_%d.png', timestamp);
    
    disp(['[FILTER] ', filter_name]);
    
    switch filter_name
        case 'rgb2gray'
            if length(parts) >= 3, option = str2double(parts{3}); else, option = 1; end
            if size(img, 3) == 3
                if option == 1
                    result = rgb2gray(img);
                else
                    result = uint8(0.2989 * double(img(:,:,1)) + 0.5870 * double(img(:,:,2)) + 0.1140 * double(img(:,:,3)));
                end
            else
                result = img;
            end
            
        case 'erosion'
            result = erosion(img);
            
        case 'dilation'
            result = dilation(img);

        case 'saltnpep'
            if length(parts) >= 3, th = str2double(parts{3}); else, th = 0.5; end
            result = addSaltAndPepperNoise(img, th)
            
        case 'blur'
            h = fspecial('gaussian', [15 15], 2);
            result = imfilter(img, h);
            
        case 'sharpen'
            result = imsharpen(img);
            
        case 'edge_detection'
            if size(img, 3) == 3, img = rgb2gray(img); end
            result = edge(img, 'Canny');
            result = uint8(result) * 255;
        
        case 'brightness'
            if length(parts) >= 4
                k = str2double(parts{3}); theta = str2double(parts{4});
            else
                k = 1; theta = 50;
            end
            result = brightness(img, k, theta)
            
        case 'vertical_edge_detection'
            % FIXED: Calls the function with exactly ONE argument
            result = applyVerticalEdgeDetection(img);
            result = uint8(result * 255); 
            
        case 'histogram'
            result = generateHistogram(img);

        case 'rgb2binary'
            if length(parts) >= 3, th = str2double(parts{3}); else, th = 0.5; end
            result = rgbtobinary(img, th);
        
        case 'contrast_stretching'
            result = contrastStretching(img)
            
        case 'gamma_noise'
            if length(parts) >= 4
                k = str2double(parts{3}); theta = str2double(parts{4});
            else
                k = 5; theta = 10;
            end
            result = gamma_noise(img, k, theta);
        
        case 'histogram_eq'
            result = histogramEqualization(img);
        
        case 'negative'
            result = negative(img);
        
        case 'fourier_transform'
            result = FourierTransform(img);
        
        case 'inv_fourier_transform'
            result = InverseFourierTransform(img);
        
        case 'brightnessnew'
        if length(parts) >= 3
            th = str2double(parts{3});
        else
            th = 10;
        end
        result = brightnessNew(img, th);
        
        case 'graytobinary'
            if length(parts) >= 3, th = str2double(parts{3}); else, th = 128; end
            if size(img, 3) == 3, gray = rgb2gray(img); else, gray = img; end
            result = graytobinary(gray, th);
            result = uint8(result) * 255;
        
        case 'correlation'
            if length(parts) >= 3, th = str2double(parts{3}); else, th = 128; end
            result = correlation(img, th);
        
        case 'mean_filter'
            if length(parts) >= 3, th = str2double(parts{3}); else, th = 128; end
            result = mean_filter(img, th);
        
        case 'weight_filter'
            if length(parts) >= 3, th = str2double(parts{3}); else, th = 128; end
            result = weightfilter(img, th);
        
        case 'point_detection'
            result = pointdetection(img);
        
        case 'point_sharpening'
            if length(parts) >= 4
                x = str2double(parts{3}); y = str2double(parts{4});
            else
                x = 5; y = 10;
            end
            result = PointSharpening(img, x, y);
        
        case 'min_filter'
            if length(parts) >= 3, x = str2double(parts{3}); else, x = 128; end
            result = min_filter(img, x);
        
        case 'median_filter'
            if length(parts) >= 3, x = str2double(parts{3}); else, x = 128; end
            result = median_filter(img, x);
        
        case 'butter_worth_low_pass'
            if length(parts) >= 4
                x = str2double(parts{3}); y = str2double(parts{4});
            else
                x = 5; y = 10;
            end
            result = ButterworthLowPassFilter(img, x, y);

        case 'butter_worth_high_pass'
            if length(parts) >= 4
                x = str2double(parts{3}); y = str2double(parts{4});
            else
                x = 5; y = 10;
            end
            result = ButterworthHighPassFilter(img, x, y);
        
        case 'ideal_low_pass'
            if length(parts) >= 3, x = str2double(parts{3}); else, x = 128; end
            result = IdealLowPassFilter(img, x);

        case 'ideal_high_pass'
            if length(parts) >= 3, x = str2double(parts{3}); else, x = 128; end
            result = IdealHighPassFilter(img, x);

        case 'gaussian_low_pass_filter'
            if length(parts) >= 3, x = str2double(parts{3}); else, x = 128; end
            result = GaussianLowPassFilter(img, x);
        
        case 'gaussian_high_pass_filter'
            if length(parts) >= 3, x = str2double(parts{3}); else, x = 128; end
            result = GaussianHighPassFilter(img, x);
        
        case 'gaussian_noise'
            if length(parts) >= 5
                x = str2double(parts{3}); y = str2double(parts{4}); z = str2double(parts{5});
            else
                x = 5; y = 10; z = 10
            end
            result = addGaussianNoise(img, x, y, z);

        case 'exponential_noise'
            if length(parts) >= 3, x = str2double(parts{3}); else, x = 128; end
            result = exponential_noise(img, x);

        case 'rayleigh_noise'
            if length(parts) >= 4
                x = str2double(parts{3}); y = str2double(parts{4});
            else
                x = 5; y = 10;
            end
            result = rayleigh_noise(img, x, y);
        
        case 'open_img'
            result = opening(img);
        
        case 'close_img'
            result = closing(img);
        
        case 'log'
            if length(parts) >= 3, x = str2double(parts{3}); else, x = 5; end
            result = Log(img, x);
        
        case 'max_filter'
            if length(parts) >= 3, x = str2double(parts{3}); else, x = 5; end
            result = max_filter(img, x)

        case 'gamma_corr'
            if length(parts) >= 3, x = str2double(parts{3}); else, x = 5; end
            result = GammaCorrection(img, x);

        case 'uniform_noise'
            if length(parts) >= 4
                x = str2double(parts{3}); y = str2double(parts{4});
            else
                x = 5; y = 10;
            end
            result = addUniformNoise(img, [x,y]);
        
        case 'sobel'
            if length(parts) >= 5
                x = str2double(parts{3}); y = str2double(parts{4}); z = str2double(parts{5});
            else
                x = 5; y = 10; z = 10
            end
            result = linesharplingSobel(img, x, y, z)
            
        otherwise
            error(['Filter not known: ', filter_name]);
    end
    
    imwrite(result, output_path);
end

function output = linesharplingSobel(oldImg,t,u,n)
oldImg=double(oldImg);
[H ,W, L]=size(oldImg);

if(t==1)
%     sobel mask
    if(u==1)
%         horizontal
filter=[-1 -2 -1;0 0 0;1 2 1];
    elseif(u==2)
%         vertical
filter=[-1 0 1;-2 0 2;-1 0 1];
    elseif(u==3)
        filter=[2 1 0;1 0 -1; 0 -1 -2];
%         left diagonal
    else
%         right diagonal 
filter=[0 -1 -2;1 0 -1;2 1 0];
    end
%   rober mask
else
%     robert mask
    if(u==1)
%         horizontal
filter=[0 1 0;0 0 0;0 -1 0];
    elseif(u==2)
%         vertical
filter=[0 0 0;1 0 -1 ;0 0 0];
    elseif(u==3)
%         left diagonal
        filter=[1 0 0 ;0 0 0;0 0 -1];

    else
%         right diagonal 
filter=[0 0 1;0 0 0;-1 0 0];
    end
end
k=1;
if(L==3)
    if(n==1)
        oldImg=padarray(oldImg,[k,k]);
        [H ,W, L]=size(oldImg);
        newImg=zeros(H, W);
    for i=1+k:H-k
        for j=1+k:W-k
            for y=1:3
            subImg(:,:,y)=(oldImg(i-k:i+k,j-k:j+k,y));
                end
            result=zeros(1,3);
            for f=1:3
                for g=1:3
                      for y=1:3
                    result(y)=subImg(f,g,y)*filter(f,g)+result(y);
                    end
                end
            end
            for y=1:3
            newImg(i,j,y)=result(y);
                end
            
        end
    end
    elseif(n==2)
        [H ,W, L]=size(oldImg);
        newImg=zeros(H, W);
        for i=1+k:H-k
            for j=1+k:W-k
                for y=1:3
             subImg(:,:,y)=(oldImg(i-k:i+k,j-k:j+k,y));
            end
            result=zeros(1,3);
            for f=1:3
                for g=1:3
                    for y=1:3
                    result(y)=subImg(f,g,y)*filter(f,g)+result(y);
                    end
                end
            end
               for y=1:3
            newImg(i,j,y)=result(y);
                end
 
            end
        end
        else
        filterImg=zeros(H+2*k,W+2*k);
        [H W L]=size(filterImg);
        for i=1:k
                for y=1:3
            filterImg(i,k+1:W-k,y)=oldImg(1,:,y);
            end
        end
     for i=1:k
                for y=1:3
            filterImg(H-k+i,k+1:W-k,y)=oldImg(end,:,y);
            end
     end
     for i=1:k
            for y=1:3
            filterImg(1:H,i,y)=filterImg(:,1,y);
            end
     end
         for i=1:k
            for y=1:3
            filterImg(1:H,W-k+i,y)=filterImg(:,end,y);
            end
         end
            for y=1:3
         filterImg(1+k:H-k,1+k:W-k,y)=oldImg(1:H,1:W,y);
        end
        for i=1+k:H-k
            for j=1+k:W-k
                for y=1:3
              subImg(:,:,y)=(filterImg(i-k:i+k,j-k:j+k,y));
            end
            result1=zeros(1,3);
            for f=1:3
                for g=1:3
                      for y=1:3
                    result(y)=subImg(f,g,y)*filter(f,g)+result(y);
                   end
                end
            end
                  for y=1:3
            newImg(i,j,y)=result(y);
                end
            end
        end
    end
%     GRAY
else
     if(n==1)
        oldImg=padarray(oldImg,[k,k]);
        [H ,W, L]=size(oldImg);
        newImg=zeros(H, W);
    for i=1+k:H-k
        for j=1+k:W-k
             subImg=(oldImg(i-k:i+k,j-k:j+k));
             result1=0;
            for f=1:3
                for g=1:3
                    result1=subImg(f,g)*filter(f,g)+result1;
                end
            end
            newImg(i,j)=result1;
            
        end
    end
    elseif(n==2)
        [H ,W, L]=size(oldImg);
        newImg=zeros(H, W);
        for i=1+k:H-k
            for j=1+k:W-k
             subImg=(oldImg(i-k:i+k,j-k:j+k));
              result1=0;
            for f=1:3
                for g=1:3
                    result1=subImg(f,g)*filter(f,g)+result1;
                end
            end
            newImg(i,j)=result1;
            end
        end
        else
        filterImg=zeros(H+2*k,W+2*k);
        [H W L]=size(filterImg);
        for i=1:k
            filterImg(i,k+1:W-k)=oldImg(1,:);
        end
     for i=1:k
            filterImg(H-k+i,k+1:W-k)=oldImg(end,:);
     end
     for i=1:k
            filterImg(1:H,i)=filterImg(:,1);

     end
         for i=1:k
            filterImg(1:H,W-k+i)=filterImg(:,end);
         end
         filterImg(1+k:H-k,1+k:W-k)=oldImg(1:h,1:w);
        for i=1+k:H-k
            for j=1+k:W-k
             subImg=(filterImg(i-k:i+k,j-k:j+k));
              result1=0;
            for f=1:3
                for g=1:3
                    result1=subImg(f,g)*filter(f,g)+result1;
                end
            end
            newImg(i,j)=result1;
            end
        end 
     end
end
newImg=newImg(1+k:H-k,1+k:W-k)+oldImg(1+k:H-k,1+k:W-k);
 output=uint8(newImg); 
end

function output = addUniformNoise(image, noiseRange)
    
    % Convert image to double
    image = im2double(image);
    
    % Validate noiseRange input
    if numel(noiseRange) < 2
        error('noiseRange must be a 2-element array [min, max]');
    end
    
    % Generate uniform noise
    noise = (noiseRange(2) - noiseRange(1)) * rand(size(image)) + noiseRange(1);
    
    % Add noise to image
    noisyImage = image + noise;
    
    % Clip values to [0, 1]
    output = max(min(noisyImage, 1), 0);
    
    % Convert back to uint8 if original was uint8
    if isa(image, 'uint8')
        output = im2uint8(output);
    end
end

function output = GammaCorrection(img, g)

is_uint8 = isa(img, 'uint8');

if is_uint8
    img = double(img) / 255;
end

img = max(img, 0);

corrected = img .^ g;

output = max(min(corrected, 1), 0);

if is_uint8
    output = uint8(corrected * 255);
end

end

function output = max_filter(img, filterSize)

    padSize = floor(filterSize / 2);
    paddedImg = padarray(img, [padSize, padSize], 'replicate');
   
    [rows, cols] = size(img);
    outimg = zeros(rows, cols);

    for i = 1:rows
        for j = 1:cols
            localRegion = paddedImg(i:i+filterSize-1, j:j+filterSize-1);
            outimg(i, j) = max(localRegion(:));
        end
    end

    output = uint8(outimg);
end


function output = Log(r, c)
    r = im2double(r);
    
    s = c * log2(1 + r);
    output = im2uint8(s);
end

function output = opening(img)
    output = erosion(img);
    output = dilation(output);
end

function output = closing(img)
    output = dilation(output);
    output = erosion(img);
end


function output = rayleigh_noise(image, a, b)

if nargin < 3
    error('???? a ? b');
end

image = double(image);
U = rand(size(image));
noise = a + b * sqrt(-2 * log(U)); 

noisy = image + noise;
output = uint8(max(0, min(255, noisy)));
end

function output = exponential_noise(image, b)

if nargin < 2
    error('???? b');
end

image = double(image);
U = rand(size(image));
noise = -b * log(U); % ????? ??

noisy = image + noise;
output = uint8(max(0, min(255, noisy)));
end

function output = addGaussianNoise(image, m, sig, perc)
image = im2double(image);

if nargin < 4 || isempty(perc)
    perc = 100; 
end
factor = perc / 100;

noise = m + sig * randn(size(image));

output = image + factor * noise;

output = max(min(output, 1), 0);

if isa(image, 'uint8')
    output = im2uint8(output);
end
end

function output = GaussianHighPassFilter(img, cutoff)
    
    if size(img, 3) == 3
        img = rgb2gray(img);  
    end

    img_double = double(img);

    F = fft2(img_double);
    
    F_shifted = fftshift(F);
    
    [M, N] = size(img);
    
    [U, V] = meshgrid(1:N, 1:M);
    
    U = U - N/2;
    V = V - M/2;
    
    D = sqrt(U.^2 + V.^2);
    
    H = 1 - exp(-(D.^2) / (2 * (cutoff^2)));  
    
    F_filtered = F_shifted .* H;
    
    F_back = ifftshift(F_filtered);
    
    img_filtered = real(ifft2(F_back));
    
    output = mat2gray(img_filtered);
end

function output = GaussianLowPassFilter(img, cutoff)
    
    if size(img, 3) == 3
        img = rgb2gray(img);  
    end

    img_double = double(img);

    F = fft2(img_double);
    
    F_shifted = fftshift(F);
    
    [M, N] = size(img);
    
    [U, V] = meshgrid(1:N, 1:M);
    
    U = U - N/2;
    V = V - M/2;
    
    D = sqrt(U.^2 + V.^2);
    
    H = exp(-(D.^2) / (2 * (cutoff^2))); 
    
    F_filtered = F_shifted .* H;
    
    F_back = ifftshift(F_filtered);
    
    img_filtered = real(ifft2(F_back));
    
    output = mat2gray(img_filtered);
end

function output = IdealHighPassFilter(img, cutoff)
    
    if size(img, 3) == 3
        img = rgb2gray(img);  
    end

    img_double = double(img);

    F = fft2(img_double);
    
    F_shifted = fftshift(F);
    
    [M, N] = size(img);
    
    [U, V] = meshgrid(1:N, 1:M);
    
    U = U - N/2;
    V = V - M/2;
    
    D = sqrt(U.^2 + V.^2);
    
    H = double(D >= cutoff);  
    
    F_filtered = F_shifted .* H;
    
    F_back = ifftshift(F_filtered);
    
    img_filtered = real(ifft2(F_back));
    
    output = mat2gray(img_filtered);
end

function output = IdealLowPassFilter(img, cutoff)
   
    if size(img, 3) == 3
        img = rgb2gray(img);  
    end

    img_double = double(img);

    F = fft2(img_double);
    
    F_shifted = fftshift(F);
    
    [M, N] = size(img);
    
    [U, V] = meshgrid(1:N, 1:M);
    
    U = U - N/2;
    V = V - M/2;
    
    D = sqrt(U.^2 + V.^2);
    
    H = double(D <= cutoff);  
    
    F_filtered = F_shifted .* H;
    
    F_back = ifftshift(F_filtered);
    
    img_filtered = real(ifft2(F_back));
    
    output = mat2gray(img_filtered);
end

function output= ButterworthHighPassFilter(img, cutoff, order)
    
    if size(img, 3) == 3
        img = rgb2gray(img);  
    end

    img_double = double(img);

    F = fft2(img_double);
    
    F_shifted = fftshift(F);
    
    [M, N] = size(img);
    
    [U, V] = meshgrid(1:N, 1:M);
    
    U = U - N/2;
    V = V - M/2;
    
    D = sqrt(U.^2 + V.^2);
    
    H = 1 ./ (1 + (cutoff ./ D).^ (2 * order));  
    
    H(D == 0) = 0;

    F_filtered = F_shifted .* H;
    
    F_back = ifftshift(F_filtered);
    
    img_filtered = real(ifft2(F_back));
    
    outout = mat2gray(img_filtered);
end

function output = ButterworthLowPassFilter(img, cutoff, order)
   
    if size(img, 3) == 3
        img = rgb2gray(img);  
    end

    img_double = double(img);

    F = fft2(img_double);
    
    F_shifted = fftshift(F);
    
    [M, N] = size(img);
    
    [U, V] = meshgrid(1:N, 1:M);
    
    U = U - N/2;
    V = V - M/2;
    
    D = sqrt(U.^2 + V.^2);
    
    H = 1 ./ (1 + (D ./ cutoff).^(2 * order));  
    
    F_filtered = F_shifted .* H;
    
    F_back = ifftshift(F_filtered);
    
    img_filtered = real(ifft2(F_back));
    
    output= mat2gray(img_filtered);
end

function output = median_filter(img, filterSize)

    padSize = floor(filterSize / 2);
    paddedImg = padarray(img, [padSize, padSize], 'replicate');
   
    [rows, cols] = size(img);
    outimg = zeros(rows, cols);

    for i = 1:rows
        for j = 1:cols
            localRegion = paddedImg(i:i+filterSize-1, j:j+filterSize-1);
            outimg(i, j) = median(localRegion(:));
        end
    end

    output = uint8(outimg);
end


function output = min_filter(img, filterSize)

    padSize = floor(filterSize / 2);
    paddedImg = padarray(img, [padSize, padSize], 'replicate');
   
    [rows, cols] = size(img);
    outimg = zeros(rows, cols);

    for i = 1:rows
        for j = 1:cols
            localRegion = paddedImg(i:i+filterSize-1, j:j+filterSize-1);
            outimg(i, j) = min(localRegion(:));
        end
    end

    output = uint8(outimg);
end

function output = PointSharpening(img, filter_size, alpha)


    if size(img, 3) == 3
        img = rgb2gray(img);  
    end

    img_double = double(img);

    blur_filter = fspecial('average', filter_size);  
    
    img_blurred = imfilter(img_double, blur_filter, 'replicate');
    
    % Apply unsharp masking: original + alpha * (original - blurred)
    img_sharpened = img_double + alpha * (img_double - img_blurred);
    
    output = uint8(mat2gray(img_sharpened) * 255);  
end

function output = pointdetection(image)
    if size(image, 3) == 3
        image = rgb2gray(image);
    end
    
    image = im2double(image);
    
    sobelX = [-1, 0, 1; -2, 0, 2; -1, 0, 1];
    sobelY = [-1, -2, -1; 0, 0, 0; 1, 2, 1];
    
    gradX = conv2(image, sobelX, 'same');
    gradY = conv2(image, sobelY, 'same');
    
    edgeImage = sqrt(gradX.^2 + gradY.^2);
    
    output = im2uint8(edgeImage);
end

function output = weightfilter(img, kernel)

    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    
    img = double(img); 

    [rows, cols] = size(img);
    [kRows, kCols] = size(kernel);

    padRows = floor(kRows / 2);
    padCols = floor(kCols / 2);

    paddedImg = padarray(img, [padRows, padCols], 'replicate');

    outimg = zeros(rows, cols);

    for i = 1:rows
        for j = 1:cols
    
            localRegion = paddedImg(i:i+kRows-1, j:j+kCols-1);
            
            weightedSum = sum(sum(localRegion .* kernel));
            
            outimg(i, j) = weightedSum;
        end
    end

    output = uint8(outimg);
end

function output = mean_filter(img, filterSize)

    padSize = floor(filterSize / 2);
    paddedImg = padarray(img, [padSize, padSize], 'replicate');
   
    [rows, cols] = size(img);
    outimg = zeros(rows, cols);

    for i = 1:rows
        for j = 1:cols
            localRegion = paddedImg(i:i+filterSize-1, j:j+filterSize-1);
            outimg(i, j) = mean(localRegion(:));
        end
    end

    output = uint8(outimg);
end

function output = correlation(img, kernel)

if size(img, 3) == 3
    img = rgb2gray(img);  
end

img_double = double(img);

[k_rows, k_cols] = size(kernel);

%padding 
pad_rows = floor(k_rows / 2);
pad_cols = floor(k_cols / 2);


img_padded = padarray(img_double, [pad_rows, pad_cols], 'both');


[img_rows, img_cols] = size(img_double);


img_correlated = zeros(img_rows, img_cols);


for i = 1:img_rows
    for j = 1:img_cols
       
        roi = img_padded(i:i+k_rows-1, j:j+k_cols-1);
        
       
        img_correlated(i, j) = sum(sum(roi .* kernel));
    end
end


output = uint8(mat2gray(img_correlated) * 255);

end

function output = histogramEqualization(img)
if size(img, 3) == 3
    img_gray = rgb2gray(img);
    equalized = histeq(img_gray); 
    output = im2uint8(equalized);
else
    output = im2uint8(histeq(img));
end
end

function output = InverseFourierTransform(I)
    fi = fft2(I);
    f1 = fftshift(fi);
    n = abs(f1);
    n = log(1 + n);
    n = mat2gray(n);

    final = fftshift(f1);   
    final = ifft2(final);  
    final = abs(final);    
    final = log(1 + final);
    output = mat2gray(final);  
end


function output = FourierTransform(img)
    if nargin < 1
        error('Input image I is required.');
    end
    fi = fft2(img);
    f1 = fftshift(fi);
    n = abs(f1);
    n = log(1 + n);
    output = mat2gray(n);
end


function output = negative(img)
img = double(img);
[w, h]=size(img);
newImg = zeros(w, h);
for i=1:w
    for j=1:h
        newImg(i,j) = 255 - img(i,j);
    end

end
output = uint8(newImg) 
end

function output = brightnessNew(img, value)
    imgDouble = double(img) + value;
    imgDouble(imgDouble > 255) = 255;
    imgDouble(imgDouble < 0) = 0;
    output = uint8(imgDouble);
end

function output = brightness(img, option, value)
    img = double(img);
    [h, w] = size(img);
    output = zeros(h, w); 

    for i = 1:h
        for j = 1:w
            if option == 1
                output(i,j) = img(i,j) + value;  % Brighten
            elseif option == 2
                output(i,j) = img(i,j) - value;  % Darken
            elseif option == 3
                output(i,j) = img(i,j) * value;  % Multiply brightness
            elseif option == 4
                if value ~= 0
                    output(i,j) = img(i,j) / value;  % Divide brightness
                else
                    output(i,j) = img(i,j);  % Avoid division by zero
                end
            end
        end
    end

    output(output > 255) = 255;
    output(output < 0) = 0;

    output = uint8(output);
end


function output = addSaltAndPepperNoise(image, noiseDensity)
   
    if noiseDensity < 0 || noiseDensity > 1
        error('Noise density must be between 0 and 1.');
    end
    
    output = imnoise(image, 'salt & pepper', noiseDensity);
end


% --- SUB FUNCTIONS ---
function output = rgbtobinary(rgb,th)
    img = rgb2gray(rgb , 2 );
    output = graytobinary (img , th);
end

% --- NEW CODE ---
function output = applyVerticalEdgeDetection(img)
    % 1. Convert to grayscale if necessary
    if size(img, 3) == 3
        img_gray = rgb2gray(img);
    else
        img_gray = img;
    end
    
    % 2. Use a hardcoded Sobel Kernel (Vertical Edge)
    % sobel_kernel = fspecial('sobel'); % <-- Removed
    sobel_kernel_v = [-1 -2 -1; 0 0 0; 1 2 1]; % Transposed for vertical
    
    % 3. Apply Filter
    % We apply the vertical kernel directly, no need for transpose (sobel_kernel')
    edge_image = imfilter(double(img_gray), sobel_kernel_v, 'replicate'); 
    
    % 4. Absolute value
    edge_image = abs(edge_image);
    
    % 5. Normalize (Assign to OUTPUT)
    output = mat2gray(edge_image);
end

function output = erosion(img)
    se = strel('disk', 5);
    if size(img, 3) == 3
        output = img;
        for i = 1:3, output(:,:,i) = imerode(img(:,:,i), se); end
    else
        output = imerode(img, se);
    end
end

function output = dilation(img)
    se = strel('disk', 5);
    if size(img, 3) == 3
        output = img;
        for i = 1:3, output(:,:,i) = imdilate(img(:,:,i), se); end
    else
        output = imdilate(img, se);
    end
end

function output = contrastStretching(img)
    imgD = double(img);
    minVal = min(imgD(:))
    maxVal = max(imgD(:))
    if(maxVal - minVal) == 0
        output = img
        return;
    end
    output = ((imgD-minVal)/(maxVal-minVal) * 255);
    output = uint8(output)
end

function noisy = gamma_noise(image, k, theta)
    image = double(image);
    noise = zeros(size(image));
    for i = 1:k
        U = rand(size(image));
        noise = noise + theta * (-log(U));
    end
    noisy = uint8(max(0, min(255, image + noise)));
end

function result = generateHistogram(img)
    f = figure('Visible', 'off');
    if size(img, 3) == 3
        subplot(3,1,1); imhist(img(:,:,1)); title('Red');
        subplot(3,1,2); imhist(img(:,:,2)); title('Green');
        subplot(3,1,3); imhist(img(:,:,3)); title('Blue');
    else
        imhist(img); title('Grayscale');
    end
    frame = getframe(f);
    result = frame.cdata;
    close(f);
end

function binary = graytobinary(gray, th)
    binary = gray >= th;
end