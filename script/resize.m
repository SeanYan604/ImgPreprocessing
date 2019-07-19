pic_name = 'test_image_5.png';
pic = imread(pic_name);
pic_resize = imresize(pic,[120,120]);
gray = rgb2gray(pic_resize);
imwrite(gray, pic_name);