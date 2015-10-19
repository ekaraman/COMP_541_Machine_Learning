void AHM(Mat I, int blockSize){
	
	//Check blockSize to be odd number
	if (blockSize % 2 == 0){
		blockSize = blockSize + 1;
	}
	
	Mat	blockImage = Mat::zeros(blockSize, blockSize, CV_8UC1);
	int nRows = I.rows;
	int nCols = I.cols;
	Mat ahmImg = I.clone();
	int rowofBlock = nRows / blockSize;
	int colofBlock= nCols / blockSize;

	//  Allocate 3D Array for Cumulative Map for each block
	int ***cumuMap;
	cumuMap = new int**[rowofBlock];
	for(int i = 0; i < rowofBlock; i++){
		cumuMap[i] = new int*[colofBlock];
		for(int j = 0; j < colofBlock; j++){
			cumuMap[i][j] = new int[256];
		}
	}

	int tmp[256];
	//Calculate cumulative values for each block
	int xImg;
	int yImg;
	for(int i = 0; i < rowofBlock; i++){
		for(int j = 0; j < colofBlock; j++){
			for(int x = 0; x < blockSize; x++){
				for(int y = 0; y < blockSize; y++){
					xImg = blockSize * i + x;
					yImg = blockSize * j + y;
					blockImage.at<uchar>(x,y) = I.at<Vec3b>(xImg,yImg)[0];
				}
			}
			//Find intesity values
			HistEquAHM(blockImage,tmp);
			for (int k = 0; k < 255; k++){
				cumuMap[i][j][k] = tmp[k];
			}
		}
	}

	int rowUpBorder = (blockSize / 2) + 1;
	int rowDownBorder = ((rowofBlock * blockSize) - (blockSize / 2)) - 1;
	int colLeftBorder = ((colofBlock * blockSize) - (blockSize / 2)) - 1;

	int tile_x;
	int tile_y;
	int tile_xup;
	int tile_xdown;
	int tile_yleft;
	int tile_yright;

	int tileULX;
	int tileURX;
	int tileLLX;
	int tileLRX;

	int tileULY;
	int tileURY;
	int tileLLY;
	int tileLRY;

	int centerULX;
	int centerURX;
	int centerLLX;
	int centerLRX;

	int centerULY;
	int centerURY;
	int centerLLY;
	int centerLRY;

	int x2_x1;
    int x2_x;
    int y2_y1;
    int y2_y;
    int x_x1;
    int y_y1;

	int fQ11;
    int fQ12;
	int fQ21;
	int fQ22;

	for(int i = 0; i < nRows; i++){
		for(int j = 0; j < nCols; j++){
			//find 4 neighbor tile's center
			tile_x = (i+1) / blockSize;
			tile_y = (j+1) / blockSize;

			if (((i+1) % blockSize) <= ((blockSize + 1) / 2)) {
				tile_xup = tile_x ;
				tile_xdown = tile_x + 1;
			}
			else{
				tile_xup = tile_x + 1;
				tile_xdown = tile_x + 2;
			}

			if (((j+1) % blockSize) <= ((blockSize + 1) / 2)) {
				tile_yleft = tile_y;
				tile_yright = tile_y + 1;
			}
			else{
				tile_yleft = tile_y + 1;
				tile_yright = tile_y + 2;
			}

			tileULX = tile_xup;
			tileURX = tile_xup;
			tileLLX = tile_xdown;
			tileLRX = tile_xdown;

			tileULY = tile_yleft;
			tileURY = tile_yright;
			tileLLY = tile_yleft;
			tileLRY = tile_yright;

			centerULX = ((tileULX - 1) * blockSize + (blockSize + 1) / 2) - 1;
			centerULY = ((tileULY - 1) * blockSize + (blockSize + 1) / 2) - 1;

            centerURX = ((tileURX - 1) * blockSize + (blockSize + 1) / 2) - 1;				
			centerURY = ((tileURY - 1) * blockSize + (blockSize + 1) / 2) - 1;

			centerLLX = ((tileLLX - 1) * blockSize + (blockSize + 1) / 2) - 1;		
			centerLLY = ((tileLLY - 1) * blockSize + (blockSize + 1) / 2) - 1;

			centerLRX = ((tileLRX - 1) * blockSize + (blockSize + 1) / 2) - 1;
			centerLRY = ((tileLRY - 1) * blockSize + (blockSize + 1) / 2) - 1;


			// Get the value in the bilinear equation.
            x2_x1 = centerLLX - centerULX;
            x2_x = centerLLX - i;
            y2_y1 = centerURY - centerULY;
            y2_y = centerURY - j;
            x_x1 = i - centerULX;
            y_y1 = j - centerULY;

			//equalImg.at<Vec3b>(y,x)[0] = saturate_cast<uchar>(scaledCum[ycc.at<Vec3b>(y,x)[0]]);

			fQ11 = saturate_cast<uchar>(cumuMap[tile_xup - 1][tile_yleft - 1][I.at<Vec3b>(i,j)[0]]);
			fQ12 = saturate_cast<uchar>(cumuMap[tile_xup - 1][tile_yright - 1][I.at<Vec3b>(i,j)[0]]);
			fQ21 = saturate_cast<uchar>(cumuMap[tile_xdown - 1][tile_yleft - 1][I.at<Vec3b>(i,j)[0]]);
			fQ22 = saturate_cast<uchar>(cumuMap[tile_xdown - 1][tile_yright - 1][I.at<Vec3b>(i,j)[0]]);
			
			ahmImg.at<Vec3b>(i,j)[0] = saturate_cast<uchar>(fQ11 * x2_x * y2_y / (x2_x1 * y2_y1) + \
                fQ21 * x_x1 * y2_y /( x2_x1 * y2_y1) + \
                fQ12 * (x2_x) * (y_y1) /(x2_x1 * y2_y1) + \
                fQ22 * (x_x1) * (y_y1) / (x2_x1 * y2_y1));
		}
	}


	Mat ahmImg_BGR;
	ahmImg_BGR = myHistEqu(ahmImg);
	cvtColor(ahmImg, ahmImg_BGR, CV_YCrCb2BGR);
	namedWindow("AHM Equalized Image", CV_WINDOW_AUTOSIZE );
	imshow("AHM Equalized Image", ahmImg_BGR);
	waitKey(0);
	cvDestroyWindow("AHM Equalized Image");
	//imwrite( "Saalfeld_AHM_Equalized_Image.jpg", ahmImg_BGR );

	//Free memory
	for(int i = 0; i < rowofBlock; i++)
	{
		for(int j = 0; j < colofBlock; j++)
		{
			free (cumuMap[i][j]);
		}
		free (cumuMap[i]);
	}
	free(cumuMap);
}
