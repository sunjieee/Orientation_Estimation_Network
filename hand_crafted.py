#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

class ROTATE
{
public:
	ROTATE() = default;
	ROTATE(const char *file_name);
	~ROTATE() {};
	
	void rotate_image();
	void save_image(const char *output_file);
	
private:
	int get_row_start(int r);
	int get_col_start(int c);
	float get_bin_ori(int bin_id);
	bool check_in(int x, int y);
	void smooth_histbin(float *histbin);
	void rotate_image_patch(int r, int c, float ori);
	void rotate_patch(int r, int c, float ori, uchar **patch);
	void gaussian_blur(int r, int c, uchar **patch);
	float gaussian_weighted(int x, int y, float delta);
	float select_main_ori(int r, int c, vector<pair<float, int> > &candidate_ori);
	float calc_ori(int r, int c);
	float calc_gravity_centre(uchar **patch);
	void calc_ori_histbin(float *histbin, uchar **patch);
	void calc_candidate_ori(vector<pair<float, int> > &candidate_ori, float *histbin);

	Mat image;
	const int ROWS = 16;
	const int COLS = 16;
	const int WIDTH_ROWS = 227;
	const int WIDTH_COLS = 227;
	const int CENTRE_ROWS = 113;
	const int CENTRE_COLS = 113;
	const int BIN_NUM = 36;
	const float GAUSSIAN_VALUE = 113.0f;
	const float ORI_RATIO = 0.8f;
	const float GAUSSIAN_MATRIX[3][3] = {{0.0947416, 0.118318, 0.0947416}, {0.118318, 0.147761, 0.118318}, {0.0947416, 0.118318, 0.0947416}};
};

ROTATE::ROTATE(const char *file_name)
{
	image = imread(file_name, 0);
}


int ROTATE::get_row_start(int r)
{
	return WIDTH_ROWS * r;
}

int ROTATE::get_col_start(int c)
{
	return WIDTH_COLS * c;
}

bool ROTATE::check_in(int x, int y)
{
	if ((x - CENTRE_ROWS) * (x - CENTRE_ROWS) + (y - CENTRE_COLS) * (y - CENTRE_COLS) <= CENTRE_COLS * CENTRE_COLS)
	{
		return true;
	}
	return false;
}

float ROTATE::get_bin_ori(int bin_id)
{
	return 1.0f * ((bin_id - BIN_NUM / 2) * (360 / BIN_NUM) + (180 / BIN_NUM));
}

float ROTATE::gaussian_weighted(int x, int y, float delta)
{
	return exp(-((x - CENTRE_ROWS)* (x - CENTRE_ROWS) + (y - CENTRE_COLS)* (y - CENTRE_COLS)) / 2.0 / delta / delta);
}

void ROTATE::gaussian_blur(int r, int c, uchar **patch)
{
	int r_s = get_row_start(r);
	int c_s = get_col_start(c);
	for (int i = 0; i < WIDTH_ROWS; ++i)
	{
		for (int j = 0; j < WIDTH_COLS; ++j)
		{
			int r_p = r_s + i;
			int c_p = c_s + j;
			if (i == 0 || j == 0 || i == WIDTH_ROWS - 1 || j == WIDTH_COLS - 1)
			{
				patch[i][j] = image.at<uchar>(r_p, c_p);
				continue;
			}
			float temp = 0.0f;
			for (int k = r_p - 1; k <= r_p + 1; ++k) 
			{
				for (int l = c_p - 1; l <= c_p + 1; ++l)
				{
					temp += image.at<uchar>(k, l) * GAUSSIAN_MATRIX[k - (r_p - 1)][l - (c_p - 1)];
				}
			}
			patch[i][j] = (uchar)(temp + 0.5);
		}
	}
}

float ROTATE::calc_gravity_centre(uchar **patch)
{
	int sum_gray_value = 0;
	int sum_gravity_value = 0;
	for (int i = 0; i < WIDTH_ROWS; ++i)
	{
		for (int j = 0; j < WIDTH_COLS; ++j)
		{
			sum_gray_value += patch[i][j];
			sum_gravity_value += patch[i][j] * (i + 1);
		}
	}
	return 1.0f * sum_gravity_value / sum_gray_value;
}

float ROTATE::select_main_ori(int r, int c, vector<pair<float, int> > &candidate_ori)
{
	int r_s = get_row_start(r);
	int c_s = get_col_start(c);
	uchar **patch = new uchar* [WIDTH_ROWS];
	for (int i = 0; i < WIDTH_ROWS; ++i)
	{
		patch[i] = new uchar [WIDTH_COLS];
	}
	float gravity_centre_max = 0.0f;
	float ori_select = 0.0f;
	int size = candidate_ori.size();
	for (int i = 0; i < size; ++i)
	{
		float ori = get_bin_ori(candidate_ori[i].second);
		rotate_patch(r, c, ori, patch);
		float gravity_centre = calc_gravity_centre(patch);
		if (gravity_centre_max < gravity_centre)
		{
			gravity_centre_max = gravity_centre;
			ori_select = ori;
		}
	}
	for (int i = 0; i < WIDTH_ROWS; ++i)
	{
		delete patch[i];
	}
	delete[] patch;
	return ori_select;
}

void ROTATE::smooth_histbin(float *histbin)
{
	float *hist_temp = new float [BIN_NUM];
	for (int i = 0; i < BIN_NUM; ++i)
	{
		hist_temp[i] = histbin[i];
	}
	for (int i = 0; i < BIN_NUM; ++i)
	{
		histbin[i] = (hist_temp[(i - 2 + BIN_NUM) % BIN_NUM] + hist_temp[(i + 2) % BIN_NUM]) / 16.0f + \
			(hist_temp[(i - 1 + BIN_NUM) % BIN_NUM] + hist_temp[(i + 1) % BIN_NUM]) / 4.0f + \
			hist_temp[i] * 6.0f / 16.0f;
	}
	delete hist_temp;
}

void ROTATE::calc_ori_histbin(float *histbin, uchar **patch)
{
	for (int i = 0; i < BIN_NUM; ++i)
	{
		histbin[i] = 0.0f;
	}
	for (int i = 1; i < WIDTH_ROWS - 1; ++i)
	{
		for (int j = 1; j < WIDTH_COLS - 1; ++j)
		{
			if (check_in(i, j))
			{
				//float dy = (float)(patch[i + 1][j] - patch[i - 1][j]);
				float dy = (float)(patch[i - 1][j] - patch[i + 1][j]);
				float dx = (float)(patch[i][j + 1] - patch[i][j - 1]);
				float magnitude = sqrt(dx * dx + dy * dy);
				float ori = atan2(dy, (dx + 1e-9)) / acos(-1.0) * 180;
				int bin = (int)(ori / (360 / BIN_NUM) + BIN_NUM / 2);
				float gaussian_add = gaussian_weighted(i, j, GAUSSIAN_VALUE);
				//histbin[bin] += magnitude * gaussian_add;
				histbin[bin] += magnitude;
			}
		}
	}
	smooth_histbin(histbin);
}

void ROTATE::calc_candidate_ori(vector<pair<float, int> > &candidate_ori, float *histbin)
{
	float histbin_max = 0.0f;
	int bin_id = 0;
	for (int i = 0; i < BIN_NUM; ++i)
	{
		if (histbin_max < histbin[i])
		{
			histbin_max = histbin[i];
			bin_id = i;
		}
		//cout << histbin[i] << " ";   //output
	}
	//cout << endl; ////
	candidate_ori.push_back(make_pair(histbin_max, bin_id));
	/*for (int i = 0; i < BIN_NUM; ++i)
	{
		if (i != bin_id && histbin[i] > histbin_max * ORI_RATIO)
		{
			candidate_ori.push_back(make_pair(histbin[i], i));
		}
	}*/
}

float ROTATE::calc_ori(int r, int c)
{
	uchar **patch = new uchar* [WIDTH_ROWS];
	for (int i = 0; i < WIDTH_ROWS; ++i)
	{
		patch[i] = new uchar [WIDTH_COLS];
	}
	gaussian_blur(r, c, patch);
	int r_s = get_row_start(r);
	int c_s = get_col_start(c);
	float *histbin = new float [BIN_NUM];
	calc_ori_histbin(histbin, patch);
	for (int i = 0; i < WIDTH_ROWS; ++i)
	{
		delete patch[i];
	}
	delete[] patch;
	vector<pair<float, int> > candidate_ori;
	calc_candidate_ori(candidate_ori, histbin);
	delete histbin;
	//return select_main_ori(r, c, candidate_ori);
	return get_bin_ori(candidate_ori[0].second);
}

void ROTATE::rotate_patch(int r, int c, float ori, uchar **patch)
{
	int r_s = get_row_start(r);
	int c_s = get_col_start(c);
	float cos_theta = cos(ori);
	float sin_theta = sin(ori);
	for (int i = 0; i < WIDTH_ROWS; ++i)
	{
		for (int j = 0; j < WIDTH_COLS; ++j)
		{
			if (check_in(i, j))
			{
				int y = (int)((j - CENTRE_COLS) * cos_theta - (i - CENTRE_ROWS) * sin_theta + CENTRE_COLS + 0.5);
				int x = (int)((j - CENTRE_COLS) * sin_theta + (i - CENTRE_ROWS) * cos_theta + CENTRE_ROWS + 0.5);
				patch[i][j] = image.at<uchar>(r_s + x, c_s + y);
			}
			else
			{
				patch[i][j] = 0;
			}
		}
	}
}

void ROTATE::rotate_image_patch(int r, int c, float ori)
{
	uchar **patch = new uchar* [WIDTH_ROWS];
	for (int i = 0; i < WIDTH_ROWS; ++i)
	{
		patch[i] = new uchar [WIDTH_COLS];
	}
	int r_s = get_row_start(r);
	int c_s = get_col_start(c);
	rotate_patch(r, c, ori, patch);
	for (int i = 0; i < WIDTH_ROWS; ++i)
	{
		for (int j = 0; j < WIDTH_COLS; ++j)
		{
			image.at<uchar>(r_s + i, c_s + j) = patch[i][j];
		}
	}
	for (int i = 0; i < WIDTH_ROWS; ++i)
	{
		delete patch[i];
	}
	delete[] patch;
}
/*
void ROTATE::rotate_image()
{
	for (int i = 0; i < ROWS; ++i)
	{
		for (int j = 0; j < COLS; ++j)
		{
			float ori = calc_ori(i, j);
			rotate_image_patch(i, j, ori);
			//cout << i << " " << j << " " << ori << endl;   ///
		}
	}
}*/
void ROTATE::rotate_image()
{
	resize(image, image, Size(227, 227));
	float ori = calc_ori(0, 0);
	cout << ori << endl;
	Mat rotMat(2, 3, CV_32FC1);
	rotMat = getRotationMatrix2D(Point(113, 113), ori, 1);
	warpAffine(image, image, rotMat, Size(227, 227));
	imshow("test", image);
	waitKey(0);
	//rotate_image_patch(0, 0, ori);
}

void ROTATE::save_image(const char *output_file)
{
	imwrite(output_file, image);
}

int main(int argc, char const *argv[])
{
	ROTATE image_object(argv[1]);
	image_object.rotate_image();
	image_object.save_image(argv[2]);
	return 0;
}
