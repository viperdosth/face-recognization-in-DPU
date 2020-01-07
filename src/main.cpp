#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include "opencv2/core/core.hpp"
// #include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <dnndk/dnndk.h>

using namespace std;
using namespace std::chrono;
using namespace cv;

#define KERNEL_CONV "face_0"
#define CONV_INPUT_NODE "Conv2D"
#define CONV_OUTPUT_NODE "MatMul_1"

#define CAM_DEV "xlnxvideosrc src-type=\"mipi\" ! video/x-raw, framerate=30/1, width=1920, height=1080, format=YUY2 ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! appsink"

bool is_reading = true;
bool is_running = true;
bool is_displaying = true;
VideoCapture video;

class Compare
{
public:
    bool operator()(const pair<int, Mat> &n1, const pair<int, Mat> &n2) const
    {
        return n1.first > n2.first;
    }
};
queue<pair<int, Mat>> read_queue;                                              // read queue
priority_queue<pair<int, Mat>, vector<pair<int, Mat>>, Compare> display_queue; // display queue
mutex mtx_read_queue;                                                          // mutex of read queue
mutex mtx_display_queue;                                                       // mutex of display queue
int read_index = 0;                                                            // frame index of input video
int display_index = 0;                                                         // frame index to display

CascadeClassifier faceCascade;

void Read(bool &is_reading)
{
    while (is_reading)
    {
        Mat img;
        if (read_queue.size() < 30)
        {
            if (!video.read(img))
            {
                cout << "Finish reading the video." << endl;
                is_reading = false;
                break;
            }
            mtx_read_queue.lock();
            read_queue.push(make_pair(read_index++, img));
            mtx_read_queue.unlock();
        }
        else
        {
            usleep(20);
        }
    }
}
// void CPUSoftmax(int8_t *src, int size, float scale, float *dst) {
//     float sum = 0.0f;

//     for (auto i = 0; i < size; ++i) {
//         dst[i] = exp(src[i] * scale);
//         sum += dst[i];
//     }

//     for (auto i = 0; i < size; ++i) {
//         dst[i] /= sum;
//     }
// }

void input_normalize_image(DPUTask *task_conv, const Mat& image) {
	DPUTensor* dpu_in = dpuGetInputTensor(task_conv, CONV_INPUT_NODE);
	int8_t* data = dpuGetTensorAddress(dpu_in);
	float scale = dpuGetTensorScale(dpu_in);
	for(int i = 0; i < 3; ++i) {
		for(int j = 0; j < image.rows; ++j) {
			for(int k = 0; k < image.cols; ++k) {
				data[j*image.rows*3+k*3+2-i] = (float(image.at<Vec3b>(j,k)[i])/255.0) * scale;
//				data[j*image.cols*3+k*3+i] = (float(image.at<Vec3b>(j,k)[i])/255.0 - 0.5)*2 * scale;
			}
		}
	}
}

void Face(DPUTask *task_conv, bool &is_running)
{
    while (is_running)
    {

        // int8_t *is_zhang = dpuGetOutputTensorAddress(task_conv, CONV_OUTPUT_NODE);
        DPUTensor *conv_out_tensor = dpuGetOutputTensor(task_conv, CONV_OUTPUT_NODE);
        int tensorSize = dpuGetTensorSize(conv_out_tensor);
        float oscale = dpuGetTensorScale(conv_out_tensor);
        vector<float> pixel(tensorSize);
        Mat img, imgGray;
        int index;
        vector<Rect> faces;
        mtx_read_queue.lock();
        if (read_queue.empty())
        {
            mtx_read_queue.unlock();
            if (is_reading)
            {
                continue;
            }
            else
            {
                is_running = false;
                break;
            }
        }
        else
        {
            index = read_queue.front().first;
            img = read_queue.front().second;
            read_queue.pop();
            mtx_read_queue.unlock();
        }

        faceCascade.load("/usr/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml");
        cvtColor(img, imgGray, CV_RGB2GRAY);
        faceCascade.detectMultiScale(imgGray, faces, 1.2, 6, 0, Size(0, 0));
        for (size_t i = 0; i < faces.size(); i++)
        {
            Mat img_face;
            Mat resize_face;
//            rectangle(img, Point(faces[i].x + 10, faces[i].y + 10),
//                      Point(faces[i].x + faces[i].width + 10, faces[i].y + faces[i].height + 10),
//                      Scalar(0, 255, 0), 1, 8);
            img_face = img(faces[i]);
            resize(img_face, resize_face, Size(64, 64));
            input_normalize_image(task_conv, resize_face);

//            dpuSetInputImage2(task_conv, (char *)CONV_INPUT_NODE, resize_face);

            dpuRunTask(task_conv);
            dpuGetOutputTensorInHWCFP32(task_conv, CONV_OUTPUT_NODE, pixel.data(), tensorSize);

//            float xqizhang = 1 / (1 + exp(-(float)pixel[0]));
//            float qizhang = 1 / (1 + exp((float)pixel[0]));
////          float xqizhang = exp(-(float)pixel[0]) / (float)pixel[0];
////          float qizhang = exp((float)pixel[0]) / (float)pixel[0];
//            std::cout << "zzzz" << tensorSize << " qqqq " << pixel.size() << std::endl;
//            std::cout << "xxxx " << pixel[0] << "," << pixel[1] << std::endl;
//            float sum = 0;
//            for (size_t p = 0; p < 2; p++) {
//            	pixel[p] = exp(pixel[p]);
//            	sum += pixel[p];
//            }
//            for (size_t p = 0; p < 2; p++) {
//            	pixel[i] /= sum;
//            }
//            for (size_t p = 0; p < 2; p++) {
//            	pixel[p] = oscale * pixel[p];
//            }
//            for (size_t p = 0; p < 2; p++) {
//            	pixel[p] = 1.0/ (1 + exp(-pixel[p]));
//            }
//            for (size_t p; p < 2; p++) {
//            	pixel[i] /= sum;
//            }
            float qizhang = (pixel[0] > pixel[1]);
            float xqizhang = (pixel[0] <= pixel[1]);
            std::cout << " yyy " << qizhang << ", " << xqizhang << std::endl;
            if (qizhang < 0.2)
            {
            	rectangle(img, Point(faces[i].x + 10, faces[i].y + 10),
            	                      Point(faces[i].x + faces[i].width + 10, faces[i].y + faces[i].height + 10),
            	                      Scalar(0, 255, 0), 1, 8);
                putText(img, "qizhang", Point(faces[i].x, faces[i].y),
                        FONT_HERSHEY_DUPLEX, 1, Scalar(0, 143, 143), 1, false);
            }
        }

        mtx_display_queue.lock();
        display_queue.push(make_pair(index, img));
        mtx_display_queue.unlock();
    }
}

void Display(bool &is_displaying)
{
    while (is_displaying)
    {
        mtx_display_queue.lock();
        if (display_queue.empty())
        {
            if (is_running)
            {
                mtx_display_queue.unlock();
                usleep(20);
            }
            else
            {
                is_displaying = false;
                break;
            }
        }
        else if (display_index == display_queue.top().first)
        {
            // Display image
            imshow("Face Recognization", display_queue.top().second);
            display_index++;
            display_queue.pop();
            mtx_display_queue.unlock();
            if (waitKey(1) == 'q')
            {
                is_reading = false;
                is_running = false;
                is_displaying = false;
                break;
            }
        }
        else
        {
            mtx_display_queue.unlock();
        }
    }
}

int main(int argc, char **argv)
{

    // Check args
//    if (argc != 2)
//    {
//        cout << "Usage of face demo: ./face file_name[string]" << endl;
//        cout << "\tfile_name: path to your video file" << endl;
//        return -1;
//    }
    DPUKernel *kernel_conv;
    DPUTask *task_conv;
    dpuOpen();
    kernel_conv = dpuLoadKernel(KERNEL_CONV);
    task_conv = dpuCreateTask(kernel_conv, 0);

    // Initializations
    string file_name;
    if (argc > 1) file_name = argv[1];
    else file_name = CAM_DEV;
    cout << "Detect video: " << file_name << endl;
    video.open(file_name);
    if (!video.isOpened())
    {
        cout << "Failed to open video: " << file_name;
        return -1;
    }

    // Run tasks for SSD
    array<thread, 3> threads = {thread(Read, ref(is_reading)),
                                thread(Face, task_conv, ref(is_running)),
                                thread(Display, ref(is_displaying))};

    for (int i = 0; i < 3; ++i)
    {
        threads[i].join();
    }

    dpuDestroyTask(task_conv);
    video.release();

    return 0;
}
