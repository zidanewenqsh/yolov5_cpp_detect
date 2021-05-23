//
// Created by qinrui on 2019/12/2.
//

#ifndef __DETECTOR_HPP__
#define __DETECTOR_HPP__
#ifdef DETECTOR_EXPORTS
#define DETECTOR_API __declspec(dllexport)
#else
#define DETECTOR_API __declspec(dllimport)
#endif
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#define Print(str) cout << #str << "=" << str << endl
#define ANCHOR_NUM 6

using namespace std;

// std::vector<torch::Tensor> non_max_suppression(torch::Tensor & preds, float score_thresh = 0.01, float iou_thresh = 0.35)
// {
//     std::vector<torch::Tensor> output;
//     for (size_t i = 0; i < preds.sizes()[0]; i++)
//     {
//         torch::Tensor pred = preds.select(0, i);
//         cout << "pred:" << pred.sizes() << endl;
//     }
//     return output;
// }

#ifdef WIN32
class DETECTOR_API Detector
#else
class Detector
#endif
{
private:
    torch::Tensor img_tensor = torch::empty({0}), preds = torch::empty({0});
    // cv::Mat img, img0;
    int batch_size = 1;
    int channel_size = 3;
    bool DIoU = false;
    int w = 640;
    int h = 640;
    float cfd_threshold = 0.25;
    float nms_threshold = 0.45;
    torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);
//    torch::Device device = torch::Device(torch::kCPU);
    torch::jit::script::Module module;
    string model_file;

    // int load_image(cv::Mat &img, const string &path, int w, int type)
    // 前向计算（单输出）
    int module_forward(torch::Tensor &y, const torch::Tensor &x);

    // // 前向计算（多输出）
    // int module_forward(std::vector<torch::Tensor> &vts_y, const torch::Tensor &x);

    // Mat数组转Tensor
    int mat2tensor(const cv::Mat& img);

public:
    std::vector<torch::Tensor> output;

    // torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);

    // torch::jit::script::Module module;

    Detector(string mfile): model_file(mfile)
    {
        init();
    }
    Detector(string mfile, int batchsize) : model_file(mfile), batch_size(batchsize)
    {
        init();
    }
    Detector(string mfile, int batchsize, int channelsize, int w, int h) : model_file(mfile), batch_size(batchsize), channel_size(channelsize), w(w), h(h)
    {
        init();
    }

    Detector(string mfile, int batchsize, int channelsize, int w, int h, float cfdthresh, float nmsthresh) : model_file(mfile), batch_size(batchsize), channel_size(channelsize), w(w), h(h), cfd_threshold(cfdthresh), nms_threshold(nmsthresh)
    {
        init();
    }
    Detector(string mfile, int batchsize, int channelsize, int w, int h, bool diou) : model_file(mfile), batch_size(batchsize), channel_size(channelsize), w(w), h(h), DIoU(diou)
    {
        init();
    }
    Detector(string mfile, int batchsize, int channelsize, int w, int h, float cfdthresh, float nmsthresh, bool diou) : model_file(mfile), batch_size(batchsize), channel_size(channelsize), w(w), h(h), cfd_threshold(cfdthresh), nms_threshold(nmsthresh), DIoU(diou)
    {
        init();
    }
    void init()
    {
        try
        {
            module = torch::jit::load(model_file);
            module.to(device);
            std::cout << "device:" << device << std::endl;
        }
        catch (c10::Error &e)
        {
            std::cerr << "error loading the model\n";
            exit(-1);
        }
    }

    void set_batchsize(int batchsize); //: batch_size(batchsize){}
    void set_channelsize(int channelsize); // : channel_size(channelsize) {}
    void set_size(int w_, int h_); // : w(w), h(h) {}
    void set_cfdthresh(float cfdthreshold); // : cfd_threshold(cfdthreshold) {}
    void set_nmsthresh(float nmsthreshold); // : cfd_threshold(nmsthreshold) {}
    void set_diou(bool diou);
    int get_batchsize(); //: batch_size(batchsize){}
    int get_channelsize(); // : channel_size(channelsize) {}
    std::tuple<int, int> get_size(); // : w(w), h(h) {}
    float get_cfdthresh(); // : cfd_threshold(cfdthreshold) {}
    float get_nmsthresh(); // : cfd_threshold(nmsthreshold) {}
    bool get_diou();



    // 加载图片
    int load_image(const string &path, cv::Mat& img, cv::Mat& img0);

    // TensorMat转数组
    // int tensor2mat();

    int forward(cv::Mat& img);
    int draw(const cv::Mat& img, const string &savepath);
    
};

#endif //__DETECTOR_HPP__
