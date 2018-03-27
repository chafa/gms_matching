// GridMatch.cpp : Defines the entry point for the console application.

//#define USE_GPU

#include "Header.h"
#include "gms_matcher.h"

void GmsMatch(Mat &img1, Mat &img2);

void runImagePair(){
    Mat img1 = imread("../data/nn_left.jpg");
    Mat img2 = imread("../data/nn_right.jpg");

    imresize(img1, 480);
    imresize(img2, 480);

    GmsMatch(img1, img2);
}


int main()
{

    std::string best_matche, j, out_capt_time, in_capt_time;
    int hN, mN, sN, hI, mI, sI, hD, mD, sD, TimeDifference, matches(0), max_matches(0);
    char data[256];

    FILE *f;

    // get capture time of the output image

    f=popen("date |cut -d ' ' -f5","r");
    if(fgets(data,256,f)!=NULL) date1.append(data);pclose(f);
    hN=std::stoi(date1.substr(0,2));
    mN=std::stoi(date1.substr(3,2));
    sN=std::stoi(date1.substr(6,2));

#ifdef USE_GPU
    int flag = cuda::getCudaEnabledDeviceCount();
    if (flag != 0){ cuda::setDevice(0); }
#endif // USE_GPU



    runImagePair();

    return 0;
}


void GmsMatch(Mat &img1, Mat &img2){
    vector<KeyPoint> kp1, kp2;
    Mat d1, d2;
    vector<DMatch> matches_all, matches_gms;

    Ptr<ORB> orb = ORB::create(10000);
    orb->setFastThreshold(0);

    if(img1.rows * img1.cols > 480 * 640 ){
        orb->setMaxFeatures(100000);
        orb->setFastThreshold(5);
    }

    orb->detectAndCompute(img1, Mat(), kp1, d1);
    orb->detectAndCompute(img2, Mat(), kp2, d2);

#ifdef USE_GPU
    GpuMat gd1(d1), gd2(d2);
    Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
    matcher->match(gd1, gd2, matches_all);
#else
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(d1, d2, matches_all);
#endif

    // GMS filter
    int num_inliers = 0;
    std::vector<bool> vbInliers;
    gms_matcher gms(kp1,img1.size(), kp2,img2.size(), matches_all);
    num_inliers = gms.GetInlierMask(vbInliers, false, false);

    cout << "Get total " << num_inliers << " matches." << endl;

    // draw matches
    for (size_t i = 0; i < vbInliers.size(); ++i)
    {
        if (vbInliers[i] == true)
        {
            matches_gms.push_back(matches_all[i]);
        }
    }

    Mat show = DrawInlier(img1, img2, kp1, kp2, matches_gms, 1);
    imshow("show", show);
    waitKey();
}


