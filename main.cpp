// GridMatch.cpp : Defines the entry point for the console application.

//#define USE_GPU

#include "Header.h"
#include "gms_matcher.h"

void GmsMatch(Mat &img1, Mat &img2);

void runImagePair(){

    std::string best_matche, j, out_capt_time, in_capt_time;
    int hN, mN, sN, hI, mI, sI, hD, mD, sD, TimeDifference, matches(0), max_matches(0);
    char data[256];

    FILE *f;

    // get capture time of the output image
string date1;
    f=popen("date |cut -d ' ' -f5","r");
    if(fgets(data,256,f)!=NULL) out_capt_time.append(data);pclose(f);
    hN=std::stoi(out_capt_time.substr(0,2));
    mN=std::stoi(out_capt_time.substr(3,2));
    sN=std::stoi(out_capt_time.substr(6,2));

    //creation d'un fichier contenant tous les noms des images déjà enregistrées

    system("ls -1 /home/halim/darknet-3/darknet/img/*.jpg | cut -d/ -f7 | sort -n>/home/halim/darknet-3/darknet/img/ImageNames.txt");
    ifstream monflux("/home/halim/darknet-3/darknet/img/ImageNames.txt");
    Mat img_out = imread("/home/halim/darknet-3/darknet/img/31.jpg");

    while(getline(monflux,j))
    {
        j = "/home/lamsi/Documents/darknet/img/" + j ;
        Mat img_in = imread(j);
        imresize(img_out, 480);
        imresize(img_in, 480);

        matches = GmsMatch(img_out, img_in);
        if(matches>max_matches)
        {
            max_matches=  matches;       //recupérer le nombre maximum de matches
            best_matche = j;              //enregistrer le nom d'image avec le maximum de matches
        }
    }

    // Récupération de la date de création de l'image

    string commande="date -r " + ImageMAtch + "| cut -d ' ' -f5";
    f=popen(commande.c_str(),"r");
    if(fgets(data,256,f)!=NULL) in_capt_time.append(data);
    pclose(f);
    hI = std::stoi(in_capt_time.substr(0,2));
    mI = std::stoi(in_capt_time.substr(3,2));
    sI = std::stoi(in_capt_time.substr(6,2));

    cout << "heure actuelle est: " << hN << endl;
    cout << "minute actuelle est: " << mN << endl;
    cout << "seconde actuelle est: " << sN << endl;
    cout << " heure de création de l image est :"<< hI <<endl;
    cout << " minute de création est "<< mI << endl;
    cout << " seconde de création est "<< sI <<endl;

    cout << "----------------------------------------" << endl;
    cout << "le maximum des matches =" << max_matches<<endl;
    cout << "le nom d'image avec le plus de matches est" << ImageMAtch << endl;
    sD = sN - sI;
    if(sD<0)
    {
        sD += 60;
        mN -= 1;
    }
    mD = mN - mI;
    if(mD<0)
    {
        mD += 60;
        hN -= 1;
    }
    hD=hN-hI;
    if(hD<0)
        hD += 24;
    TimeDifference = (hD*3600) + (mD*60) + sD;

    cout << "le temps entre l'entrée et la sortie de la voiture est: " << TimeDifference << " secondes" << endl;

}


int main()
{

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


