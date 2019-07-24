#include <vector>
#include <stdio.h>
#include <stdio.h>
#include "opencv\cv.h"
#include "opencv2//ml//ml.hpp"
#include "opencv2//imgproc/imgproc.hpp"
#include "opencv2//highgui//highgui.hpp"
#include "opencv2//imgproc//imgproc.hpp"
#include <opencv2//core/mat.hpp>

using namespace std;
using namespace cv;
/*Input:Takes as argument the GLCM matrix of an image
*Output:Returns the energy by using formula mentioned in Haralick's Research Paper
*/
static double calcEnergy(vector<vector<double>> &arr){
    int i,j;
    double ans=0.0;
    
    int n=(int)arr.size();
    
    for(i=0;i<n;i++){
        for(j=0;j<n;j++){
            ans+=pow((arr[i][j]),2);
        }
    }
    return ans;
    
}
/*Input:Takes as argument the GLCM matrix of an image
*Output:Returns the entropy by using formula mentioned in Haralick's Research Paper
*/
static double calcEntropy(vector<vector<double>> &arr){
    int i,j;
    double ans=0.0;
    
    int n= (int)arr.size();
    
    for(i=0;i<n;i++){
        for(j=0;j<n;j++){
            if(arr[i][j]==0.0)
                continue;
            ans+=-(arr[i][j])*(log(arr[i][j]));
        }
    }
    return ans;
}
/*Input:Takes as argument the GLCM matrix of an image
*Output:Returns the Homogenity by using formula mentioned in Haralick's Research Paper
*/
static double calcHomogenity(vector<vector<double>> &arr){
    int i,j;
    double ans=0.0;
    int n=(int)arr.size();
    
    for(i=0;i<n;i++){
        for(j=0;j<n;j++){
            ans+=(arr[i][j]/(1+abs(i-j)));
        }
    }
    return ans;
}
/*Input:Takes as argument the GLCM matrix of an image
*Output:Returns the contrast by using formula mentioned in Haralick's Research Paper
*/
static double calcContrast(vector<vector<double>> &arr){
    int i,j;
    double ans=0.0;
    
    int n=(int)arr.size();
    
    for(i=0;i<n;i++){
        for(j=0;j<n;j++){
            ans+=pow(i-j,2)*arr[i][j];
        }
    }
    return ans;
}
/*Input:Takes as argument the GLCM matrix of an image
*Output:Returns the correlation by using formula mentioned in Haralick's Research Paper
*/
static double calcCorrelation(vector<vector<double>> &arr){
    int i,j;

    double rowMean=0.0;
    double colMean=0.0;
    double rowStd=0.0;
    double colStd=0.0;
    int n=(int)arr.size();
    double var1=0.0;
    for(i=0;i<n;i++){
        var1=0.0;
        for(j=0;j<n;j++){
            var1+=arr[i][j];
        }
        var1*=i;
        rowMean+=var1;
    }
    //rowMean calculated

    for(j=0;j<n;j++){
        var1=0.0;
        for(i=0;i<n;i++){
            var1+=arr[i][j];
        }
        var1*=j;
        colMean+=var1;
    }
    //colMean calculated

    for(i=0;i<n;i++){
        var1=0.0;
        for(j=0;j<n;j++){
            var1+=arr[i][j]*pow(i-rowMean,2);
        }
        rowStd+=var1;
    }
    rowStd=sqrt(rowStd);
    //rowStd calculated

    for(i=0;i<n;i++){
        var1=0.0;
        for(j=0;j<n;j++){
            var1+=arr[j][i]*pow(j-colMean,2);
        }
        colStd+=var1;
    }
    colStd=sqrt(colStd);
    double ans=0.0;

    for(i=0;i<n;i++){
        for(j=0;j<n;j++){
            ans+=arr[i][j]*(i-rowMean)*(j-colMean);
        }
    }
    ans=ans/(rowStd*colStd);
    return ans;
}
/*Input:Takes as argument the GLCM matrix of an image
*Output:Returns the variance by using formula mentioned in Haralick's Research Paper
*/
static double calcVariance(vector<vector<double>> &arr){
 int i,j;
 int n=arr.size();
 double glcmSumHor=0.0;
 for(i=0;i<n;i++){
  for(j=0;j<n;j++){
   glcmSumHor+=arr[i][j];
  }
 }
 //optimization possible here
 double var1=0.0;
 for(i=0;i<n;i++){
  for(j=0;j<n;j++){
   var1+=arr[i][j];
  }
 }
 var1/=(n*n);
 double mean=var1*glcmSumHor;
 double ans=0.0;
 for(i=0;i<n;i++){
  var1=0.0;
  for(j=0;j<n;j++){
   var1+=(i+1-mean)*(i+1-mean)*arr[i][j];
  }
  ans+=var1;
 }
 return ans;
}
/*Input:Takes as argument the GLCM matrix of an image
*Output:Returns the pX-Y vector by using formula mentioned in Haralick's Research Paper
pX-Y is used to calculate some haralick features as mentioned in Haralick's Research paper
*/
static vector<double> p_XminusY(vector<vector<double>> &arr){
 int i,j,k;
 int n=arr.size();
 double var1=0.0;
 vector<double> diagSum;
 i=0;
 j=0;
 while(i<n&&j<n){
  var1+=arr[i][j];
  i++;
  j++;
 }
 diagSum.push_back(var1);
 for(k=1;k<n;k++){
  i=0;j=0;
  var1=0.0;
  while((i+k)<n&&j<n){
   var1+=arr[i+k][j];
   i++;
   j++;
  }
  i=0;
  j=0;
  while(i<n&&(j+k)<n){
   var1+=arr[i][j+k];
   i++;
   j++;
  }
  diagSum.push_back(var1);
 }
 return diagSum;
}
/*Input:Takes as argument the GLCM matrix of an image
*Output:Returns the pX+Y by using formula mentioned in Haralick's Research Paper
*pX+Y is used to calculate some features as mentioned in Haralick's Research Paper
*/
static vector<double> p_XplusY(vector<vector<double>> &arr){
 int i,j,k;
 int n=arr.size();
 double var1=0.0;
 vector<double> diagSum(2*n-1);
 i=0;j=n-1;
 while(i<n&&j>-1){
  var1+=arr[i][j];
  i++;
  j--;
 }
 diagSum[n-1]=var1;
 for(k=1;k<n;k++){
  i=0;j=n-1;
  var1=0.0;
  while((i+k)<n&&j>-1){
   var1+=arr[i+k][j];
   i++;
   j--;
  }
  diagSum[n-1+k]=var1;
  i=0;j=0;var1=0.0;
  while(i<n&&(j+k-1)>-1){
   var1+=arr[i][j+k-1];
   i++;
   j--;
  }
  diagSum[k-1]=var1;
 }
 return diagSum;
}
/*Input:Takes as argument the GLCM matrix of an image
*Output:Returns the sum average by using formula mentioned in Haralick's Research Paper
*/
static double calcSumAverage(vector<vector<double>> &arr){
 int i,j,k;
 double ans=0.0;
 int n=arr.size();
 vector<double> pXplusY1;
 pXplusY1=p_XplusY(arr);
 for(i=2;i<=2*n;i++){
  ans+=i*pXplusY1[i-2];
 }
 return ans;
}
/*Input:Takes as argument the GLCM matrix of an image
*Output:Returns the sum entropy by using formula mentioned in Haralick's Research Paper
*/
static double calcSumEntropy(vector<vector<double>> &arr){
 int i,j;
 vector<double> brr;
 brr=p_XplusY(arr);
 int n=arr.size();
 double ans=0.0;
 for(i=0;i<brr.size();i++){
  if(brr[i]==0){
   continue;
  }
  ans+=(brr[i])*(log(brr[i]));
 }
 return -1*ans;
}
/*Input:Takes as argument the GLCM matrix of an image
*Output:Returns the difference Variance by using formula mentioned in Haralick's Research Paper
*/
static double calcDifferenceVariance(vector<vector<double>> &arr){
 int i,j;
 double ans=0.0;
 double mean=0.0;
 vector<double> brr;
 brr=p_XminusY(arr);
 for(i=0;i<brr.size();i++){
  mean+=brr[i];
 }
 mean/=brr.size();
 for(i=0;i<brr.size();i++){
  ans+=(pow(i-mean,2))*brr[i];
 }
 return ans;
}
/*Input:Takes as argument the GLCM matrix of an image
*Output:Returns the difference entropy by using formula mentioned in Haralick's Research Paper
*/
static double calcDifferenceEntropy(vector<vector<double>> &arr){
 int i,j;
 double ans=0.0;
 vector<double> brr;
 brr= p_XminusY(arr);
 for(i=0;i<brr.size();i++){
  if(brr[i]!=0){
   ans+=(brr[i])*log(brr[i]);
  }
 }
 return -1*ans;
}
/*Input:Takes as argument the GLCM matrix of an image
*Output:Returns the Information Measure of Correlation I feature by using formula mentioned in Haralick's Research Paper
*/
static double calcInformationMeasureofCorrelationI(vector<vector<double>> &arr){
    int i,j,k;
    int n=arr.size();
    vector<double> rowCoOcMat(n,0);
    vector<double> colCoOcMat(n,0);
    vector<vector<double>> logrc(n);
    vector<double> logc(n);
    vector<double> logr(n);
    double HXY1=0.0;
    double numerator=0.0;
    double HX=0.0;
    double HY=0.0;
    double denominator=0.0;
    for(i=0;i<n;i++){
        for(j=0;j<n;j++){
            rowCoOcMat[i]+=arr[i][j];
            colCoOcMat[i]+=arr[j][i];
        }
    }
    for(i=0;i<n;i++){
        for(j=0;j<n;j++){
            logrc[i].push_back(rowCoOcMat[i]*colCoOcMat[j]);
            if(logrc[i][j]==0){
                continue;
            }
            else{
                logrc[i][j]=log(logrc[i][j]);//log2 required here
            }
        }
    }
    for(i=0;i<n;i++){
        for(j=0;j<n;j++){
            HXY1+=-1*arr[i][j]*logrc[i][j];
        }
    }
    numerator=calcEntropy(arr)-HXY1;//optimization possible
    //logc calculation
    for(i=0;i<n;i++){
        if(colCoOcMat[i]==0){
            logc[i]=0;
            continue;
        }
        else{
            logc[i]=log(colCoOcMat[i]);//log2 required here
        }
    }
    //HX calculation
    for(i=0;i<n;i++){
        HX+=-1*colCoOcMat[i]*logc[i];
    }
    //logr calculation
    for(i=0;i<n;i++){
        if(rowCoOcMat[i]==0){
            logr[i]=0;
            continue;
        }
        else{
            logr[i]=log(rowCoOcMat[i]);//log2 required here
        }
    }
    //HY calculation
    for(i=0;i<n;i++){
        HY+=-1*rowCoOcMat[i]*logr[i];
    }
    denominator=max(HX,HY);
    return numerator/denominator;
}
/*Input:Takes as argument the GLCM matrix of an image
*Output:Returns the Information Measure of Correlation II feature by using formula mentioned in Haralick's Research Paper
*/
static double calcInformationMeasureofCorrelationII(vector<vector<double>> &arr){
 int i,j,k;
 int n=arr.size();
 double HXY2=0.0;
 vector<double> rowCoOcMat(n,0);
 vector<double> colCoOcMat(n,0);
 vector<vector<double>> logrc(n);
 //prerequisites for calculation
 for(i=0;i<n;i++){
  for(j=0;j<n;j++){
   rowCoOcMat[i]+=arr[i][j];
   colCoOcMat[i]+=arr[j][i];
  }
 }
 //calculating logrc matrix
 for(i=0;i<n;i++){
  for(j=0;j<n;j++){
   logrc[i].push_back(rowCoOcMat[i]*colCoOcMat[j]);
   if(logrc[i][j]==0){
    continue;
   }
   else{
    logrc[i][j]=log(logrc[i][j]);//log2 required here
   }
  }
 }
 //HXY2 calculation
 for(i=0;i<n;i++){
  for(j=0;j<n;j++){
   HXY2+=-1*rowCoOcMat[i]*colCoOcMat[j]*logrc[i][j];
  }
 }
 return sqrt((1-exp(-2*HXY2-calcEntropy(arr))));//optimization available
}
/*Input:Takes as argument the GLCM matrix of an image
*Output:Returns the sumVariance Feature by using formula mentioned in Haralick's Research Paper
*/
static double calcSumVariance(vector<vector<double>> &arr){
 int i,j,k;
 int n=arr.size();
 double ans=0.0;
 vector<double> p_XplusY1;
 p_XplusY1=p_XplusY(arr);
 double var1=calcInformationMeasureofCorrelationI(arr);
 for(i=0;i<2*n-1;i++){
  ans+=pow(i+2-var1,2)*p_XplusY1[i];
 }
 return ans;
}
/*Input:Takes an image as an input
Output:Returnas a vector that has 13 haralick features is the following order
1.Energy
2.Contrast
3.Correlation
*4.variance
5.Homogenity
6.Sum Average
7.Sum Variance
8.Sum Entropy
9.Entropy
10.Differnece Variance
11.Differnce Entropy
12.Information Measure of Correlation I
13.Information Measure of Correlation II
Before calculating the GLCM matrix the function first converts the image to grey scale.
Then it performs scaling of pixel intensities and brings them between 0-7(including 0 and 7 both)
*/
static vector<double> glcm(Mat &img){
    int i,j;
    int n=-1;
    n=(int)pow(2,3);
    vector<vector<double>> glcm0(n);
    
    //glcm0 calculations start
    for(i=0;i<n;i++){
        for(j=0;j<n;j++){
            glcm0[i].push_back(0);
        }
    }
 //scaling glcmmatrix from 256 to 8
 for(i=0;i<img.rows;i++){
  for(j=0;j<img.cols;j++){
   if(img.at<uchar>(i,j)>=0&&img.at<uchar>(i,j)<32){
    img.at<uchar>(i,j)=0;
   }
   else if(img.at<uchar>(i,j)>=32&&img.at<uchar>(i,j)<64){
    img.at<uchar>(i,j)=1;
   }
   else if(img.at<uchar>(i,j)>=64&&img.at<uchar>(i,j)<96){
    img.at<uchar>(i,j)=2;
   }
   else if(img.at<uchar>(i,j)>=96&&img.at<uchar>(i,j)<128){
    img.at<uchar>(i,j)=3;
   }
   else if(img.at<uchar>(i,j)>=128&&img.at<uchar>(i,j)<160){
    img.at<uchar>(i,j)=4;
   }
   else if(img.at<uchar>(i,j)>=160&&img.at<uchar>(i,j)<192){
    img.at<uchar>(i,j)=5;
   }
   else if(img.at<uchar>(i,j)>=192&&img.at<uchar>(i,j)<224){
    img.at<uchar>(i,j)=6;
   }
   else {
    img.at<uchar>(i,j)=7;
   }
  }
 }
    //scaling over
    for(i=0;i<img.rows;i++){
        for(j=0;j<img.cols-1;j++){
   glcm0[img.at<uchar>(i,j)][img.at<uchar>(i,j+1)]+=1.0;
        }
    }
 //calculating matrix sum for normalization of glcm0
 double glcmSumHor=0.0;
 for(i=0;i<n;i++){
  for(j=0;j<n;j++){
   glcmSumHor+=glcm0[i][j];
  }
 } 
 //normalization of glcm0
 for(i=0;i<n;i++){
  for(j=0;j<n;j++){
   glcm0[i][j]/=glcmSumHor;
  }
 }
    vector<double> ans;
  ans.push_back(calcEnergy(glcm0));
 ans.push_back(calcContrast(glcm0));
 ans.push_back(calcCorrelation(glcm0));
 ans.push_back(calcVariance(glcm0));
 ans.push_back(calcHomogenity(glcm0));
 ans.push_back(calcSumAverage(glcm0));
 ans.push_back(calcSumVariance(glcm0));
 ans.push_back(calcSumEntropy(glcm0));
 ans.push_back(calcEntropy(glcm0));
 ans.push_back(calcDifferenceVariance(glcm0));
 ans.push_back(calcDifferenceEntropy(glcm0));
 ans.push_back(calcInformationMeasureofCorrelationI(glcm0));
 ans.push_back(calcInformationMeasureofCorrelationII(glcm0));
    return ans;
}

static class cancerDetect
{
 int trainingSetSize;
 int numFeatures;
 int testSetSize;
 vector<int> yTrainLabel;
 vector<int> yTestActual;
 vector<vector<double>> trainingSetFeatures;
 vector<vector<float>> testSetFeatures;
 CvSVMParams params;
 CvSVM SVM;
 
public:
 /*
  Below is the constructor for the cancerDetect class
  Input:
  1. Size of training set =number of images used for training and model generation purpose
  2. number of Features=13(Always) This is kept so that if you want to add or remove features you have the liberty to do so
  3. test set size =1(Always)
 
 */
 cancerDetect(int trainingSetSize,int numFeatures,int testSetSize){
  this->trainingSetSize=trainingSetSize;
  this->numFeatures=numFeatures;
  this->testSetSize=testSetSize;
  trainingSetFeatures.resize(trainingSetSize);
  testSetFeatures.resize(testSetSize);
  params=CvSVMParams();
  SVM=CvSVM();
 }
 void setTrainingYLabels(string fileName){
  FILE *fptr;
  fptr=fopen(fileName.c_str(),"r");
  if(fptr==NULL){
   printf("Test Y Labels file NOT FOUND\n");
   system("pause");
   exit(0);
  }
  else{
   for(int i=0;i<trainingSetSize;i++){
    int x;
    fscanf(fptr,"%d\n",&x);
    yTrainLabel.push_back(x);
   } 
  }
  fclose(fptr);
 }
 void normalizeTraining(){
  int i,j;
  vector<double> minVal(numFeatures,0.0);
  vector<double> maxVal(numFeatures,0.0);
  for(i=0;i<numFeatures;i++){
   for(j=0;j<trainingSetSize;j++){
    if(minVal[i]>trainingSetFeatures[j][i]){
     minVal[i]=trainingSetFeatures[j][i];
    }
    if(maxVal[i]<trainingSetFeatures[j][i]){
     maxVal[i]=trainingSetFeatures[j][i];
    }
   }
  }
  //normalizing
  for(i=0;i<trainingSetSize;i++){
   for(j=0;j<numFeatures;j++){
    trainingSetFeatures[i][j]=(trainingSetFeatures[i][j]-minVal[j])/(maxVal[j]-minVal[j]);
   } 
  } 
 }
 Mat getTrainingYLabels(){
  Mat yLabelTrainingMat(trainingSetSize,1,CV_32SC1);
  for(int i=0;i<yTrainLabel.size();i++){
   yLabelTrainingMat.at<int>(i,0)=yTrainLabel[i];
  }
  return yLabelTrainingMat;
 }
 void loadTrainingImages(vector<string> imgName){
  for(int i=0;i<trainingSetSize;i++){
   printf("Loading Training Image Number %d\n",i+1);
   Mat img=imread(imgName[i],CV_LOAD_IMAGE_GRAYSCALE);
   if(img.data==NULL){
    printf("Training image not found\n");
    system("pause");
    system("exit");
   }
   vector<double> glmcFeatures=glcm(img);
   for(int j=0;j<glmcFeatures.size();j++){
    trainingSetFeatures[i].push_back(glmcFeatures[j]);
   }
  }
  normalizeTraining();
 }
 Mat getTrainingFeatures(){
  Mat trainingSetFeaturesMat(trainingSetSize,numFeatures,CV_32FC1);
  for(int i=0;i<trainingSetFeatures.size();i++){
   for(int j=0;j<trainingSetFeatures[i].size();j++){
     trainingSetFeaturesMat.at<float>(i,j)=(float)trainingSetFeatures[i][j];
   }
  }
  return trainingSetFeaturesMat;
 }
 void setC(int c=100000){
  params.C=c;
 }
 
 void setGamma(int gamma=10){
  params.gamma=gamma;
 }
 void setKernel(){
  params.svm_type = CvSVM::C_SVC;
 }
 void setType(){
  params.svm_type=CvSVM::LINEAR;
 }
 void setTermCriteria(long int maxIter=1000000,double epsilon=1e-6){
  params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, maxIter, epsilon);
 }
 void train(){
  printf("Training Started\n");
  Mat trainingSetFeaturesMat=getTrainingFeatures();
  Mat yLabelTrainingMat=getTrainingYLabels();
     SVM.train_auto(trainingSetFeaturesMat, yLabelTrainingMat,Mat(),Mat(),params);
  SVM.save("SVMTrainedModel.xml");
 }
 vector<int> predictTraining(){
  printf("Output Prediction Started\n");
  Mat output;
  vector<int> yPredicted;
  Mat trainingSetFeaturesMat=getTrainingFeatures();
  SVM.predict(trainingSetFeaturesMat,output);
  for(int i=0;i<output.rows;i++){
   for(int j=0;j<output.cols;j++){
    yPredicted.push_back((int)output.at<float>(i,j));
   }
  }
  return yPredicted;
 }
 /*
  Input: Filename of the trained model
  Note: the trained model mus be present in the current directory of the source code file
  Output: give a trained model if file is available and is not corrupted.
  When the trained model is not found,it prints not found and safely exits
  Errors: 
  1. When the trained model found is corrupted, it throws expection
 
 */
 void loadTrainedModel(string fileName="SVMTrainedModel.xml"){
  FILE * fptr;
  fptr = fopen(fileName.c_str(),"r");
  if(fptr==NULL){
   printf("Trained Model Not Found\n");
   system("pause");
   exit(0);
  }
  SVM.load(fileName.c_str());
 }
 float predictIndividual(string fileName){
  Mat img=imread(fileName,CV_LOAD_IMAGE_GRAYSCALE);
   if(img.empty()==true){
    printf("Individual Test image NOT FOUND\n");
    system("pause");
    exit(0);
   }
   vector<double> glcmFeatures=glcm(img);
   Mat testSetFeaturesMat(1,numFeatures,CV_32FC1);
   //double minVal[]={0.116540,0.011078,0.883495,12.118193,0.906269,6.714640,59.611940,1.156459,1.169301,0.023933,0.060896,-0.949098,0.997657};
   double minVal[]={0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.949098,0.0};
   double maxVal[]={0.398504,0.277893,0.995990,36.121505,0.994461,11.786890,166.655167,2.349777,2.485299,0.247313,0.545609,0.0,0.999983};
   for(int i=0;i<numFeatures;i++){
    testSetFeaturesMat.at<float>(0,i)=(float)((glcmFeatures[i]-minVal[i])/(maxVal[i]-minVal[i]));//changed here 1,i initially
   }
   Mat output;
   float x= SVM.predict(testSetFeaturesMat);
   return x; 
 }
};