#pragma once
#include "Header.h"
#include<string>
#include<cstring>
#include<string.h>
#include <msclr\marshal.h>
#include<msclr\marshal_windows.h>
#include<msclr\marshal_cppstd.h>

namespace Project16 {
 using namespace System;
 using namespace System::ComponentModel;
 using namespace System::Collections;
 using namespace System::Windows::Forms;
 using namespace System::Data;
 using namespace System::Drawing;
 /// <summary>
 /// Summary for MyForm
 /// </summary>
 public ref class MyForm : public System::Windows::Forms::Form
 {
 public:
    MyForm(void)
    {
       InitializeComponent();
       //
       //TODO: Add the constructor code here
       //
    }
 protected:
  /// <summary>
  /// Clean up any resources being used.
  /// </summary>
  ~MyForm()
  {
   if (components)
   {
    delete components;
   }
  }
 private: System::Windows::Forms::Button^  button1;
 protected: 
 private: System::Windows::Forms::RichTextBox^  richTextBox1;
 private:
  /// <summary>
  /// Required designer variable.
  /// </summary>
  System::ComponentModel::Container ^components;
#pragma region Windows Form Designer generated code
  /// <summary>
  /// Required method for Designer support - do not modify
  /// the contents of this method with the code editor.
  /// </summary>
  void InitializeComponent(void)
  {
   this->button1 = (gcnew System::Windows::Forms::Button());
   this->richTextBox1 = (gcnew System::Windows::Forms::RichTextBox());
   this->SuspendLayout();
   // 
   // button1
   // 
   this->button1->Location = System::Drawing::Point(173, 252);
   this->button1->Name = L"button1";
   this->button1->Size = System::Drawing::Size(146, 48);
   this->button1->TabIndex = 0;
   this->button1->Text = L"Predict";
   this->button1->UseVisualStyleBackColor = true;
   this->button1->Click += gcnew System::EventHandler(this, &MyForm::button1_Click);
   // 
   // richTextBox1
   // 
   this->richTextBox1->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 12, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point, 
    static_cast<System::Byte>(0), true));
   this->richTextBox1->Location = System::Drawing::Point(147, 136);
   this->richTextBox1->Name = L"richTextBox1";
   this->richTextBox1->Size = System::Drawing::Size(206, 48);
   this->richTextBox1->TabIndex = 1;
   this->richTextBox1->Text = L"";
   this->richTextBox1->TextChanged += gcnew System::EventHandler(this, &MyForm::richTextBox1_TextChanged);
   // 
   // MyForm
   // 
   this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
   this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
   this->ClientSize = System::Drawing::Size(508, 412);
   this->Controls->Add(this->button1);
   this->Controls->Add(this->richTextBox1);
   this->Name = L"MyForm";
   this->Text = L"Cancer Detector";
   this->Load += gcnew System::EventHandler(this, &MyForm::MyForm_Load);
   this->ResumeLayout(false);
  }
#pragma endregion
 private: System::Void button1_Click(System::Object^  sender, System::EventArgs^  e) {
    int i,j,k;
 int numFeatures=13;
 int testSetSize=37;
 int trainingSetSize=110;
 vector<string> trainingImgName;
 vector<string> testImgNames;
 for(i=0;i<trainingSetSize;i++){
  string str="img2 (";
  str.append(to_string(i+1));
  str.append(").bmp");
  trainingImgName.push_back(str);
 }
 for(i=0;i<testSetSize;i++){
  string str="img5 (";
  str.append(to_string(i+1));
  str.append(").jpg");
  testImgNames.push_back(str);
 }
 cancerDetect detector(trainingSetSize,numFeatures,testSetSize);
 string yTrainFileName="trainY.txt";//set this
 string yTestFileName="yTestActual.txt";
 //detector.setC();
 //detector.setGamma();
 detector.setTermCriteria();
 detector.setType();
 detector.setKernel();
 detector.setTrainingYLabels(yTrainFileName);
 //detector.loadTrainingImages(trainingImgName);
 //detector.train();
 detector.loadTrainedModel();
 //printf("Training Accuracy %lf\n",detector.trainingAccuracy());
 //detector.loadTestImages(testImgNames);
 //detector.setTestYLabels(yTestFileName);
 //printf("Test Accuracy %lf\n",detector.testAccuracy());
 //float x= detector.predictIndividual("img2 (35).bmp");//image name as string should be passed here
 //System::String ^  in=System::Convert::ToString("Cancerous");
 //MessageBox::Show(in);
 float x =-1.0;
 OpenFileDialog ^ openFileDialog3;
 openFileDialog3 = gcnew OpenFileDialog();
 System::IO::Stream^ myStream;
 
 if(openFileDialog3->ShowDialog()==System::Windows::Forms::DialogResult::OK){
  
  System::IO::StreamReader ^sr =gcnew
   System::IO::StreamReader(openFileDialog3->FileName);
  if( (myStream=openFileDialog3->OpenFile()) !=nullptr){
   myStream->Close();
  }
  sr->Close();
  msclr::interop::marshal_context context;
  std::string standardString=context.marshal_as<std::string>(openFileDialog3->FileName);
  x= detector.predictIndividual(standardString); 
 }

 if(x>0.5){
  richTextBox1->Text="Cancerous";
 }
 else{
  richTextBox1->Text="Non-Cancerous";
 }
 for(int i=0;i<testSetSize;i++)
 {
  //detector.predictIndividual(testImgNames[i]);
 }
    }
 private: System::Void MyForm_Load(System::Object^  sender, System::EventArgs^  e) {
    }
private: System::Void richTextBox1_TextChanged(System::Object^  sender, System::EventArgs^  e) {
   }
};
}
