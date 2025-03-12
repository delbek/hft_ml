#include "config.h"

class LinearRegression 
{
public:
    LinearRegression(int trainingSize, int numFeatures, double* trainingData, double* trainingLabels, 
                    double convergenceThreshold = 0.01, double learningRate = 0.1, int maxIterations = 1000);
    LinearRegression(LinearRegression&& other) = delete;
    LinearRegression& operator=(LinearRegression&& other) = delete; 
    LinearRegression(const LinearRegression& other) = delete;
    LinearRegression& operator=(const LinearRegression& other) = delete;
    ~LinearRegression();

    void fit();
    double predict(double* x);

private:
    void updateTheta();
    void initializeTheta();
    double* computeGradient();
    double mse();

private:
    int m_TrainingSize;
    int m_NumFeatures;
    double* m_TrainingData; // of size trainingSize * numFeatures
    double* m_TrainingLabels; // of size trainingSize
    double m_ConvergenceThreshold;
    double m_LearningRate;
    double* m_Theta;
    double* m_LastPredictions;
    int m_MaxIterations;
};
